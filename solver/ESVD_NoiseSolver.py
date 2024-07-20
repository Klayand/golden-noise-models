import json
import os
import math
import random
import argparse
import numpy as np
import accelerate
from torch import nn
import copy

import torch
import einops
from torch.nn.functional import mse_loss
import torch.distributed as dist
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from reward_model.eval_pickscore import PickScore
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from torch import Callable
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from model import NoiseUnet, SVDNoiseUnet
from reward_model.eval_pickscore import PickScore
from diffusers.models.normalization import AdaGroupNorm

__all__ = ['SVD_Embedding_Solver']


class SVD_Embedding_Solver:
    def __init__(
            self,
            pipeline: nn.Module,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            config=None,
            local_rank=None,

    ):
        self.config = config

        self.ddp_mode = self.config.ddp
        self.do_classifier_free_guidance = self.config.do_classifier_free_guidance
        self.guidance_scale = self.config.guidance_scale
        self.guidance_rescale = self.config.guidance_rescale
        self.pretrained = self.config.pretrained_path

        self.pipeline = pipeline

        self.conv_in = copy.deepcopy(self.pipeline.unet.conv_in)
        self.conv_in.load_state_dict(self.pipeline.unet.conv_in.state_dict().copy())
        self.out_channels = self.conv_in.out_channels
        self.in_channels = self.conv_in.in_channels

        self.unet_embedding = NoiseUnet(self.conv_in, self.in_channels, self.out_channels).to(device).to(torch.float32)
        self.unet_svd = SVDNoiseUnet().to(device).to(torch.float32)

        self._alpha = torch.Tensor([0.]).to(device).to(torch.float32)
        self._alpha.requires_grad_(True)

        self._beta = torch.Tensor([0.]).to(device).to(torch.float32)
        self._beta.requires_grad_(True)

        self.text_embedding = AdaGroupNorm(2048 * 77, 4, 1, eps=1e-6).to(device).to(torch.float32)

        if eval(self.pretrained) is not None:
            gloden_unet = torch.load(self.pretrained)
            self.unet_svd.load_state_dict(gloden_unet["unet_svd"])
            self.unet_embedding.load_state_dict(gloden_unet["unet_embedding"])
            self.text_embedding.load_state_dict(gloden_unet["embeeding"])
            self._alpha = gloden_unet["alpha"]
            self._beta = gloden_unet["beta"]
            print("Successfully Load!")

        self.alpha = 1
        self.beta = 1

        self.optimizer = torch.optim.AdamW(
            list(self.unet_embedding.parameters()) + list(self.unet_svd.parameters()) + list(
                self.text_embedding.parameters()) + [self._alpha] + [self._beta], lr=1e-4)

        self.local_rank = local_rank

        self.device = device

        # initialization
        self.init()

    def init(self):
        # change device
        self.pipeline.to(self.device)
        self.unet_svd.to(self.device)
        self.unet_embedding.train()

        # self.pipeline.unet.requires_grad_(False)
        # self.unet.load_state_dict(self.pipeline.unet.state_dict(), strict=False)
        self.unet_embedding.train()
        self.unet_svd.train()

    def train(
            self, train_loader: DataLoader, val_loader, total_epoch=5, save_path='./golden_unet'
    ):

        for epoch in range(1, total_epoch + 1):
            self.unet_svd.train()
            self.unet_embedding.train()
            self.text_embedding.train()
            train_loss = 0

            if self.ddp_mode:
                train_loader.sampler.set_epoch(epoch)

            # train
            pbar = tqdm(train_loader)
            for step, (original_noise, optimized_noise, prompt, random_seed) in enumerate(pbar, 1):
                original_noise, optimized_noise = original_noise.to(self.device), optimized_noise.to(self.device)

                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self.pipeline.encode_prompt(prompt=prompt[0], device=self.device)

                add_text_embeds = pooled_prompt_embeds

                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
                prompt_embeds = prompt_embeds.float().view(prompt_embeds.shape[0], -1)
                text_emb = self.text_embedding(original_noise.float(), prompt_embeds)

                encoder_hidden_states_svd = original_noise
                encoder_hidden_states_embedding = original_noise + text_emb

                golden_embedding = self.unet_embedding(encoder_hidden_states_embedding.float())

                golden_noise = self.unet_svd(encoder_hidden_states_svd.float()) + (
                            2 * torch.sigmoid(self._alpha) - 1) * text_emb + self._beta * golden_embedding

                loss = self.alpha * mse_loss(golden_noise, optimized_noise.float())
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % 50 == 0:
                    pbar.set_postfix_str(f"loss={train_loss / 50}")
                    train_loss = 0.

            with torch.no_grad():
                self.unet_svd.eval()
                self.unet_embedding.eval()
                self.text_embedding.eval()

                total_eval_loss = 0.
                count = 0.

                for i, (original_noise, optimized_noise, prompt, random_seed) in enumerate(val_loader):
                    original_noise, optimized_noise = original_noise.to(self.device), optimized_noise.to(self.device)

                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = self.pipeline.encode_prompt(prompt=prompt[0], device=self.device)

                    add_text_embeds = pooled_prompt_embeds

                    text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
                    prompt_embeds = prompt_embeds.float().view(prompt_embeds.shape[0], -1)
                    text_emb = self.text_embedding(original_noise.float(), prompt_embeds)

                    encoder_hidden_states_svd = original_noise
                    encoder_hidden_states_embedding = original_noise + text_emb

                    golden_embedding = self.unet_embedding(encoder_hidden_states_embedding.float())

                    golden_noise = self.unet_svd(encoder_hidden_states_svd.float()) + (
                                2 * torch.sigmoid(self._alpha) - 1) * text_emb + self._beta * golden_embedding

                    total_eval_loss += mse_loss(golden_noise, optimized_noise) * len(original_noise)
                    count += len(original_noise)

                print("Eval Loss:", round(total_eval_loss.item() * 100 / count, 2), "%")

        torch.save({"unet_embedding": self.unet_embedding.state_dict(),
                    "unet_svd": self.unet_svd.state_dict(),
                    "embeeding": self.text_embedding.state_dict(),
                    "alpha": self._alpha,
                    "beta": self._beta},
                   f"{save_path}.pth")

        return self.unet_svd, self.unet_embedding

    def generate(self,
                 latent,
                 optimized=None,
                 reward_model=None,
                 prompt=None,
                 save_postfix=None,
                 save_pic=None,
                 idx=None,
                 config=None):

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(prompt=prompt, device=self.device)

        prompt_embeds = prompt_embeds.float().view(prompt_embeds.shape[0], -1)
        text_emb = self.text_embedding(latent.float(), prompt_embeds)

        encoder_hidden_states_svd = latent
        encoder_hidden_states_embedding = latent + text_emb

        golden_embedding = self.unet_embedding(encoder_hidden_states_embedding.float())

        golden_noise = self.unet_svd(encoder_hidden_states_svd.float()) + (
                    2 * torch.sigmoid(self._alpha) - 1) * text_emb + self._beta * golden_embedding

        self.pipeline = self.pipeline.to(torch.float16)
        latent = latent.half()
        golden_noise = golden_noise.half()

        golden_img = self.pipeline(
            prompt=prompt,
            height=config.size,
            width=config.size,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.inference_step,
            latents=golden_noise).images[0]

        original_img = self.pipeline(
            prompt=prompt,
            height=config.size,
            width=config.size,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.inference_step,
            latents=latent).images[0]

        if save_pic is not None:
            base_dir = save_pic
            if not os.path.exists(base_dir):
                os.mkdir(base_dir)
            if not os.path.exists(os.path.join(base_dir, 'origin')):
                os.mkdir(os.path.join(base_dir, 'origin'))
            if not os.path.exists(os.path.join(base_dir, 'optim')):
                os.mkdir(os.path.join(base_dir, 'optim'))
            if not os.path.exists(os.path.join(base_dir, 'show')):
                os.mkdir(os.path.join(base_dir, 'show'))

            original_img.save(os.path.join(base_dir, 'origin', f'{idx}.png'))
            golden_img.save(os.path.join(base_dir, 'optim', f'{idx}.png'))

            new_width = original_img.width + golden_img.width
            new_image = Image.new("RGB", (new_width, original_img.height))
            new_image.paste(original_img, (0, 0))
            new_image.paste(golden_img, (original_img.width, 0))
            # 保存拼接后的图片
            new_image.save(os.path.join(base_dir, 'show', f'{idx}.png'))

        if optimized is not None:
            optimized_img = self.pipeline(
                prompt=prompt,
                height=config.size,
                width=config.size,
                guidance_scale=config.guidance_scale,
                num_inference_steps=config.inference_step,
                latents=optimized).images[0]

            if save_postfix is not None:
                optimized_img.save(f'golden_img_optimized_{save_postfix}.png')

            if config.metric_version == 'PickScore':
                after_rewards, optimized_scores = reward_model.calc_probs(prompt, optimized_img)
            elif config.metric_version == 'HPSv2':
                optimized_scores = reward_model.score([optimized_img], prompt, hps_version="v2.1")[0]
            elif config.metric_version == 'ImageReward':
                optimized_scores = reward_model.score(prompt, optimized_img)
            elif config.metric_version == 'AES':
                optimized_scores = reward_model(optimized_img)

        if config.metric_version == 'PickScore':
            before_rewards, original_scores = reward_model.calc_probs(prompt, original_img)
            golden_rewards, golden_scores = reward_model.calc_probs(prompt, golden_img)
        elif config.metric_version == 'HPSv2':
            original_scores = reward_model.score([original_img], prompt, hps_version="v2.1")[0]
            golden_scores = reward_model.score([golden_img], prompt, hps_version="v2.1")[0]
        elif config.metric_version == 'ImageReward':
            original_scores = reward_model.score(prompt, original_img)
            golden_scores = reward_model.score(prompt, golden_img)
        elif config.metric_version == 'AES':
            original_scores = reward_model(original_img)
            golden_scores = reward_model(golden_img)

        if save_postfix is not None:
            golden_img.save(f'golden_img_news_{save_postfix}.png')
            original_img.save(f'golden_img_originals_{save_postfix}.png')

        print(f'prompt:{prompt}')
        if optimized is not None:
            print(f'origin_score:{original_scores},  optim_score:{optimized_scores}, golden_score:{golden_scores}')
        else:
            print(f'origin_score:{original_scores}, golden_score:{golden_scores}')

        return original_scores, golden_scores





