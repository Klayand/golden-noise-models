import json
import os
import random
import argparse
import numpy as np
import accelerate
import torch
import torch.distributed as dist

from PIL import Image
from torch import nn
from torch.nn.functional import mse_loss
from diffusers import DDIMScheduler, DDIMInverseScheduler
from reward_model.eval_pickscore import PickScore
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from solvers import solver_dict
from noise_dataset import NoiseDataset
from reward_model.eval_pickscore import PickScore
from utils.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

DEVICE = torch.device("cuda" if torch.cuda else "cpu")


def get_args():
    parser = argparse.ArgumentParser()

    # ddp mode
    parser.add_argument("--ddp", default=False, type=bool)

    # model and dataset construction
    parser.add_argument("--model", default='svd_unet+unet',
                        choices=['unet', 'vit', 'svd_uent', 'svd_unet+unet', 'e_unet'], type=str)
    parser.add_argument("--benchmark-type", default='pick', choices=['pick', 'draw'], type=str)
    parser.add_argument("--train", default=False, type=bool)
    parser.add_argument("--test", default=True, type=bool)

    # hyperparameters
    parser.add_argument('--do-classifier-free-guidance', default=True, type=bool)
    parser.add_argument("--inference-step", default=10, type=int)
    parser.add_argument("--size", default=1024, type=int)
    parser.add_argument("--RatioT", default=0.9, type=float)
    parser.add_argument("--guidance-scale", default=5.5, type=float)
    parser.add_argument("--guidance-rescale", default=0.0, type=float)
    parser.add_argument("--all-file", default=False, type=bool)
    parser.add_argument("--evaluate", default=False, type=bool)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=16, type=int)
    parser.add_argument("--metric-version", default='PickScore', choices=['PickScore', 'HPS v2', 'AES', 'ImageReward'],
                        type=str)

    # path configuration
    parser.add_argument("--prompt-path",
                        default='/home/zhouzikai/NoiseModel/denoising-optimization-for-diffusion-models-main/prompt2seed_SDXL_10_50000.txt',
                        type=str)
    parser.add_argument("--data-dir",
                        default='/home/zhouzikai/NoiseModel/denoising-optimization-for-diffusion-models-main/datasets/noise_pairs_SDXL_10_50000',
                        type=str)
    parser.add_argument('--pretrained-path', type=str,
                        default='/home/zhouzikai/NoiseModel/denoising-optimization-for-diffusion-models-main/checkpoints')
    parser.add_argument('--save-ckpt-path', type=str,
                        default='/home/zhouzikai/NoiseModel/denoising-optimization-for-diffusion-models-main/checkpoints/')

    # discard the bad samples
    parser.add_argument("--discard", default=False, type=bool)
    parser.add_argument("--discard-path",
                        default='/home/zhouzikai/NoiseModel/denoising-optimization-for-diffusion-models-main/bad_sample_pick.json',
                        type=str)

    args = parser.parse_args()

    print("generating config:")
    print(f"Config: {args}")
    print('-' * 100)

    return args


if __name__ == '__main__':
    dtype = torch.float16
    args = get_args()

    if args.ddp:
        dist.init_process_group(backend='nccl')
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)

    # construct the diffusion models and human perference models
    reward_model = PickScore()
    pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype,
                                                         variant='fp16',
                                                         safety_checker=None, requires_safety_checker=False)
    # pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # construct the solver
    try:
        if args.ddp:
            solver = solver_dict[args.model](
                pipeline=pipeline,
                local_rank=local_rank,
                config=args
            )
        else:
            solver = solver_dict[args.model](
                pipeline=pipeline,
                config=args
            )
    except:
        print("Solver does not exist!")
        assert False

    # construct the dataset
    NoiseDataset_100 = NoiseDataset(
        discard=args.discard,
        all_file=args.all_file,
        evaluate=args.evaluate,
        data_dir=args.data_dir,
        prompt_path=args.prompt_path)

    if args.discard and args.discard_path is not None:

        bad_samples = []
        with open(args.discard_path, 'r') as file:
            content = file.readlines()

            for row in content:
                try:
                    bad_samples.append(eval(row)['path'])
                except:
                    continue
        NoiseDataset_100.discard_bad_sample(bad_samples)

    # test_noise, test_optimized_noise, prompt = NoiseDataset_100.__getitem__(10043) # 2

    from sklearn.model_selection import StratifiedShuffleSplit

    labels = [0 for i in range(len(NoiseDataset_100))]
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]

    trainset = torch.utils.data.Subset(NoiseDataset_100, train_indices)
    valset = torch.utils.data.Subset(NoiseDataset_100, valid_indices)

    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)

    else:
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    val_loader = DataLoader(valset, batch_size=50, shuffle=False)

    if args.train:
        solver.train(
            train_loader,
            val_loader,
            total_epoch=args.epochs,
            save_path=args.save_ckpt_path)

    if args.test:
        random_seed = 120
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

        latent = None
        random_latent = torch.randn(1, 4, 128, 128, dtype=dtype).to(DEVICE)

        postfix = ''

        # test_prompt = "Solid Black color"  # a black man
        # w.generate(random_latent, latent, reward_model, prompt=test_prompt, save_postfix=test_prompt.replace(" ","_")+"_svd+embedding")

        # test_prompt = "A white background"  # a black man
        # w.generate(random_latent, latent, reward_model, prompt=test_prompt, save_postfix=test_prompt.replace(" ","_")+"_svd+embedding")

        test_prompt = "Trump was shot in the right ear while speaking at a campaign rally in Pennsylvania, but extended his arm and shouted 'Fight'"  # a black man
        solver.generate(random_latent,
                        latent,
                        reward_model,
                        config=args,
                        prompt=test_prompt,
                        save_postfix=test_prompt.replace(" ", "_") + postfix)

        random_latent = torch.randn(1, 4, 128, 128, dtype=dtype).to(DEVICE)

        test_prompt = "full glass of water standing on a mountain"  # a black man
        solver.generate(random_latent,
                        latent,
                        reward_model,
                        config=args,
                        prompt=test_prompt,
                        save_postfix=test_prompt.replace(" ", "_") + postfix)
        random_latent = torch.randn(1, 4, 128, 128, dtype=dtype).to(DEVICE)

        test_prompt = "A sign that says ""Hatsune Miku es real"""  # a black man
        solver.generate(random_latent,
                        latent,
                        reward_model,
                        config=args,
                        prompt=test_prompt,
                        save_postfix=test_prompt.replace(" ", "_") + postfix)

    if args.ddp:
        dist.destroy_process_group()