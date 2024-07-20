import json
import numpy as np
import math
import csv
import random
import argparse
import torch
import os
import torch.distributed as dist

from PIL import Image
from diffusers import DDIMScheduler, DDIMInverseScheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from reward_model.eval_pickscore import PickScore
from utils.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline



device = torch.device('cuda')


def get_args():
    parser = argparse.ArgumentParser()
    parser
    parser.add_argument("--prompt_path", default='./datasets/prompt/test_unique_caption_zh.csv', type=str)
    parser.add_argument("--benchmark_type", default='pick', choices=['pick', 'draw'], type=str)
    parser.add_argument("--seed_path", default='./datasets/prompt2seed/SDXL-10-prompt2seed_pick.json', type=str)
    parser.add_argument("--output_dir_path", default='./datasets/output_SDXL_10_pick', type=str)
    parser.add_argument("--inference_step", default=10, type=int)
    parser.add_argument("--size", default=1024, type=int)
    parser.add_argument("--T_max", default=1, type=int)
    parser.add_argument("--RatioT", default=0.9, type=float)    #if RatioT==1,则退化为start point优化
    parser.add_argument("--denoising_cfg", default=5.5, type=float)
    parser.add_argument("--inversion_cfg", default=1.0, type=float)

    args =  parser.parse_args()
    return args


def load_prompt(path, seed_path, prompt_version):
    if prompt_version == 'pick':
        prompts = []
        with open(path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[1] == "caption":
                    continue
                prompts.append(row[1])

        # prompts = prompts[0:101]
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)
        prompts = tmp_prompt_list

        #seed
        with open(seed_path) as f:
            seed_list = json.load(f)

        return prompts, seed_list
    else:
        prompts = []
        with open(path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] == "Prompts":
                    continue
                prompts.append(row[0])

        prompts = prompts[0:200]
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)
        prompts = tmp_prompt_list

        #seed
        with open(seed_path) as f:
            seed_list = json.load(f)
        return prompts, seed_list


def load_pick_prompt(path):
    prompts = []
    seeds = []
    with open(path, 'r', encoding='utf-8') as file:
        content = file.readlines()
        
        for row in content:
            prompts.append(eval(row)['caption'])
            seeds.append(eval(row)['seed'])
        
    return prompts, seeds


if __name__ == '__main__':

    dtype = torch.float16
    args = get_args()

    
    # load pipe
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype,
                                                     variant='fp16',
                                                     safety_checker=None, requires_safety_checker=False).to(device)
    
    # unet = DDP(pipe.unet, device_ids=[local_rank], output_device=local_rank)

    inverse_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',
                                                             subfolder='scheduler')
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inv_scheduler = inverse_scheduler

    # load metric
    reward_model = PickScore()

    # load benchmark
    # len(prompt_list) = 100    len(seed_list) = 100
    # prompt_list, seed_list = load_prompt(args.prompt_path, args.seed_path, prompt_version=args.benchmark_type)
    prompt_list, seed_list = load_pick_prompt(
        path='/home/zhouzikai/NoiseModel/denoising-optimization-for-diffusion-models-main/datasets/pickscore/train_1000000.json'
    )


    # mkdir
    # base_dir = args.output_dir_path
    base_dir = '/home/zhouzikai/NoiseModel/denoising-optimization-for-diffusion-models-main/datasets/Output_test'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if not os.path.exists(os.path.join(base_dir,'origin')):
        os.mkdir(os.path.join(base_dir,'origin'))
    if not os.path.exists(os.path.join(base_dir,'optim')):
        os.mkdir(os.path.join(base_dir,'optim'))
    if not os.path.exists(os.path.join(base_dir,'show')):
        os.mkdir(os.path.join(base_dir,'show'))
    if not os.path.exists(os.path.join(base_dir,'latents')):
        os.mkdir(os.path.join(base_dir,'latents'))

    T_max = args.T_max
    size = args.size
    shape = (1, 4, size // 8, size // 8)
    num_steps = args.inference_step
    guidance_scale = args.denoising_cfg
    inversion_guidance_scale = args.inversion_cfg
    ratioT = args.RatioT

    before_score, after_score, positive = 0, 0, 0

    for idx, prompt in enumerate(prompt_list):
        random_seed = seed_list[idx]  # 拿到seed_list中的 seed

        np.random.seed(int(random_seed))
        torch.manual_seed(int(random_seed))
        torch.cuda.manual_seed(int(random_seed))
        generator = torch.manual_seed(random_seed)
        start_latents = torch.randn(shape, generator=generator, dtype=dtype).to(device)

        original_img = pipe(
            prompt=prompt,
            height=size,
            width=size,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            latents=start_latents).images[0]

        # original_img.save(os.path.join(base_dir, 'origin', f'{idx}.png'))

        loss_res = []
        optim_img = pipe.forward_ours(
            prompt=prompt,
            height=size,
            width=size,
            guidance_scale=guidance_scale,
            inversion_guidance_scale=inversion_guidance_scale,
            num_inference_steps=num_steps,
            latents=start_latents,
            T_max=T_max,
            index=idx,
            seed=random_seed).images[0]  # 多加了一个 index，方便存储数据

        # optim_img.save(os.path.join(base_dir, 'optim', f'{idx}.png'))

    #     # original_img = optim_img
    #     new_width = original_img.width + optim_img.width
    #     new_image = Image.new("RGB", (new_width, original_img.height))
    #     new_image.paste(original_img, (0, 0))
    #     new_image.paste(optim_img, (original_img.width, 0))
    #     # 保存拼接后的图片
    #     new_image.save(os.path.join(base_dir, 'show', f'{idx}.png'))

    #     before_rewards, original_scores = reward_model.calc_probs(prompt, original_img)
    #     after_rewards, optimized_scores = reward_model.calc_probs(prompt, optim_img)

    #     before_score += original_scores
    #     after_score += optimized_scores

    #     print(f'seed:{random_seed},  prompt:{prompt}')
    #     print(f'origin_score:{original_scores},  optim_score:{optimized_scores}')

    #     if optimized_scores > original_scores:
    #         positive += 1

    # print(f'positive ratio = {(positive / len(prompt_list))*100}%')
    # print(f'original score = {before_score / len(prompt_list)}')
    # print(f'optim score = {after_score / len(prompt_list)}')


    # -------------------------------------------------------------------------
    # this version is for generate dataset with size of 100 * 5000
    # for seed in range(80, 100):
    #     random_seed = seed_list[str(seed)]  # 拿到seed_list中的 seed
    #     for idx, prompt in enumerate(prompt_list):
    #         # create a file to record (idx, prompt, seed)
    #         with open('prompt2seed_SDXL_10_50000.txt', 'a') as f:
    #             f.write(f'{idx}, {prompt}, {random_seed}\n')

    #         # random_seed = seed_list[str(idx)]
    #         np.random.seed(int(random_seed))
    #         torch.manual_seed(int(random_seed))
    #         torch.cuda.manual_seed(int(random_seed))
    #         generator = torch.manual_seed(random_seed)
    #         start_latents = torch.randn(shape, generator=generator, dtype=dtype).to(device)

    #         original_img = pipe(
    #             prompt=prompt,
    #             height=size,
    #             width=size,
    #             guidance_scale=guidance_scale,
    #             num_inference_steps=num_steps,
    #             latents=start_latents,
    #             ratioT=ratioT).images[0]

    #         # original_img.save(os.path.join(base_dir, 'origin', f'{idx}.png'))

    #         loss_res = []
    #         optim_img = pipe.forward_ours(
    #             prompt=prompt,
    #             height=size,
    #             width=size,
    #             guidance_scale=guidance_scale,
    #             inversion_guidance_scale=inversion_guidance_scale,
    #             num_inference_steps=num_steps,
    #             latents=start_latents,
    #             T_max=T_max,
    #             index=idx,
    #             seed=random_seed).images[0]  # 多加了一个 index，方便存储数据

            # optim_img.save(os.path.join(base_dir, 'optim', f'{idx}.png'))

    #         # original_img = optim_img
    #         new_width = original_img.width + optim_img.width
    #         new_image = Image.new("RGB", (new_width, original_img.height))
    #         new_image.paste(original_img, (0, 0))
    #         new_image.paste(optim_img, (original_img.width, 0))
    #         # 保存拼接后的图片
    #         new_image.save(os.path.join(base_dir, 'show', f'{idx}.png'))

    #         before_rewards, original_scores = reward_model.calc_probs(prompt, original_img)
    #         after_rewards, optimized_scores = reward_model.calc_probs(prompt, optim_img)

    #         before_score += original_scores
    #         after_score += optimized_scores

    #         print(f'seed:{random_seed},  prompt:{prompt}')
    #         print(f'origin_score:{original_scores},  optim_score:{optimized_scores}')

    #         if optimized_scores > original_scores:
    #             positive += 1
    # print(f'positive ratio = {(positive / len(prompt_list))*100}%')
    # print(f'original score = {before_score / len(prompt_list)}')
    # print(f'optim score = {after_score / len(prompt_list)}')

    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------

    # for idx, prompt in enumerate(prompt_list):
    #     random_seed = seed_list[str(idx % len(prompt_list))]    # 拿到seed_list中的 seed
    
    #     # create a file to record (idx, prompt, seed)
    #     with open('prompt2seed_SDXL_10_100.txt', 'w') as f:
    #         f.write(f'{idx}, {prompt}, {random_seed}\n')
    
    #     random_seed = seed_list[str(idx)]
    #     np.random.seed(int(random_seed))
    #     torch.manual_seed(int(random_seed))
    #     torch.cuda.manual_seed(int(random_seed))
    #     generator = torch.manual_seed(random_seed)
    #     start_latents = torch.randn(shape, generator=generator, dtype=dtype).to(device)
    
    #     original_img = pipe(
    #         prompt=prompt,
    #         height=size,
    #         width=size,
    #         guidance_scale=guidance_scale,
    #         num_inference_steps=num_steps,
    #         latents=start_latents).images[0]
    
    #     original_img.save(os.path.join(base_dir, 'origin', f'{idx}.png'))
    
    #     loss_res = []
    #     optim_img = pipe.forward_ours(
    #         prompt=prompt,
    #         height=size,
    #         width=size,
    #         guidance_scale=guidance_scale,
    #         inversion_guidance_scale=inversion_guidance_scale,
    #         num_inference_steps=num_steps,
    #         latents=start_latents,
    #         T_max=T_max,
    #         index=idx,
    #         ratioT=ratioT).images[0]    # 多加了一个 index，方便存储数据
    
    #     optim_img.save(os.path.join(base_dir, 'optim', f'{idx}.png'))
    
    #     # original_img = optim_img
    #     new_width = original_img.width + optim_img.width
    #     new_image = Image.new("RGB", (new_width, original_img.height))
    #     new_image.paste(original_img, (0, 0))
    #     new_image.paste(optim_img, (original_img.width, 0))
    #     # 保存拼接后的图片
    #     new_image.save(os.path.join(base_dir, 'show', f'{idx}.png'))
    
    #     before_rewards, original_scores = reward_model.calc_probs(prompt, original_img)
    #     after_rewards, optimized_scores = reward_model.calc_probs(prompt, optim_img)
    
    #     before_score += original_scores
    #     after_score += optimized_scores
    
    #     print(f'seed:{random_seed},  prompt:{prompt}')
    #     print(f'origin_score:{original_scores},  optim_score:{optimized_scores}')
    
    #     if optimized_scores > original_scores:
    #         positive += 1


    # print(f'positive ratio = {(positive / len(prompt_list))*100}%')
    # print(f'original score = {before_score / len(prompt_list)}')
    # print(f'optim score = {after_score / len(prompt_list)}')
    # -----------------------------------------------------------------------------------------