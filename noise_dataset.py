import os
import numpy as np

from torch.utils.data import DataLoader, Dataset

from utils import load_prompt, load_pick_prompt


class NoiseDataset(Dataset):
    def __init__(self,
                 prompt_version='pick',
                 all_file=False,
                 discard=False,
                 evaluate=False,
                 data_dir="/home/zhouzikai/NoiseModel/denoising-optimization-for-diffusion-models-main/datasets/noise_pairs_SDXL_10_50000",
                 prompt_path="/home/zhouzikai/NoiseModel/denoising-optimization-for-diffusion-models-main/prompt2seed_SDXL_10_50000.txt"):

        # self.prompt_file = load_prompt(prompt_path, prompt_version)
        self.prompt_file, self.seed_file = load_prompt(prompt_path, prompt_version)

        self.evaluate = evaluate

        if all_file:
            self.dir_path = [os.path.join(data_dir, path) for path in os.listdir(data_dir)]

            self.file_paths = []
            for dir in self.dir_path:
                self.file_paths.extend([os.path.join(dir, path) for path in os.listdir(dir)])
        else:
            self.file_paths = [os.path.join(data_dir, path) for path in os.listdir(data_dir)]

        self.original_noise_list = []
        self.optimized_noise_list = []
        self.prompt_list = []

        # when you need to discard bad samples, you should comment the line below
        if not discard:
            self.set_npz()

    def discard_bad_sample(self, bad_samples):
        count = 0
        pos_count = 0
        results = []
        for file in self.file_paths:
            if file not in bad_samples:
                results.append(file)
                pos_count += 1
            count += 1

        print(f"success rate {round(pos_count * 100 / count, 2)}%")
        self.file_paths = results
        self.set_npz()

    def set_npz(self, ):
        for file in self.file_paths:
            file_content = np.load(file)
            self.original_noise_list.append(file_content['arr_0'].squeeze())
            self.optimized_noise_list.append(file_content['arr_1'].squeeze())
            self.prompt_list.append(self.prompt_file[file_content['arr_2']])

    def __getitem__(self, idx):
        if self.evaluate:
            return self.original_noise_list[idx], self.optimized_noise_list[idx], self.prompt_list[idx], \
            self.file_paths[idx]
        else:
            return self.original_noise_list[idx], self.optimized_noise_list[idx], self.prompt_list[idx], self.seed_file[
                idx]
            # return self.original_noise_list[idx], self.optimized_noise_list[idx], self.prompt_list[idx]

    def __len__(self):
        return len(self.prompt_list)