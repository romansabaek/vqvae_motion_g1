# agent_dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import random

class MotionVQDataset(Dataset):
    def __init__(self, mocap_data: torch.Tensor, end_indices: list, window_size: int, mean: torch.Tensor, std: torch.Tensor):
        self.window_size = window_size
        self.mean = mean
        self.std = std

        self.motion_list = []
        start_idx = 0
        mocap_data = mocap_data.cpu()
        for end_idx in end_indices:
            self.motion_list.append(mocap_data[start_idx:end_idx + 1])
            start_idx = end_idx + 1

        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def __len__(self):
        return len(self.motion_list)

    def __getitem__(self, index):
        motion = self.motion_list[index]
        n_frames = motion.shape[0]

        if n_frames < self.window_size:
            # Original T2M-GPT padding logic
            # Duplicate last frame to match window_size
            shortage = self.window_size - n_frames
            padding = motion[-1:].repeat(shortage, 1)
            motion = torch.cat([motion, padding], dim=0)
        else:
            # Random start point and cut window_size
            random_start = random.randint(0, n_frames - self.window_size)
            motion = motion[random_start : random_start + self.window_size]

        # Z-score normalization
        motion = (motion - self.mean) / self.std

        return motion


def cycle(iterable):
    """
    Make DataLoader into an infinite iterator
    Original T2M-GPT utility function
    """
    while True:
        for x in iterable:
            yield x