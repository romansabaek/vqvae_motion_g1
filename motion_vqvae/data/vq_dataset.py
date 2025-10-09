# agent_dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import random

class MotionVQDataset(Dataset):
    def __init__(self, mocap_data: torch.Tensor, end_indices: list, window_size: int, mean: torch.Tensor, std: torch.Tensor):
        """
        Args:
            mocap_data (np.ndarray): (총 프레임 수, 관절 채널 수) 형태의 전체 모션 데이터
            end_indices (list): 각 모션 시퀀스의 끝 인덱스 리스트
            window_size (int): 학습에 사용할 프레임 윈도우 크기
            mean (np.ndarray): 정규화를 위한 평균값
            std (np.ndarray): 정규화를 위한 표준편차값
        """
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