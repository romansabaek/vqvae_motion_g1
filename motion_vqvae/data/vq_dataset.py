# agent_dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import random

class MotionVQDataset(Dataset):
    def __init__(self, mocap_data: torch.Tensor, end_indices: list, window_size: int, mean: torch.Tensor, std: torch.Tensor, policy_ids: np.ndarray = None):
        self.window_size = window_size
        self.mean = mean
        self.std = std
        self.policy_ids = policy_ids  # Policy IDs per frame [F] - None if not available

        self.motion_list = []
        self.policy_id_list = []  # Policy IDs per motion
        start_idx = 0
        mocap_data = mocap_data.cpu()
        for motion_idx, end_idx in enumerate(end_indices):
            motion = mocap_data[start_idx:end_idx + 1]
            self.motion_list.append(motion)
            
            # Extract corresponding policy IDs for this motion
            if self.policy_ids is not None:
                motion_policy_ids = self.policy_ids[start_idx:end_idx + 1]
                self.policy_id_list.append(motion_policy_ids)
            else:
                self.policy_id_list.append(None)
            
            start_idx = end_idx + 1

        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def __len__(self):
        return len(self.motion_list)

    def __getitem__(self, index):
        motion = self.motion_list[index]
        n_frames = motion.shape[0]
        motion_policy_ids = self.policy_id_list[index]

        if n_frames < self.window_size:
            # Original T2M-GPT padding logic
            # Duplicate last frame to match window_size
            shortage = self.window_size - n_frames
            padding = motion[-1:].repeat(shortage, 1)
            motion = torch.cat([motion, padding], dim=0)
            
            # Pad policy IDs similarly
            if motion_policy_ids is not None:
                pad_value = motion_policy_ids[-1] if len(motion_policy_ids) > 0 else -1
                policy_padding = np.full(shortage, pad_value, dtype=np.int64)
                motion_policy_ids = np.concatenate([motion_policy_ids, policy_padding])
        else:
            # Random start point and cut window_size
            random_start = random.randint(0, n_frames - self.window_size)
            motion = motion[random_start : random_start + self.window_size]
            
            # Extract corresponding policy IDs
            if motion_policy_ids is not None:
                motion_policy_ids = motion_policy_ids[random_start : random_start + self.window_size]

        # Z-score normalization
        motion = (motion - self.mean) / self.std

        # Return motion and policy IDs (if available)
        # Always return a tuple to ensure DataLoader handles it correctly
        if motion_policy_ids is not None:
            return motion, torch.from_numpy(motion_policy_ids).long()
        else:
            # Return a dummy tensor with -1 values when policy_ids are not available
            # This ensures DataLoader can properly collate the batch
            dummy_policy_ids = torch.full((self.window_size,), -1, dtype=torch.long)
            return motion, dummy_policy_ids


def cycle(iterable):
    """
    Make DataLoader into an infinite iterator
    Original T2M-GPT utility function
    """
    while True:
        for x in iterable:
            yield x