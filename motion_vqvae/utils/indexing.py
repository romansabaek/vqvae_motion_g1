import numpy as np

def build_real_to_sim_idx(sim_to_real_idx, real_dof):
    """
    Build a mapping from real joint indices to simulated joint indices.
    
    Args:
        sim_to_real_idx (list): List of indices mapping simulated joints to real joints.
        real_dof (int): Total number of real degrees of freedom (dof).
        
    Returns:
        list: Mapping from real joint indices to simulated joint indices.
    """
    real_to_sim = [-1] * real_dof  # Initialize all real joints as unmapped
    for sim_idx, real_idx in enumerate(sim_to_real_idx):
        if real_idx < real_dof:
            real_to_sim[real_idx] = sim_idx
        else:
            breakpoint
    return real_to_sim


def sim_to_real_joint_values(sim_joint_values, sim_to_real_indices, real_dof):
    """
    Convert simulated joint values to real joint values.
    
    Args:
        sim_joint_values (np.ndarray): Sim joint values of shape (sim_dof,)
        sim_to_real_indices (List[int]): Indices in real joint array where sim values go
        real_dof (int): Total DoF in the real robot

    Returns:
        np.ndarray: Real joint values of shape (real_dof,), zero-padded where no sim values exist.
    """
    real_joint_values = np.zeros(real_dof)
    for sim_idx, real_idx in enumerate(sim_to_real_indices):
        if real_idx != -1:
            real_joint_values[real_idx] = sim_joint_values[sim_idx]
    return real_joint_values


def real_to_sim_joint_values(real_joint_values, real_to_sim_indices, sim_dof):
    """
    Extract simulated joint values from real joint values.
    
    Args:
        real_joint_values (np.ndarray): Real joint values of shape (real_dof,)
        real_to_sim_indices (List[int]): Mapping of real index to sim index (use -1 for non-sim joints)

    Returns:
        np.ndarray: Simulated joint values of shape (sim_dof,)
    """
    sim_joint_values = np.zeros(sim_dof)
    for real_idx, sim_idx in enumerate(real_to_sim_indices):
        if sim_idx != -1:
            sim_joint_values[sim_idx] = real_joint_values[real_idx]
    return sim_joint_values


import torch

def real_to_sim_joint_values_batch(real_joint_values, real_to_sim_indices, sim_dof):
    """
    Extract simulated joint values from batched real joint values (PyTorch version).

    Args:
        real_joint_values (torch.Tensor): Tensor of shape (B, real_dof)
        real_to_sim_indices (List[int]): Mapping from real joint index to sim index (-1 if ignored)
        sim_dof (int): Number of simulated DoFs

    Returns:
        torch.Tensor: Simulated joint values of shape (B, sim_dof)
    """
    B, real_dof = real_joint_values.shape
    sim_joint_values = torch.zeros(B, sim_dof, dtype=real_joint_values.dtype, device=real_joint_values.device)

    for real_idx, sim_idx in enumerate(real_to_sim_indices):
        if sim_idx != -1:
            sim_joint_values[:, sim_idx] = real_joint_values[:, real_idx]
    
    return sim_joint_values


def sim_to_real_joint_values_batch(sim_joint_values, sim_to_real_indices, real_dof):
    """
    Convert simulated joint values to real joint values (PyTorch batch version).

    Args:
        sim_joint_values (torch.Tensor): Tensor of shape (B, sim_dof)
        sim_to_real_indices (List[int]): Mapping from sim joint index to real index (-1 if ignored)
        real_dof (int): Number of real DoFs

    Returns:
        torch.Tensor: Real joint values of shape (B, real_dof)
    """
    B, sim_dof = sim_joint_values.shape
    real_joint_values = torch.zeros(B, real_dof, dtype=sim_joint_values.dtype, device=sim_joint_values.device)

    for sim_idx, real_idx in enumerate(sim_to_real_indices):
        if real_idx != -1:
            real_joint_values[:, real_idx] = sim_joint_values[:, sim_idx]
    
    return real_joint_values


if __name__ == "__main__":
    # Example usage
      # Simulated joints mapping to real joints list of length (sim_dof, )
    sim_to_real_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    real_dof = 29  # Total number of real degrees of freedom
    real_to_sim_idx = build_real_to_sim_idx(sim_to_real_idx, real_dof)
    
    print("Real to Sim Mapping:", real_to_sim_idx)

    sim_joint_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    real_joint_values = sim_to_real_joint_values(sim_joint_values, sim_to_real_idx, real_dof)
    sim_joint_values_extracted = real_to_sim_joint_values(real_joint_values, real_to_sim_idx, len(sim_to_real_idx))

    print("Sim joint values: ", sim_joint_values)
    print("Real joint values: ", real_joint_values)
    print("Extracted Sim joint values: ", sim_joint_values_extracted)