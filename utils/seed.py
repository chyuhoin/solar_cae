"""
随机种子设置与设备选择工具。
"""
import torch
import random
import numpy as np

def set_seed(seed: int = 42):
    """
    设置所有相关库的随机种子，保证实验可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """
    自动检测并返回可用的计算设备（优先使用CUDA）。
    """
    return torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
