"""
模型保存与加载工具。
"""
import os
import torch

def save_checkpoint(state, path):
    """
    保存模型、优化器等状态到指定路径。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, map_location=None):
    """
    加载模型、优化器等状态。
    """
    return torch.load(path, map_location=map_location)
