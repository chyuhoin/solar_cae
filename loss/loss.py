"""
加权重建损失、SSIM损失等。
"""
import torch
import torch.nn.functional as F

def create_channel_weights(num_channels: int, core_idx: int, device: torch.device) -> torch.Tensor:
    """
    根据与核心通道的距离，创建用于损失函数加权的张量。
    """
    if not 0 <= core_idx < num_channels:
        raise ValueError("core_idx 必须在通道范围内")
    weights = torch.ones(num_channels, device=device)
    for i in range(6, 11):
        if core_idx - i >= 0:
            weights[core_idx - i] = 5.0
        if core_idx + i < num_channels:
            weights[core_idx + i] = 5.0
    for i in range(1, 6):
        if core_idx - i >= 0:
            weights[core_idx - i] = 10.0
        if core_idx + i < num_channels:
            weights[core_idx + i] = 10.0
    weights[core_idx] = 20.0
    return weights.view(1, -1, 1, 1)

def weighted_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, l1_w: float = 0.5, l2_w: float = 0.5) -> torch.Tensor:
    """
    计算带有通道加权的重建损失。
    """
    l1_element_loss = F.l1_loss(pred, target, reduction='none')
    l2_element_loss = F.mse_loss(pred, target, reduction='none')
    weighted_l1 = (l1_element_loss * weights).mean()
    weighted_l2 = (l2_element_loss * weights).mean()
    return l1_w * weighted_l1 + l2_w * weighted_l2

def ssim(pred: torch.Tensor, target: torch.Tensor, C1=0.01**2, C2=0.03**2, window_size: int = 11) -> torch.Tensor:
    """
    简化版SSIM，通道均值。
    """
    pad = window_size // 2
    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_x ** 2
    sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=pad) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean(dim=(1,2,3))

def ssim_loss(pred: torch.Tensor, target: torch.Tensor, w: float = 0.2) -> torch.Tensor:
    """
    SSIM损失，w为权重。
    """
    if w <= 0:
        return torch.zeros([], device=pred.device)
    return (1 - ssim(pred, target).mean()) * w
