"""
patch网格生成工具，支持重叠与边界处理。
"""
from typing import List, Tuple, Optional

def compute_patch_grid(h: int, w: int, patch: int, stride: Optional[int] = None,
                       drop_last: bool = False) -> List[Tuple[int, int]]:
    """
    计算图像分块的左上角坐标列表。
    参数：
        h, w: 图像高宽
        patch: patch尺寸（正方形）
        stride: 步长，默认等于patch（无重叠）
        drop_last: 是否丢弃边界不满的patch
    返回：
        坐标列表[(y,x)]
    """
    if stride is None:
        stride = patch
    coords = []
    y = 0
    while True:
        x = 0
        y_end = y + patch
        if y_end > h:
            if drop_last:
                break
            y = h - patch
            y_end = h
        while True:
            x_end = x + patch
            if x_end > w:
                if drop_last:
                    break
                x = w - patch
                x_end = w
            coords.append((y, x))
            if x + stride >= w:
                break
            x += stride
        if y + stride >= h:
            break
        y += stride
    # 去重
    coords = list(dict.fromkeys(coords))
    return coords
