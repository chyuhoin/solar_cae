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

    def _generate_starts(dimension: int, patch_size: int, step: int, drop: bool) -> List[int]:
        """
        生成一个维度上的所有起始坐标。
        """
        starts = []
        current_start = 0
        
        # 所有能完整容纳的patch
        while current_start + patch_size <= dimension:
            starts.append(current_start)
            current_start += step
        
        if not drop:
            boundary_start = max(0, dimension - patch_size)
            
            if not starts or starts[-1] < boundary_start:
                starts.append(boundary_start)
                
        
        return starts

    y_starts = _generate_starts(h, patch, stride, drop_last)
    x_starts = _generate_starts(w, patch, stride, drop_last)

    coords = []
    for y in y_starts:
        for x in x_starts:
            coords.append((y, x))
            
    return coords

