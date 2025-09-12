"""
高光谱太阳patch数据集，支持patch过滤、归一化、数据增强与缓存。
"""
from torch.utils.data import Dataset
from astropy.io import fits
from typing import Sequence, List, Tuple, Dict, Optional
import numpy as np
import random
from utils.io import FITSArrayReader, InMemoryReader
from utils.patch import compute_patch_grid

class HyperspecSolarPatchDataset(Dataset):
    """
    高光谱太阳patch数据集，支持懒加载、归一化、过滤和增强。
    """
    def __init__(self,
                 file_paths: Sequence[str],
                 patch_size: int = 256,
                 stride: Optional[int] = None,
                 random_patches_per_image: Optional[int] = None,
                 drop_last: bool = False,
                 norm: str = 'none',
                 var_threshold: float = 0.0,
                 nonzero_threshold: float = 0.0,
                 augment: bool = False,
                 cache_handles: int = 4,
                 files_to_cache_per_epoch: int = 100,
                 dtype=np.float32):
        super().__init__()
        self.file_paths = list(file_paths)                          # fits图像文件目录
        self.patch_size = patch_size                                # patch尺寸
        self.stride = stride if stride is not None else patch_size  # 步长
        self.random_patches_per_image = random_patches_per_image    # 每图像随机采样patch数
        self.drop_last = drop_last                                  # 是否丢弃边界不满的patch
        self.norm = norm                                            # 归一化方法
        self.var_threshold = var_threshold                          # 方差过滤阈值
        self.nonzero_threshold = nonzero_threshold                  # 非零比例过滤阈值
        self.augment = augment                                      # 是否数据增强
        self.cache_handles = cache_handles                          # 同时打开的文件句柄数
        self.files_to_cache_per_epoch = files_to_cache_per_epoch    # 每epoch缓存的文件数
        self.dtype = dtype                                          # 数据类型
        self.reader = None                                          # 把FITS文件读取到内存的工具
        assert len(self.file_paths) > 0, "No input files!"
        self.C, self.H, self.W = self._inspect_shape(self.file_paths[0])
        
        # 注意：这里创建的index是完整的索引列表，实际遍历时会根据“热集合”过滤
        self.index: List[Tuple[int, int, int]] = []                 # (file_idx, y, x) 分别表示文件索引与patch左上角坐标
        for fi, fp in enumerate(self.file_paths):
            if self.random_patches_per_image is None:
                coords = compute_patch_grid(self.H, self.W, self.patch_size, self.stride,
                                            drop_last=self.drop_last)
            else:
                coords = self._sample_random_coords(self.random_patches_per_image, self.H, self.W, self.patch_size)
            for (y,x) in coords:
                self.index.append((fi, y, x))
        self._handle_cache = {}
        self._handle_lru = []

        # 启动一个“热集合”内存缓存（可选，也可以注释掉）
        self.refresh_cache()

    def _inspect_shape(self, path: str) -> Tuple[int,int,int]:
        # 只打开 header 获取形状，不触发整幅读取
        hdul = fits.open(path, memmap=True)
        try:
            h = hdul[0]; data = h.data
            if data is None or data.ndim != 3:
                h = hdul[1]
            C = h.data.shape[0]
            C = 118 if C == 116 else C
            H = 2048; W = 2048  # 固定为 2048
        finally:
            hdul.close()
        return (C, H, W)
    
    def refresh_cache(self):
        """
        清空旧缓存，随机选择新文件子集并加载到内存。
        """
        # 只改变“热文件集合”，不改变 self.index，保证 __len__ 稳定
        sampled = random.sample(self.file_paths, min(self.files_to_cache_per_epoch, len(self.file_paths)))
        self.reader = InMemoryReader(sampled)  # 这里的reader会把选中的文件全部载入内存

        # 建立一个反向查找：path -> in-memory 索引
        self._hot_map = {fp:i for i, fp in enumerate(sampled)}

        # 构建“活动样本”索引（仅来自热文件）
        hot_set = set(sampled)
        self._active_indices = [] # 这里的active_indices是动态更新的，只包括热集合内的样本
        for gi, (fi, y, x) in enumerate(self.index):
            if self.file_paths[fi] in hot_set:
                self._active_indices.append(gi)

    def _sample_random_coords(self, n: int, H: int, W: int, patch: int) -> List[Tuple[int,int]]:
        coords = []
        if patch >= H and patch >= W:
            return [(0,0)]
        for _ in range(n):
            y = random.randint(0, max(0, H-patch))
            x = random.randint(0, max(0, W-patch))
            coords.append((y,x))
        return coords
    
    def __len__(self) -> int:
        return len(self._active_indices) # 只会返回热集合内的样本数，__getitem__ 也只会遍历这些样本
    
    def _get_handle(self, file_idx: int) -> FITSArrayReader:
        if file_idx in self._handle_cache:
            if file_idx in self._handle_lru:
                self._handle_lru.remove(file_idx)
            self._handle_lru.insert(0, file_idx)
            return self._handle_cache[file_idx]
        fp = self.file_paths[file_idx]
        reader = FITSArrayReader(fp)
        self._handle_cache[file_idx] = reader
        self._handle_lru.insert(0, file_idx)
        while len(self._handle_lru) > self.cache_handles:
            old = self._handle_lru.pop()
            h = self._handle_cache.pop(old)
            h.close()
        return reader
    
    def _load_patch(self, file_idx: int, y: int, x: int) -> np.ndarray:
        h = self._get_handle(file_idx)
        arr = h.read_patch(y, x, self.patch_size)
        return arr
    
    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        if self.norm == 'none':
            return arr
        elif self.norm == 'minmax':
            mn = np.nanmin(arr); mx = np.nanmax(arr)
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
            else:
                arr = np.zeros_like(arr)
            return arr
        elif self.norm == 'zscore':
            mean = np.nanmean(arr); std = np.nanstd(arr)
            if std > 0:
                arr = (arr - mean) / std
            else:
                arr = arr - mean
            return arr
        elif self.norm == 'perpatch':
            mn = arr.min(); mx = arr.max()
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
            return arr
        else:
            raise ValueError(f"Unknown norm {self.norm}")
        
    def _filter_patch(self, arr: np.ndarray) -> bool:
        # 过滤掉不合格的patch
        if self.var_threshold > 0.0: # 过滤方差过小的patch
            v = float(arr.var())
            if v < self.var_threshold:
                return False
        if self.nonzero_threshold > 0.0: # 过滤几乎全是0的patch
            nz = float((arr > 0).mean())
            if nz < self.nonzero_threshold:
                return False
        return True
    
    def _augment(self, t):
        import torch
        if random.random() < 0.5:
            t = torch.flip(t, dims=[2])
        if random.random() < 0.5:
            t = torch.flip(t, dims=[1])
        k = random.randint(0,3)
        if k:
            t = torch.rot90(t, k, dims=[1,2])
        return t
    
    def __getitem__(self, idx: int): # 这里的idx是self._active_indices的索引
        # 将 DataLoader 的 idx 映射回全局 index
        gi = self._active_indices[idx] # 全局的index
        file_idx, y, x = self.index[gi]
        fp = self.file_paths[file_idx] # 真实的文件名

        # 走内存热缓存
        hot_idx = self._hot_map.get(fp, None)
        if hot_idx is not None:
            arr = self.reader.read_patch(hot_idx, y, x, self.patch_size)
        else:
            # 极少数兜底（理论上不会走到）
            arr = self._load_patch(file_idx, y, x)

        arr = arr.astype(self.dtype, copy=False)
        arr = self._normalize(arr)
        
        # 有些patch方差过小或者几乎全是0，需要过滤
        if not self._filter_patch(arr): 
            for _ in range(10):
                # 随机采样一个新patch
                y2 = random.randint(0, max(0, self.H - self.patch_size))
                x2 = random.randint(0, max(0, self.W - self.patch_size))
                arr = self._load_patch(file_idx, y2, x2)
                arr = arr.astype(self.dtype, copy=False)
                arr = self._normalize(arr)
                if self._filter_patch(arr):
                    break
        import torch
        t = torch.from_numpy(arr)
        if self.augment:
            t = self._augment(t)
        return t

def train_val_split(files: Sequence[str], val_frac: float = 0.1, seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    按文件划分训练/验证集，避免同一图像patch泄漏。
    """
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_frac))
    val_files = files[:n_val]
    train_files = files[n_val:]
    return train_files, val_files
