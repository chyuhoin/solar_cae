"""
FITS文件读写、内存映射与归一化工具。
"""
import numpy as np
from astropy.io import fits
from datetime import datetime
from typing import List, Tuple

# 通道最大值，用于归一化
_channel_max = np.array([
    5242,4468,4942,5046,4812,4903,4710,4660,4435,4574,4760,4700,4839,4683,
    4597,4661,4621,4496,4637,4610,4577,4439,4588,4522,4570,4432,4653,4498,
    4555,4442,4271,4427,4377,4245,4467,4497,4374,4380,4420,4262,4032,4218,
    4275,4286,4139,4062,4057,4274,4232,4136,4108,4051,4163,4084,3958,4136,
    3994,4046,4088,4065,4037,3985,3945,3934,3952,3979,3986,3964,3969,4013,
    4015,4036,4058,4045,4004,4052,4087,4103,4125,4149,4176,4180,4186,4200,
    4186,4152,4150,4138,4056,4006,3995,3982,3938,3880,3814,3792,3676,3708,
    3542,3700,3599,3617,3510,3615,3984,3516,3379,3555,3561,3366,3508,3413,
    3480,3363,3481,3406,3549,3368
], dtype=np.float32)

import math

def _crop_window(cx: float, cy: float) -> Tuple[int,int,int,int]:
    # 以 floor(cx) 作为参考像素，保证宽高恒为 2048
    x0 = math.floor(cx) - 1023
    y0 = math.floor(cy) - 1023
    x1 = x0 + 2048
    y1 = y0 + 2048
    return y0, y1, x0, x1

def _crpix_in_cropped(cx: float, cy: float) -> Tuple[float, float]:
    # 回写到裁剪坐标系的参考像素位置（1-based 的 FITS 语义你这里沿用原逻辑）
    fx = cx - math.floor(cx)      # [0,1) 的小数部分
    fy = cy - math.floor(cy)
    return 1023 + fx, 1023 + fy


def _readchase_full(file: str) -> Tuple[np.ndarray, dict]:
    """
    读取FITS文件并归一化，返回数据和头信息。
    """
    hdu = fits.open(file, memmap=True)
    try:
        data = hdu[0].data.astype(np.float32)
        header = dict(hdu[0].header)
    except:
        data = hdu[1].data.astype(np.float32)
        header = dict(hdu[1].header)
    if data.ndim != 3:
        data = hdu[1].data.astype(np.float32)
        header = dict(hdu[1].header)
        if data.ndim != 3:
            hdu.close()
            raise TypeError(f'file {file} is not Chase\'s file, please use other function to read.')
    obs_time = datetime.strptime(header['DATE_OBS'], "%Y-%m-%dT%H:%M:%S")

    # 由于数据准备的失误，2023年4月18日前的文件CRPIX1/CRPIX2写反了
    if obs_time < datetime.strptime('2023-04-18', "%Y-%m-%d"):
        cy = header['CRPIX1']; cx = header['CRPIX2']
    else:
        cx = header['CRPIX1']; cy = header['CRPIX2']
    y0 = int(cy - 1023); y1 = int(cy + 1025)
    x0 = int(cx - 1023); x1 = int(cx + 1025)
    data = data[:, y0:y1, x0:x1]

    # 绝大多数图像都是118通道，少数116通道的在首尾各补一层
    if data.shape not in [(118,2048,2048), (116,2048,2048)]:
        hdu.close()
        raise TypeError(f'Chase file {file} is corrupted, please check.')
    if data.shape[0] == 116:
        first = data[0:1]
        last  = data[-1:]
        data = np.concatenate([first, data, last], axis=0)
    header['CRPIX1'] = 1023 + cx - int(cx)
    header['CRPIX2'] = 1023 + cy - int(cy)
    header['NAXIS1']  = 2048
    header['NAXIS2']  = 2048

    hdu.close()
    data = data / _channel_max[:, None, None]
    return data, header

class InMemoryReader:
    """
    一次性将所有FITS文件载入内存，节省I/O时间。
    """
    def __init__(self, file_list: List[str]):
        self.file_list = file_list
        self.cache: List[np.ndarray] = []
        self.headers: List[dict] = []
        for fp in file_list:
            data, hdr = _readchase_full(fp)
            self.cache.append(data.astype(np.float16))
            self.headers.append(hdr)
        C, H, W = self.cache[0].shape
        self.shape = (C, H, W)
    def read_patch(self, file_idx: int, y: int, x: int, patch: int) -> np.ndarray:
        arr = self.cache[file_idx]
        C, H, W = arr.shape
        y1 = min(y + patch, H)
        x1 = min(x + patch, W)
        sub = arr[:, y:y1, x:x1].astype(np.float32)
        if sub.shape[1]!=patch or sub.shape[2]!=patch:
            buf = np.zeros((C, patch, patch), dtype=np.float32)
            buf[:, :sub.shape[1], :sub.shape[2]] = sub
            return buf
        return sub
    def read_full(self, file_idx: int) -> np.ndarray:
        return self.cache[file_idx].astype(np.float32)
    def close(self):
        self.cache.clear()
        self.headers.clear()

class FITSArrayReader:
    def __init__(self, path: str):
        self.path = path
        self._hdul = fits.open(path, memmap=True)
        self._hdu_idx = 0
        data = self._hdul[0].data
        hdr  = self._hdul[0].header
        if data is None or data.ndim != 3:
            data = self._hdul[1].data
            hdr  = self._hdul[1].header
            self._hdu_idx = 1
        if data is None or data.ndim != 3:
            self.close()
            raise TypeError(f'file {path} is not Chase format')

        header = dict(hdr)
        obs_time = datetime.strptime(header['DATE_OBS'], "%Y-%m-%dT%H:%M:%S")
        if obs_time < datetime.strptime('2023-04-18', "%Y-%m-%d"):
            cy, cx = header['CRPIX1'], header['CRPIX2']
        else:
            cx, cy = header['CRPIX1'], header['CRPIX2']

        # 保存裁剪窗口（不立即拷贝数据）
        self._y0, self._y1, self._x0, self._x1 = _crop_window(cx, cy)
        C_all, H_all, W_all = data.shape
        self._need_pad_118 = (C_all == 116)
        if C_all not in (116, 118):
            self.close()
            raise TypeError(f'Chase file {path} has unexpected channel count {C_all}')

        # 目标形状恒为 (118, 2048, 2048)
        self.shape = (118, 2048, 2048)
        self.header = header
        self.header['CRPIX1'], self.header['CRPIX2'] = _crpix_in_cropped(cx, cy)
        self.header['NAXIS1'] = 2048
        self.header['NAXIS2'] = 2048

    def _read_raw_slice(self, y0, y1, x0, x1) -> np.ndarray:
        """从原始 memmap 中切片（可能需要边界 clamp 与零填充；不归一化，不改 dtype）。"""
        mm = self._hdul[self._hdu_idx].data  # memmap
        C_all, H_all, W_all = mm.shape

        yy0 = max(0, y0); yy1 = min(H_all, y1)
        xx0 = max(0, x0); xx1 = min(W_all, x1)
        sub = mm[:, yy0:yy1, xx0:xx1]  # 仍是 memmap view

        # 如果越界，做零填充到 (C_all, y1-y0, x1-x0)
        outH, outW = y1 - y0, x1 - x0
        if (yy0,yy1,xx0,xx1) != (y0,y1,x0,x1):
            buf = np.zeros((C_all, outH, outW), dtype=sub.dtype)
            oy0 = yy0 - y0; oy1 = oy0 + (yy1 - yy0)
            ox0 = xx0 - x0; ox1 = ox0 + (xx1 - xx0)
            buf[:, oy0:oy1, ox0:ox1] = sub
            sub = buf  # ndarray（非 memmap）
        return sub

    def read_patch(self, y: int, x: int, patch: int) -> np.ndarray:
        # 映射到原图坐标系
        y0 = self._y0 + y; y1 = y0 + patch
        x0 = self._x0 + x; x1 = x0 + patch
        sub = self._read_raw_slice(y0, y1, x0, x1)  # (C_all, patch, patch)

        # 116→118：首尾各补一层（按 edge 复制）
        if self._need_pad_118:
            first = sub[0:1]
            last  = sub[-1:]
            sub = np.concatenate([first, sub, last], axis=0)  # (118, ...)

        # 归一化 + dtype 转换（懒计算）
        sub = sub.astype(np.float32, copy=False)
        sub /= _channel_max[:, None, None]
        return sub

    def read_full(self) -> np.ndarray:
        # 仅当确实需要整幅才调用；仍然按需读取
        return self.read_patch(0, 0, 2048)

    def close(self):
        if getattr(self, "_hdul", None) is not None:
            try: self._hdul.close()
            except: pass
            self._hdul = None

    def __del__(self):
        self.close()
