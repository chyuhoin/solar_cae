"""
参数解析与配置管理模块。
"""
import argparse
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
    data_root: str
    fits_glob: str = "*.fits"
    patch_size: int = 256
    stride: int = 256
    latent_ch: int = 16
    mid_ch: int = 128
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_frac: float = 0.1
    num_workers: int = 4
    amp: bool = True
    seed: int = 42
    l1_w: float = 0.5
    l2_w: float = 0.5
    ssim_w: float = 0.0
    var_threshold: float = 0.0
    nonzero_threshold: float = 0.0
    augment: bool = False
    drop_last: bool = False
    save_dir: str = "runs/cae"
    log_interval: int = 50
    resume: Optional[str] = None
    checkpoint: Optional[str] = None
    encode_only: bool = False
    encode_out: str = "encoded"
    norm: str = 'none'

def parse_args() -> TrainConfig:
    """
    命令行参数解析，返回训练配置对象。
    """
    p = argparse.ArgumentParser(description="Solar Hyperspectral Patch-wise CAE")
    p.add_argument('--data-root', type=str, required=True, help='FITS文件所在目录')
    p.add_argument('--fits-glob', type=str, default='*.fits', help='FITS文件匹配模式')
    p.add_argument('--patch-size', type=int, default=256)
    p.add_argument('--stride', type=int, default=256)
    p.add_argument('--latent-ch', type=int, default=16)
    p.add_argument('--mid-ch', type=int, default=128)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--val-frac', type=float, default=0.1)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--no-amp', dest='amp', action='store_false')
    p.set_defaults(amp=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--l1-w', type=float, default=0.5)
    p.add_argument('--l2-w', type=float, default=0.5)
    p.add_argument('--ssim-w', type=float, default=0.0)
    p.add_argument('--var-threshold', type=float, default=0.0)
    p.add_argument('--nonzero-threshold', type=float, default=0.0)
    p.add_argument('--augment', action='store_true')
    p.add_argument('--drop-last', action='store_true')
    p.add_argument('--save-dir', type=str, default='runs/cae')
    p.add_argument('--log-interval', type=int, default=50)
    p.add_argument('--resume', type=str, default=None, help='断点恢复路径')
    p.add_argument('--checkpoint', type=str, default=None, help='推理/恢复用模型路径')
    p.add_argument('--encode-only', action='store_true', help='仅编码，不训练')
    p.add_argument('--encode-out', type=str, default='encoded', help='编码输出目录')
    p.add_argument('--encode-norm', type=str, default='channel', help='编码时归一化方式：全局/通道/无')
    p.add_argument('--norm', type=str, default='none', choices=['none','minmax','zscore','perpatch'])
    args = p.parse_args()
    cfg = TrainConfig(**vars(args))
    return cfg
