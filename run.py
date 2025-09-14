#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Solar Full-Disk Hyperspectral Patch-wise Convolutional Autoencoder (CAE)
=======================================================================

项目入口脚本
------------
本脚本作为高光谱太阳图像降维项目的主入口，负责参数解析、数据集构建、模型初始化、训练与推理流程调度。

主要功能：
- 支持对FITS格式的高光谱太阳数据进行分块训练和降维编码。
- 训练卷积自编码器（CAE），实现谱维降维（如118通道→16通道），空间分辨率保持不变。
- 支持patch过滤、归一化、数据增强、混合精度训练、断点恢复、模型导出等。
- 推理阶段可批量编码整幅图像，输出降维后的数据供下游目标检测使用。

使用方法请参考README.md。
"""

"""
主入口脚本，负责参数解析、数据集构建、模型初始化、训练/推理流程调度。
"""
import os
import glob
import json
from dateutil.parser import parse
from dataclasses import asdict
from config import parse_args
from utils.seed import set_seed, get_device
from datasets.solar_patch import HyperspecSolarPatchDataset, train_val_split
from model.cae import ChannelCAE
from loss.loss import create_channel_weights
from train.train_loop import run_epoch
from train.checkpoint import save_checkpoint, load_checkpoint
from encode.encode_util import encode_full_image


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device()
    print("Config:\n" + json.dumps(asdict(cfg), indent=2))
    print(f"Device: {device}")
    pattern = os.path.join(cfg.data_root, cfg.fits_glob)
    file_list = sorted(glob.glob(pattern))
    if len(file_list) == 0:
        raise SystemExit(f"No files matched {pattern}")
    print(f"Found {len(file_list)} files")
    if cfg.encode_only:
        if cfg.checkpoint is None:
            raise SystemExit("--encode-only requires --checkpoint")
        from utils.io import FITSArrayReader
        rdr = FITSArrayReader(file_list[0])
        in_ch = rdr.shape[0]
        rdr.close()
        model = ChannelCAE(in_ch=in_ch, latent_ch=cfg.latent_ch, mid_ch=cfg.mid_ch)
        state = load_checkpoint(cfg.checkpoint, map_location=device)
        model.load_state_dict(state['model'])
        model.to(device)
        os.makedirs(cfg.encode_out, exist_ok=True)
        import torch
        with torch.no_grad():
            now = 0
            for fp in file_list:
                rdr = FITSArrayReader(fp)
                arr = rdr.read_full(); rdr.close()
                arr = arr.astype('float32', copy=False)
                z = encode_full_image(model, arr, patch=cfg.patch_size, stride=cfg.stride, device=device, batch_size=cfg.batch_size if hasattr(cfg,'batch_size') else 4)
                base = os.path.basename(fp)
                hdu_date = parse(base.split("RSM")[1][:8]).strftime("%Y_%m%d")
                hdu_time = base.split("T")[1][:6]
                dtime = hdu_date+'_'+hdu_time
                out_path = os.path.join(cfg.encode_out, f"{dtime}.npy")
                import numpy as np
                np.save(out_path, z)
                print(f"\r 已完成 {now+1}/{len(file_list)}: {out_path}", end='', flush=True)
                now += 1
        print("Encoding complete.")
        return
    train_files, val_files = train_val_split(file_list, cfg.val_frac, cfg.seed)
    print(f"Train files: {len(train_files)} | Val files: {len(val_files)}")
    CACHE_SIZE = 50 # 不通过命令行参数设置，请一定一定根据内存实际情况微调！！！！！！
    train_ds = HyperspecSolarPatchDataset(train_files,
                                          patch_size=cfg.patch_size,
                                          stride=cfg.stride,
                                          drop_last=cfg.drop_last,
                                          norm=cfg.norm,
                                          var_threshold=cfg.var_threshold,
                                          nonzero_threshold=cfg.nonzero_threshold,
                                          files_to_cache_per_epoch=CACHE_SIZE,
                                          augment=cfg.augment)
    val_ds = HyperspecSolarPatchDataset(val_files,
                                        patch_size=cfg.patch_size,
                                        stride=cfg.stride,
                                        drop_last=cfg.drop_last,
                                        norm=cfg.norm,
                                        var_threshold=0.0,
                                        nonzero_threshold=0.0,
                                        files_to_cache_per_epoch=CACHE_SIZE,
                                        augment=False)
    print(f"Train patches: {len(train_ds)} | Val patches: {len(val_ds)}")
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    in_ch = train_ds.C
    print(f"Detected input channels: {in_ch}")
    model = ChannelCAE(in_ch=in_ch, latent_ch=cfg.latent_ch, mid_ch=cfg.mid_ch)
    model.to(device)
    print(f"Creating channel weights centered at index 67.")
    channel_weights = create_channel_weights(num_channels=in_ch, core_idx=67, device=device)
    import torch
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type=='cuda'))
    start_epoch = 0
    best_val = float('inf')
    if cfg.resume or cfg.checkpoint:
        ckpt_path = cfg.resume if cfg.resume is not None else cfg.checkpoint
        state = load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start_epoch = state.get('epoch',0)+1
        best_val = state.get('best_val', best_val)
        print(f"Resumed from {ckpt_path} @ epoch {start_epoch-1} best_val={best_val:.4e}")
    history = []
    patience = 10
    no_improve = 0

    with open(os.path.join(cfg.save_dir, 'log.csv'), 'w') as f:
        f.write("epoch,train_loss,train_l1,train_l2,train_ssim,val_loss,val_l1,val_l2,val_ssim\n")
    
    for epoch in range(start_epoch, cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")

        # 刷新 train/val 的热缓存，如果Dataset那边不刷新，那就在这边要刷新
        # train_ds.refresh_cache()
        # val_ds.refresh_cache()

        # 每个 epoch 重新构造 DataLoader，确保热缓存生效
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds,
                                batch_size=cfg.batch_size, shuffle=True,
                                num_workers=cfg.num_workers,
                                pin_memory=True, drop_last=True,
                                persistent_workers=False)   # 细节：一定要关掉prersistent_workers，因为常驻workers会导致Dataset的热缓存失效
        val_loader = DataLoader(val_ds,
                                batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.num_workers,
                                pin_memory=True, drop_last=False,
                                persistent_workers=False)     # 关掉
        tr = run_epoch(model, train_loader, optimizer, scaler if cfg.amp else None, device,
                       channel_weights=channel_weights,
                       train=True, l1_w=cfg.l1_w, l2_w=cfg.l2_w, ssim_w=cfg.ssim_w, log_interval=cfg.log_interval)
        vl = run_epoch(model, val_loader, optimizer, scaler if cfg.amp else None, device,
                       channel_weights=channel_weights,
                       train=False, l1_w=cfg.l1_w, l2_w=cfg.l2_w, ssim_w=cfg.ssim_w, log_interval=cfg.log_interval)
        history.append({'epoch': epoch, 'train': tr, 'val': vl})
        print(f"Train loss: {tr['loss']:.6f} | Val loss: {vl['loss']:.6f}")
        save_checkpoint({'epoch': epoch,
                         'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scaler': scaler.state_dict() if cfg.amp else {},
                         'best_val': best_val,
                         'config': asdict(cfg),
                         'history': history},
                        os.path.join(cfg.save_dir, 'last.pt'))
        
        with open(os.path.join(cfg.save_dir, 'log.csv'), 'a') as f:
            f.write(f"{epoch},{tr['loss']},{tr['l1']},{tr['l2']},{tr['ssim']},{vl['loss']},{vl['l1']},{vl['l2']},{vl['ssim']}\n")
            
        if vl['loss'] < best_val:
            best_val = vl['loss']
            no_improve = 0
            save_checkpoint({'epoch': epoch,
                             'model': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'scaler': scaler.state_dict() if cfg.amp else {},
                             'best_val': best_val,
                             'config': asdict(cfg),
                             'history': history},
                            os.path.join(cfg.save_dir, 'best.pt'))
            print("** New best model saved **")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping after {patience} epochs without val improvement.")
                break
    print("Training complete.")

if __name__ == '__main__':
    main()

