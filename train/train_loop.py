"""
训练与验证循环，支持AMP与多损失加权。
"""
from tqdm import tqdm
import torch
import torch.nn.functional as F
from loss.loss import ssim

def run_epoch(model, loader, optimizer, scaler, device, channel_weights, train=True, l1_w=0.5, l2_w=0.5, ssim_w=0.0, log_interval=50):
    """
    单轮训练或验证，支持混合精度和多损失加权。
    """
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_l1 = 0.0
    total_l2 = 0.0
    total_ssim = 0.0
    n = 0
    pbar = tqdm(enumerate(loader), total=len(loader), disable=(len(loader)<2))
    for it, batch in pbar:
        batch = batch.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            if scaler is not None and train:
                with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type=='cuda' else torch.bfloat16, enabled=True):
                    out, _ = model(batch)
                    from loss.loss import weighted_reconstruction_loss, ssim_loss
                    l_rec = weighted_reconstruction_loss(out, batch, channel_weights, l1_w=l1_w, l2_w=l2_w)
                    l_ssim = ssim_loss(out, batch, w=ssim_w)
                    loss = l_rec + l_ssim
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                out, _ = model(batch)
                from loss.loss import weighted_reconstruction_loss, ssim_loss
                l_rec = weighted_reconstruction_loss(out, batch, channel_weights, l1_w=l1_w, l2_w=l2_w)
                l_ssim = ssim_loss(out, batch, w=ssim_w)
                loss = l_rec + l_ssim
                if train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        bs = batch.shape[0]
        total_loss += float(loss.detach()) * bs
        total_l1 += float(l_rec.detach()) * bs
        total_l2 += float(F.mse_loss(out.detach(), batch)) * bs
        total_ssim += float((1-ssim(out.detach(), batch).mean()).detach()) * bs
        n += bs
        if it % log_interval == 0:
            pbar.set_description(f"{'train' if train else 'val'} loss={total_loss/n:.4e}")
    return dict(loss=total_loss/n, l1=total_l1/n, l2=total_l2/n, ssim=total_ssim/n)
