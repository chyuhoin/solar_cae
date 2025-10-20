import numpy as np
import torch
from utils.patch import compute_patch_grid
from utils.seed import get_device

def encode_full_image(model, arr: np.ndarray, patch: int=256, stride: int=None, device=None, batch_size: int=4, use_window: bool=True, verbose: bool=False) -> np.ndarray:
    """
    编码整幅高光谱图像为降维后的latent通道。
    使用重叠窗口和Hanning窗函数实现平滑拼接。
    """
    model.eval()
    if device is None:
        device = get_device()
    
    stride = stride if stride is not None else patch // 2
    C, H, W = arr.shape
    coords = compute_patch_grid(H, W, patch, stride, drop_last=False)
    
    window = None
    if use_window:
        window_1d = np.hanning(patch)
        window_2d = np.outer(window_1d, window_1d)
        window = torch.from_numpy(window_2d.astype(np.float32)).to(device)
    
    with torch.no_grad():
        latent_ch = model.latent_ch
        out_arr = np.zeros((latent_ch, H, W), dtype=np.float32)
        counts = np.zeros((H, W), dtype=np.float32)
        
        if verbose:
            total_patches = len(coords)
            from tqdm import tqdm
            coords = tqdm(coords, total=total_patches, desc="编码图像")
        
        batches = []
        locs = []
        
        for (y, x) in coords:
            end_y = min(y + patch, H)
            end_x = min(x + patch, W)
            actual_h = end_y - y
            actual_w = end_x - x
            
            if actual_h < patch or actual_w < patch:
                temp_patch = np.zeros((C, patch, patch), dtype=np.float32)
                temp_patch[:, :actual_h, :actual_w] = arr[:, y:end_y, x:end_x]
                patch_np = temp_patch
            else:
                patch_np = arr[:, y:y+patch, x:x+patch]
            
            patch_t = torch.from_numpy(patch_np).unsqueeze(0).to(device)
            batches.append(patch_t)
            locs.append((y, x, actual_h, actual_w))
            
            if len(batches) == batch_size:
                batch_t = torch.cat(batches, dim=0)
                z = model.encode(batch_t)
                z_np = z.cpu().numpy()
                
                for i, (yy, xx, act_h, act_w) in enumerate(locs):
                    ps = z_np[i]
                    
                    if use_window and window is not None:
                        win = torch.nn.functional.interpolate(
                            window.unsqueeze(0).unsqueeze(0), 
                            size=(ps.shape[1], ps.shape[2]), 
                            mode='bilinear'
                        ).squeeze().cpu().numpy()
                        ps = ps * win[None, :, :]
                    
                    h_slice = slice(yy, yy + min(ps.shape[1], act_h))
                    w_slice = slice(xx, xx + min(ps.shape[2], act_w))
                    
                    out_arr[:, h_slice, w_slice] += ps[:, :act_h, :act_w]
                    counts[h_slice, w_slice] += win[:act_h, :act_w] if use_window else 1
                
                batches = []
                locs = []
        
        # 处理最后一批
        if batches:
            batch_t = torch.cat(batches, dim=0)
            z = model.encode(batch_t)
            z_np = z.cpu().numpy()
            
            for i, (yy, xx, act_h, act_w) in enumerate(locs):
                ps = z_np[i]
                
                if use_window and window is not None:
                    win = torch.nn.functional.interpolate(
                        window.unsqueeze(0).unsqueeze(0), 
                        size=(ps.shape[1], ps.shape[2]), 
                        mode='bilinear'
                    ).squeeze().cpu().numpy()
                    ps = ps * win[None, :, :]
                
                h_slice = slice(yy, yy + min(ps.shape[1], act_h))
                w_slice = slice(xx, xx + min(ps.shape[2], act_w))
                
                out_arr[:, h_slice, w_slice] += ps[:, :act_h, :act_w]
                counts[h_slice, w_slice] += win[:act_h, :act_w] if use_window else 1
    
    counts = np.maximum(counts, 1e-6)
    
    # 归一化，避免除零
    for c in range(latent_ch):
        out_arr[c] /= counts
    
    return np.flip(out_arr, axis=1)
