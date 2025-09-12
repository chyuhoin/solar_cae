"""
整图编码工具，将高光谱图像降维为latent通道。
"""
import numpy as np
from utils.patch import compute_patch_grid
from utils.seed import get_device

def encode_full_image(model, arr: np.ndarray, patch: int=256, stride: int=None, device=None, batch_size: int=4, verbose: bool=True) -> np.ndarray:
    """
    编码整幅高光谱图像为降维后的latent通道。
    """
    import torch
    model.eval()
    if device is None:
        device = get_device()
    stride = stride if stride is not None else patch
    C,H,W = arr.shape
    coords = compute_patch_grid(H,W,patch,stride,stride==patch)
    with torch.no_grad():
        latent_ch = model.latent_ch
        out_arr = np.zeros((latent_ch, H, W), dtype=np.float32)
        counts = np.zeros((H, W), dtype=np.float32)
        batches = []
        locs = []
        for (y,x) in coords:
            patch_np = arr[:, y:y+patch, x:x+patch]
            patch_t = torch.from_numpy(patch_np).unsqueeze(0).to(device)
            batches.append(patch_t)
            locs.append((y,x))
            if len(batches) == batch_size:
                batch_t = torch.cat(batches, dim=0)
                z = model.encode(batch_t)
                z_np = z.cpu().numpy()
                for i,(yy,xx) in enumerate(locs):
                    ps = z_np[i]
                    h_slice = slice(yy, yy+ps.shape[1])
                    w_slice = slice(xx, xx+ps.shape[2])
                    out_arr[:, h_slice, w_slice] += ps
                    counts[h_slice, w_slice] += 1
                batches = []; locs=[]
        if batches:
            batch_t = torch.cat(batches, dim=0)
            z = model.encode(batch_t)
            z_np = z.cpu().numpy()
            for i,(yy,xx) in enumerate(locs):
                ps = z_np[i]
                h_slice = slice(yy, yy+ps.shape[1])
                w_slice = slice(xx, xx+ps.shape[2])
                out_arr[:, h_slice, w_slice] += ps
                counts[h_slice, w_slice] += 1
    counts[counts==0] = 1
    out_arr /= counts
    return np.flip(out_arr, axis=1)
