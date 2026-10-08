"""Real-ESRGAN "realesr-general-x4v3" (SRVGGNetCompact) on CPU, blended with a Lanczos upscale.

The compact general model restores edges/lights nicely but can make skin look painted, so the output is
mixed with a plain Lanczos upscale (`detail` = weight of the GAN result) and the denoise-strength trick
from Real-ESRGAN (interpolating the x4v3 and wdn-x4v3 weights) is exposed as `denoise`.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        body += [nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1)]
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for m in self.body:
            out = m(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


class Upscaler:
    def __init__(self, weights, wdn_weights=None, denoise=0.5, tile=256):
        sd = torch.load(weights, map_location="cpu")
        sd = sd.get("params_ema", sd.get("params", sd))
        if wdn_weights is not None and denoise < 1:
            sd2 = torch.load(wdn_weights, map_location="cpu")
            sd2 = sd2.get("params_ema", sd2.get("params", sd2))
            sd = {k: denoise * sd[k] + (1 - denoise) * sd2[k] for k in sd}
        self.net = SRVGGNetCompact()
        self.net.load_state_dict(sd)
        self.net.eval()
        self.tile = tile

    @torch.inference_mode()
    def x4(self, img):
        """img: HxWx3 float32 RGB [0,1] -> 4H x 4W x 3."""
        t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].float()
        _, _, h, w = t.shape
        pad = 10
        out = torch.zeros(1, 3, h * 4, w * 4)
        ts = self.tile
        for y in range(0, h, ts):
            for x in range(0, w, ts):
                y0, x0 = max(y - pad, 0), max(x - pad, 0)
                y1, x1 = min(y + ts + pad, h), min(x + ts + pad, w)
                o = self.net(t[:, :, y0:y1, x0:x1])
                oy, ox = (y - y0) * 4, (x - x0) * 4
                hh, ww = min(ts, h - y) * 4, min(ts, w - x) * 4
                out[:, :, y * 4:y * 4 + hh, x * 4:x * 4 + ww] = o[:, :, oy:oy + hh, ox:ox + ww]
        return out[0].permute(1, 2, 0).clamp(0, 1).numpy()


def upscale_to(img, size, upscaler=None, detail=0.6):
    """Resize HxWx3 float RGB to `size`=(W,H). With an Upscaler, mixes GAN x4 (downsampled) with Lanczos."""
    lan = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    if upscaler is None or detail <= 0:
        return lan.clip(0, 1)
    sr = cv2.resize(upscaler.x4(img), size, interpolation=cv2.INTER_AREA)
    return (lan * (1 - detail) + sr * detail).clip(0, 1)
