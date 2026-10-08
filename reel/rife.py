"""CPU frame interpolation with RIFE v4.25 (weights: flownet_v4.25.pkl from the vs-rife model release)."""
import numpy as np
import torch
import torch.nn.functional as F

from .third_party.rife_ifnet_v4_25 import Head, IFNet


class Rife:
    def __init__(self, weights_path, scale=1.0):
        sd = torch.load(weights_path, map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items() if "module." in k}
        self.net = IFNet(scale, False)
        self.net.load_state_dict(sd, strict=False)
        self.net.eval()
        self.enc = Head()
        self.enc.load_state_dict({k.replace("encode.", ""): v for k, v in sd.items() if "encode." in k})
        self.enc.eval()
        self._grid_key = None
        self._cache = {}

    def _prep(self, h, w):
        ph, pw = -(-h // 64) * 64, -(-w // 64) * 64
        if self._grid_key != (ph, pw):
            self.div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0])
            gx = torch.linspace(-1.0, 1.0, pw).view(1, 1, 1, pw).expand(-1, -1, ph, -1)
            gy = torch.linspace(-1.0, 1.0, ph).view(1, 1, ph, 1).expand(-1, -1, -1, pw)
            self.grid = torch.cat([gx, gy], 1)
            self._grid_key = (ph, pw)
        return ph, pw

    def _tensor(self, img, ph, pw):
        """img: HxWx3 float32 RGB in [0,1]."""
        t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].float()
        return F.pad(t, (0, pw - img.shape[1], 0, ph - img.shape[0]), mode="replicate")

    def interpolate(self, img0, img1, t, ensemble=False):
        """Return the frame at fraction t (0..1) between img0 and img1 (HxWx3 float32 RGB [0,1]).
        ensemble=True averages with a horizontally-flipped pass (steadier on slow, detailed motion)."""
        out = self._interpolate(img0, img1, t)
        if ensemble and 0 < t < 1:
            fl = lambda x: np.ascontiguousarray(x[:, ::-1])
            out = (out + fl(self._interpolate(fl(img0), fl(img1), t))) / 2
        return out

    @torch.inference_mode()
    def _interpolate(self, img0, img1, t):
        if t <= 0:
            return img0
        if t >= 1:
            return img1
        h, w = img0.shape[:2]
        ph, pw = self._prep(h, w)
        a, b = self._tensor(img0, ph, pw), self._tensor(img1, ph, pw)
        f0, f1 = self.enc(a), self.enc(b)
        ts = torch.full((1, 1, ph, pw), float(t))
        out = self.net(a, b, ts, self.div, self.grid, f0, f1)
        return out[0, :, :h, :w].permute(1, 2, 0).clamp(0, 1).numpy()


@torch.inference_mode()
def _ifnet_flow(net, img0, img1, timestep, div, grid, f0, f1):
    """IFNet.forward from rife_ifnet_v4_25, but returning the final bidirectional flow and fusion mask."""
    from .third_party.rife_warplayer import warp

    warped_img0, warped_img1 = img0, img1
    flow = mask = feat = None
    blocks = [net.block0, net.block1, net.block2, net.block3, net.block4]
    for i in range(5):
        if flow is None:
            flow, mask, feat = blocks[i](torch.cat((img0, img1, f0, f1, timestep), 1), None, scale=net.scale_list[i])
        else:
            wf0 = warp(f0, flow[:, :2], div, grid)
            wf1 = warp(f1, flow[:, 2:4], div, grid)
            fd, mask, feat = blocks[i](
                torch.cat((warped_img0, warped_img1, wf0, wf1, timestep, mask, feat), 1), flow, scale=net.scale_list[i]
            )
            flow = flow + fd
        warped_img0 = warp(img0, flow[:, :2], div, grid)
        warped_img1 = warp(img1, flow[:, 2:4], div, grid)
    return flow, torch.sigmoid(mask)


def _morph_impl(self, img0, img1, t, weight=None, extra0=None, extra1=None):
    """Flow-based morph: both images are warped to time t; they are blended with `weight`
    (scalar or HxW map, 0 -> img0, 1 -> img1) instead of RIFE's learned occlusion mask, so the
    content cross-dissolves while the geometry moves smoothly (no double images)."""
    from .third_party.rife_warplayer import warp

    h, w = img0.shape[:2]
    ph, pw = self._prep(h, w)
    with torch.inference_mode():
        a, b = self._tensor(img0, ph, pw), self._tensor(img1, ph, pw)
        f0, f1 = self.enc(a), self.enc(b)
        ts = torch.full((1, 1, ph, pw), float(t))
        flow, _ = _ifnet_flow(self.net, a, b, ts, self.div, self.grid, f0, f1)
        wa = warp(a, flow[:, :2], self.div, self.grid)[0, :, :h, :w].permute(1, 2, 0).numpy()
        wb = warp(b, flow[:, 2:4], self.div, self.grid)[0, :, :h, :w].permute(1, 2, 0).numpy()
        ex = []
        for e, fl in ((extra0, flow[:, :2]), (extra1, flow[:, 2:4])):
            if e is None:
                ex.append(None)
                continue
            et = torch.from_numpy(np.ascontiguousarray(e[None, None] if e.ndim == 2 else e.transpose(2, 0, 1)[None]))
            et = F.pad(et.float(), (0, pw - w, 0, ph - h), mode="replicate")
            r = warp(et, fl, self.div, self.grid)[0, :, :h, :w].numpy()
            ex.append(r[0] if e.ndim == 2 else r.transpose(1, 2, 0))
    wgt = t if weight is None else weight
    wgt = np.asarray(wgt, np.float32)
    if wgt.ndim == 2:
        wgt = wgt[..., None]
    out = (wa * (1 - wgt) + wb * wgt).clip(0, 1)
    if extra0 is not None or extra1 is not None:
        return out, wa, wb, ex
    return out, wa, wb


Rife.morph = _morph_impl
