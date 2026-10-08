"""Frame renderer: time-remapped, stabilised, aligned source frames -> virtual camera -> morph ->
upscale -> look (grade/fx/text) -> ffmpeg.

An *edit* object provides:
  fps, duration, out_size
  layers(t)      -> list of dicts {clip, f, T (3x3 source->reference), weight}  (1 or 2 entries)
  camera(t)      -> 3x3 reference->output-pixels matrix
  look(img, ctx) -> graded output image (float RGB, out_size)
"""
import subprocess
import time

import cv2
import numpy as np

from . import paths
from .frames import mat3


class Renderer:
    def __init__(self, edit, clips, rife, upscaler=None, gan_detail=0.65, canvas_scale=0.45):
        self.edit = edit
        self.clips = clips
        self.rife = rife
        self.up = upscaler
        self.gan_detail = gan_detail
        W, H = edit.out_size
        # intermediate canvas (source-like resolution) where warping/morphing happens before upscaling
        self.cw, self.ch = int(round(W * canvas_scale / 2) * 2), int(round(H * canvas_scale / 2) * 2)
        self.canvas = np.diag([self.cw / W, self.ch / H, 1.0])

    def _warp(self, img, M, size, interp=cv2.INTER_CUBIC):
        return cv2.warpAffine(img, M[:2], size, flags=interp, borderMode=cv2.BORDER_REFLECT101)

    def source_layers(self, t):
        """Warp every active layer into the canvas; returns list of (img, matte, candle_xy_out, weight)."""
        out = []
        Cam = self.edit.camera(t)
        for L in self.edit.layers(t):
            clip = self.clips[L["clip"]]
            img = clip.frame(L["f"])
            if L.get("color") is not None:
                img = L["color"](img)
            M = self.canvas @ Cam @ mat3(L["T"])
            cimg = self._warp(img, M, (self.cw, self.ch))
            matte = self._warp(clip.matte_at(L["f"]), M, (self.cw, self.ch), cv2.INTER_LINEAR)
            cand = None
            if L.get("candle") is not None:
                p = Cam @ mat3(L["T"]) @ np.array([*L["candle"], 1.0])
                cand = p[:2]
            out.append(dict(img=cimg, matte=matte, candle=cand, weight=L.get("weight", 1.0), clip=L["clip"],
                            f=L["f"]))
        return out

    def warp_to_canvas(self, img, T, t, interp=cv2.INTER_CUBIC, border=cv2.BORDER_REFLECT101):
        """Warp a source/reference-space image through T (3x3) and the camera at time t onto the canvas."""
        M = self.canvas @ self.edit.camera(t) @ mat3(T)
        return cv2.warpAffine(img, M[:2], (self.cw, self.ch), flags=interp, borderMode=border)

    def to_output(self, p, T, t):
        """Map a source point through T and the camera to output pixel coordinates."""
        q = self.edit.camera(t) @ mat3(T) @ np.array([p[0], p[1], 1.0])
        return q[:2]

    def frame(self, k):
        e = self.edit
        t = k / e.fps
        if getattr(e, "compose", None) is not None:
            img, matte, cand, w, layers = e.compose(t, self)
            return self._finish(img, matte, cand, w, layers, t, k)
        layers = self.source_layers(t)
        if len(layers) == 1:
            L = layers[0]
            img, matte, cand = L["img"], L["matte"], L["candle"]
            w = 0.0 if L["clip"] == e.first_clip else 1.0
        else:
            a, b = layers
            w = float(b["weight"])
            spatial = e.morph_weight_map(t, (self.cw, self.ch), a, b) if hasattr(e, "morph_weight_map") else None
            img, _, _ = self.rife.morph(a["img"], b["img"], w, weight=spatial)
            matte = a["matte"] * (1 - w) + b["matte"] * w
            cand = None if a["candle"] is None or b["candle"] is None else a["candle"] * (1 - w) + b["candle"] * w
        return self._finish(img, matte, cand, w, layers, t, k)

    def _finish(self, img, matte, cand, w, layers, t, k):
        e = self.edit
        W, H = e.out_size
        if self.up is not None:
            from .upscale import upscale_to

            big = upscale_to(img, (W, H), self.up, detail=self.gan_detail)
        else:
            big = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4).clip(0, 1)
        matte_big = cv2.resize(matte, (W, H), interpolation=cv2.INTER_LINEAR)
        ctx = dict(t=t, k=k, matte=matte_big, candle=cand, morph=w, layers=layers)
        return e.look(big.astype(np.float32), ctx)

    def render_png(self, out_dir, frames, log_every=10):
        """Render frames to lossless PNGs (for chunked / parallel renders that are encoded once at the end)."""
        from pathlib import Path

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for i, k in enumerate(frames):
            p = out_dir / f"{k:04d}.png"
            if p.exists():
                continue
            img = (np.clip(self.frame(k), 0, 1) * 255 + 0.5).astype(np.uint8)
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if log_every and (i + 1) % log_every == 0:
                el = time.time() - t0
                print(f"  [{frames[0]}-{frames[-1]}] frame {i + 1}/{len(frames)}  {el / (i + 1):.2f}s/frame", flush=True)

    def render(self, out_path, frames=None, audio=None, crf=16, preview_every=0, log_every=10):
        e = self.edit
        W, H = e.out_size
        n = int(round(e.duration * e.fps))
        ks = range(n) if frames is None else frames
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(e.fps), "-i", "-"]
        if audio is not None:
            cmd += ["-i", str(audio)]
        cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-tune", "film", "-pix_fmt", "yuv420p",
                "-profile:v", "high", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                "-movflags", "+faststart"]
        if audio is not None:
            cmd += ["-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-shortest"]
        cmd += [str(out_path)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        t0 = time.time()
        for i, k in enumerate(ks):
            img = self.frame(k)
            p.stdin.write((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
            if preview_every and k % preview_every == 0:
                prev = paths.WORK / "preview"
                prev.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(prev / f"{k:04d}.jpg"), cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
            if log_every and (i + 1) % log_every == 0:
                el = time.time() - t0
                print(f"  frame {i + 1}/{len(ks)}  {el / (i + 1):.2f}s/frame  eta {(len(ks) - i - 1) * el / (i + 1):.0f}s",
                      flush=True)
        p.stdin.close()
        p.wait()
        if p.returncode:
            raise RuntimeError("ffmpeg failed")
        return out_path


def coverage(edit, src_w=464, src_h=832, step=1):
    """For each output frame, how far (source px) the output rectangle reaches outside each layer's source.

    Returns a list of (k, clip, overshoot_px); overshoot > 0 means reflected border pixels would show.
    """
    W, H = edit.out_size
    corners = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1], [W / 2, 0, 1], [W / 2, H, 1],
                        [0, H / 2, 1], [W, H / 2, 1]], np.float64).T
    res = []
    n = int(round(edit.duration * edit.fps))
    for k in range(0, n, step):
        t = k / edit.fps
        Cam = edit.camera(t)
        for L in edit.layers(t):
            M = Cam @ mat3(L["T"])
            p = np.linalg.inv(M) @ corners
            ox = np.maximum(np.maximum(-p[0], p[0] - src_w), 0).max()
            oy = np.maximum(np.maximum(-p[1], p[1] - src_h), 0).max()
            res.append((k, L["clip"], float(max(ox, oy))))
    return res
