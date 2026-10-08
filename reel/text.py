"""Animated typography rendered with Pillow onto RGBA layers (supports variable-font axes)."""
import functools

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import paths


@functools.lru_cache(maxsize=None)
def font(file, size, wght=None, opsz=None):
    f = ImageFont.truetype(str(paths.FONTS / file), int(round(size)))
    try:
        axes = {a["name"].lower() if isinstance(a["name"], str) else a["name"].decode().lower(): a
                for a in f.get_variation_axes()}
    except Exception:
        axes = {}
    if axes:
        vals = []
        for name, a in axes.items():
            if name.startswith("weight") and wght is not None:
                vals.append(float(np.clip(wght, a["minimum"], a["maximum"])))
            elif name.startswith("optical") and opsz is not None:
                vals.append(float(np.clip(opsz, a["minimum"], a["maximum"])))
            else:
                vals.append(float(a["default"]))
        f.set_variation_by_axes(vals)
    return f


def text_width(text, fnt, tracking_em=0.0):
    size = fnt.size
    w = 0.0
    for i, ch in enumerate(text):
        w += fnt.getlength(ch)
        if i < len(text) - 1:
            w += tracking_em * size
    return w


def draw_text_layer(shape, text, file, size, x, baseline, color=(1, 1, 1), opacity=1.0, wght=None,
                    tracking_em=0.0, blur=0.0, dy=0.0, align="left", shadow=(0, 2, 12, 0.45), glow=None,
                    scale=1.0, letter_progress=None):
    """Return an HxWx4 float32 premultiplied RGBA layer with the text.

    letter_progress: optional callable(i, n) -> opacity multiplier per character (for staggered reveals).
    glow: optional (radius_px, (r, g, b), strength) outer glow.
    """
    h, w = shape[:2]
    fnt = font(file, size * scale, wght)
    tw = text_width(text, fnt, tracking_em)
    if align == "center":
        x = x - tw / 2
    elif align == "right":
        x = x - tw
    ascent, descent = fnt.getmetrics()
    y_top = baseline + dy - ascent
    alpha = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(alpha)
    cx = x
    n = len(text)
    for i, ch in enumerate(text):
        a = 255
        if letter_progress is not None:
            a = int(255 * float(np.clip(letter_progress(i, n), 0, 1)))
        if a > 0 and ch != " ":
            d.text((cx, y_top), ch, font=fnt, fill=a)
        cx += fnt.getlength(ch) + tracking_em * fnt.size
    A = np.asarray(alpha, np.float32) / 255
    if blur > 0.05:
        A = cv2.GaussianBlur(A, (0, 0), blur)
    out = np.zeros((h, w, 4), np.float32)
    if shadow is not None and shadow[3] > 0:
        sx, sy, sb, so = shadow
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        S = cv2.warpAffine(A, M, (w, h))
        S = cv2.GaussianBlur(S, (0, 0), max(sb, 0.1)) * so * opacity
        out[..., 3] = S  # black shadow: rgb stays 0 (premultiplied)
    if glow is not None:
        gr, gc, gs = glow
        G = cv2.GaussianBlur(A, (0, 0), gr) * gs * opacity
        out = over(out, np.dstack([G[..., None] * np.array(gc, np.float32), G]))
    col = np.array(color, np.float32)
    T = A * opacity
    out = over(out, np.dstack([T[..., None] * col, T]))
    return out


def over(dst, src):
    """Premultiplied 'over' of two RGBA layers."""
    a = src[..., 3:4]
    return src + dst * (1 - a)


def composite(img, layer):
    """Composite a premultiplied RGBA layer over an RGB image."""
    return img * (1 - layer[..., 3:4]) + layer[..., :3]


def bbox(layer, thresh=0.02):
    ys, xs = np.nonzero(layer[..., 3] > thresh)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def reveal(t, t0, dur, ease=None):
    """0..1 progress of a reveal starting at t0 lasting dur (easeOutCubic by default)."""
    u = float(np.clip((t - t0) / max(dur, 1e-6), 0, 1))
    return 1 - (1 - u) ** 3 if ease is None else float(ease(u))


def animated_line(shape, t, words, t_in, file, size, x, baseline, color, wght=None, stagger=0.12, dur=0.6,
                  rise=14, blur_in=10, t_out=None, out_dur=0.5, blur_out=8, opacity=1.0, glow=None, align="left",
                  tracking_em=0.0, shadow=(0, 2, 12, 0.45)):
    """Render a line whose words blur/rise/fade in one after another, and blur/fade out together."""
    fnt = font(file, size, wght)
    space = fnt.getlength(" ")
    widths = [text_width(w_, fnt, tracking_em) for w_ in words]
    total = sum(widths) + space * (len(words) - 1)
    x0 = x - total / 2 if align == "center" else (x - total if align == "right" else x)
    layer = np.zeros((shape[0], shape[1], 4), np.float32)
    out_p = 0.0 if t_out is None else reveal(t, t_out, out_dur, ease=lambda u: u * u * (3 - 2 * u))
    cx = x0
    for i, (w_, ww) in enumerate(zip(words, widths)):
        p = reveal(t, t_in + i * stagger, dur)
        a = p * (1 - out_p) * opacity
        if a > 0.004:
            L = draw_text_layer(shape, w_, file, size, cx, baseline, color=color, opacity=a, wght=wght,
                                tracking_em=tracking_em, blur=blur_in * (1 - p) + blur_out * out_p,
                                dy=rise * (1 - p), glow=glow, shadow=shadow)
            layer = over(layer, L)
        cx += ww + space
    return layer
