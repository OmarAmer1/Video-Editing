"""Image effects at output resolution (float32 RGB in [0, 1], HxWx3)."""
import cv2
import numpy as np


def ease_in_out_sine(u):
    u = np.clip(u, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def smoothstep(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def smootherstep(u):
    u = np.clip(u, 0, 1)
    return u * u * u * (u * (6 * u - 15) + 10)


def ease_out_cubic(u):
    u = np.clip(u, 0, 1)
    return 1 - (1 - u) ** 3


def ease_out_quint(u):
    u = np.clip(u, 0, 1)
    return 1 - (1 - u) ** 5


def ease_in_out_cubic(u):
    u = np.clip(u, 0, 1)
    return np.where(u < 0.5, 4 * u ** 3, 1 - (-2 * u + 2) ** 3 / 2)


def ramp(t, t0, t1, ease=smoothstep):
    if t1 <= t0:
        return float(t >= t1)
    return float(ease((t - t0) / (t1 - t0)))


def luma(img):
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def curves(img, points):
    """Per-channel tone curve through (x, y) control points (monotone cubic)."""
    from scipy.interpolate import PchipInterpolator

    xs, ys = zip(*points)
    f = PchipInterpolator(xs, ys)
    lut = np.clip(f(np.linspace(0, 1, 1024)), 0, 1).astype(np.float32)
    idx = np.clip((img * 1023).astype(np.int32), 0, 1023)
    return lut[idx]


def saturation(img, s):
    y = luma(img)[..., None]
    return np.clip(y + (img - y) * s, 0, 1)


def split_tone(img, shadows=(0.0, 0.0, 0.0), highlights=(0.0, 0.0, 0.0), balance=0.5):
    y = luma(img)[..., None]
    w_hi = np.clip((y - balance) * 2 + 0.5, 0, 1)
    return np.clip(img + np.array(shadows, np.float32) * (1 - w_hi) + np.array(highlights, np.float32) * w_hi, 0, 1)


def mono(img, mix=(0.28, 0.60, 0.12),
         pts=((0, 0.035), (0.12, 0.10), (0.35, 0.36), (0.6, 0.62), (0.8, 0.78), (1, 0.93)),
         tint_shadow=(0.92, 0.97, 1.06), tint_high=(1.05, 1.0, 0.92), split=0.5):
    """Silver monochrome: green-weighted mix keeps skin tonal, soft highlight knee keeps the hijab from
    clipping, cool-ink shadows / warm-paper highlights split tone."""
    y = img[..., 0] * mix[0] + img[..., 1] * mix[1] + img[..., 2] * mix[2]
    y = curves(y, list(pts))
    w = y[..., None]
    tint = np.array(tint_shadow, np.float32) * (1 - w) + np.array(tint_high, np.float32) * w
    tint = 1 + (tint - 1) * split
    return np.clip(y[..., None] * tint, 0, 1)


def bloom(img, threshold=0.75, radius=25, strength=0.5, tint=(1.0, 0.92, 0.8)):
    """Additive glow from highlights (halation-style: warm tint)."""
    y = luma(img)
    m = np.clip((y - threshold) / max(1 - threshold, 1e-3), 0, 1)[..., None] * img
    small = cv2.resize(m, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    r = max(radius / 4, 1)
    g = cv2.GaussianBlur(small, (0, 0), r) * 0.6 + cv2.GaussianBlur(small, (0, 0), r * 3) * 0.4
    g = cv2.resize(g, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    g *= np.array(tint, np.float32)
    return 1 - (1 - img) * (1 - np.clip(g * strength, 0, 1))  # screen


def radial_glow(shape, center, radius, color=(1.0, 0.79, 0.54), strength=0.35, falloff=2.0):
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / max(radius, 1)
    g = np.exp(-d ** falloff * 2.5) * strength
    return g[..., None] * np.array(color, np.float32)


def screen(img, layer):
    return 1 - (1 - img) * (1 - np.clip(layer, 0, 1))


def radial_mask(shape, center, radius, feather=0.45):
    """1 inside radius, soft edge of `feather`*radius."""
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    f = max(radius * feather, 1)
    return np.clip((radius - d) / f + 0.5, 0, 1)


def vignette(img, strength=0.35, roundness=1.0, softness=0.6):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xx / w - 0.5) * 2, (yy / h - 0.5) * 2 * (h / w) ** (1 - roundness) * (w / h) ** 0
    ny = (yy / h - 0.5) * 2
    d = np.sqrt(nx ** 2 * 0.8 + ny ** 2 * 0.6)
    v = 1 - strength * smoothstep((d - (1 - softness)) / softness)
    return img * v[..., None]


class Grain:
    """Temporal film grain: luminance-weighted, slightly soft, fixed seed per frame index."""

    def __init__(self, shape, seed=0):
        self.h, self.w = shape[:2]
        self.seed = seed

    def apply(self, img, k, amount=0.04, size=1.4, chroma=0.15):
        r = np.random.default_rng(self.seed * 100003 + k)
        hs, ws = int(self.h / size), int(self.w / size)
        n = r.standard_normal((hs, ws, 3)).astype(np.float32)
        n = n[..., :1] * (1 - chroma) + n * chroma
        n = cv2.resize(n, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        y = luma(img)[..., None]
        wgt = 4 * y * (1 - y) * 0.8 + 0.2  # strongest in mid-tones
        return np.clip(img + n * amount * wgt, 0, 1)


def rounded_rect_mask(shape, rect, radius, feather=2.0):
    """rect=(x0, y0, x1, y1) in pixels; anti-aliased rounded rectangle mask."""
    h, w = shape[:2]
    x0, y0, x1, y1 = rect
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = (x1 - x0) / 2 - radius, (y1 - y0) / 2 - radius
    qx, qy = np.abs(xx - cx) - hx, np.abs(yy - cy) - hy
    d = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - radius
    return np.clip(0.5 - d / feather, 0, 1)


def film_gate(img, open_amount=0.0, radius=60, inset=(42, 56), border_color=(0.02, 0.02, 0.02), edge_glow=0.08):
    """Vintage rounded film-gate frame (echoes the client's original edit). open_amount 0 = framed, 1 = gone."""
    h, w = img.shape[:2]
    s = 1 + 0.18 * open_amount
    ix, iy = inset[0] * (1 - open_amount) - 0.0, inset[1] * (1 - open_amount)
    cx, cy = w / 2, h / 2
    hw, hh = (w / 2 - ix) * s, (h / 2 - iy) * s
    m = rounded_rect_mask(img.shape, (cx - hw, cy - hh, cx + hw, cy + hh), radius * (1 + open_amount), feather=6)
    inner_soft = cv2.GaussianBlur(m, (0, 0), 18)
    shade = 1 - 0.35 * (1 - inner_soft) * (1 - open_amount)
    out = img * shade[..., None]
    glow = np.clip(cv2.GaussianBlur(1 - m, (0, 0), 6) * m, 0, 1)[..., None] * edge_glow * (1 - open_amount)
    out = out + glow
    bc = np.array(border_color, np.float32)
    return out * m[..., None] + bc * (1 - m[..., None])


def chromatic_aberration(img, amount=1.5):
    if amount <= 0:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    for c, s in ((0, 1 + amount / w), (2, 1 - amount / w)):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
        out[..., c] = cv2.warpAffine(img[..., c], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def gate_weave(k, amp=1.2, seed=3):
    """Small per-frame translation (px) imitating film gate weave."""
    r = np.random.default_rng(seed)
    ph = r.uniform(0, 2 * np.pi, 4)
    return (amp * (0.6 * np.sin(k * 0.37 + ph[0]) + 0.4 * np.sin(k * 1.13 + ph[1])),
            amp * (0.6 * np.sin(k * 0.29 + ph[2]) + 0.4 * np.sin(k * 0.97 + ph[3])))


def flicker(k, amount=0.025, seed=4):
    r = np.random.default_rng(seed * 7919 + k)
    return 1 + amount * (r.standard_normal() * 0.6 + np.sin(k * 0.9) * 0.4)


def dust(shape, k, density=0.6, seed=5):
    """Sparse film dust/specks/hairs layer (returns HxW alpha of bright specks)."""
    h, w = shape[:2]
    r = np.random.default_rng(seed * 104729 + k)
    layer = np.zeros((h, w), np.float32)
    for _ in range(r.poisson(density * 3)):
        x, y = r.integers(0, w), r.integers(0, h)
        rad = r.uniform(1.0, 3.0)
        cv2.circle(layer, (int(x), int(y)), int(rad), float(r.uniform(0.3, 0.8)), -1, lineType=cv2.LINE_AA)
    if r.random() < 0.15 * density:
        x = r.integers(0, w)
        pts = np.array([[x + r.integers(-20, 20), r.integers(0, h)] for _ in range(4)], np.int32)
        cv2.polylines(layer, [pts], False, float(r.uniform(0.2, 0.5)), 1, lineType=cv2.LINE_AA)
    return cv2.GaussianBlur(layer, (0, 0), 0.7)


# ------------------------------------------------------------------------------- matte-aware effects


def bg_blur(img, matte, sigma):
    """Synthetic shallow focus: blur only the background (normalised convolution, no halo)."""
    if sigma <= 0.3:
        return img
    bgw = np.clip(1 - matte, 0, 1)[..., None]
    s = 0.5
    small_i = cv2.resize(img * bgw, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    small_w = cv2.resize(bgw, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if small_w.ndim == 2:
        small_w = small_w[..., None]
    bi = cv2.GaussianBlur(small_i, (0, 0), sigma * s)
    bw = cv2.GaussianBlur(small_w, (0, 0), sigma * s)
    if bw.ndim == 2:
        bw = bw[..., None]
    blurred = bi / np.maximum(bw, 1e-4)
    blurred = cv2.resize(blurred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    m = matte[..., None]
    return img * m + blurred * (1 - m)


def flame_mask(img, center, px=2.7, lum=0.78):
    """Soft mask of candle flames above `center` (output px; px = output pixels per source pixel).

    Flames are thin, very bright structures: a white top-hat (bright detail smaller than ~9 source px)
    combined with a brightness gate, inside a feathered window above the candle.
    """
    h, w = img.shape[:2]
    out = np.zeros((h, w), np.float32)
    if center is None:
        return out
    x0, x1 = int(max(center[0] - 70 * px, 0)), int(min(center[0] + 70 * px, w))
    y0, y1 = int(max(center[1] - 62 * px, 0)), int(min(center[1] + 6 * px, h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return out
    roi = img[y0:y1, x0:x1]
    y = luma(roi)
    k = int(9 * px) | 1
    op = cv2.morphologyEx(y, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = np.clip((y - op - 0.05) / 0.08, 0, 1) * np.clip((y - lum) / 0.1, 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), px * 0.8)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    win = np.clip((xx - x0) / (8 * px), 0, 1) * np.clip((x1 - xx) / (8 * px), 0, 1)
    win *= np.clip((yy - y0) / (8 * px), 0, 1) * np.clip((center[1] + 6 * px - yy) / (14 * px), 0, 1)
    out[y0:y1, x0:x1] = np.clip(m * 1.5 * win, 0, 1)
    return out


def flame_glow(base, mask, strength=1.0, core=(1.0, 0.66, 0.28), inner=(1.0, 0.58, 0.20), outer=(1.0, 0.50, 0.16),
               px=2.7):
    """Re-light candle flames as amber emitters over `base` (e.g. the monochrome image).

    The flame is colourised *keeping its luminance* (an additive tint would just push the already-white
    flame further to white), with a small white-hot centre, then a warm two-radius glow is screened on top."""
    if strength <= 0 or mask.max() <= 0:
        return base
    m = np.clip(mask * min(strength, 1.0), 0, 1)
    y = luma(base)[..., None]
    col = np.clip(y * 1.05 * np.array(core, np.float32), 0, 1)
    k = max(int(2 * px) | 1, 3)
    hot = cv2.GaussianBlur(cv2.erode(mask, np.ones((k, k), np.uint8)), (0, 0), max(px * 0.8, 0.5)) ** 2
    hot = hot[..., None]
    col = col * (1 - 0.45 * hot) + np.array((1.0, 0.90, 0.66), np.float32) * 0.45 * hot
    lit = base * (1 - m[..., None]) + col * m[..., None]
    g = (cv2.GaussianBlur(mask, (0, 0), 5 * px)[..., None] * np.array(inner, np.float32) * 1.6 +
         cv2.GaussianBlur(mask, (0, 0), 15 * px)[..., None] * np.array(outer, np.float32) * 1.1) * strength
    return screen(lit, g)


class Particles:
    """Deterministic particle systems (bokeh orbs, embers, sparkles) rendered as soft discs."""

    def __init__(self, n, seed, spawn, life, vel, size, color, opacity, sway=0.0):
        r = np.random.default_rng(seed)
        self.p = []
        for i in range(n):
            self.p.append(dict(
                t0=spawn(r, i), life=life(r), pos=None, seed=r.integers(1 << 30),
                vel=vel(r), size=size(r), color=color(r), opacity=opacity(r), ph=r.uniform(0, 2 * np.pi),
                sway=sway * r.uniform(0.5, 1.5)))
        self._origin = None

    def render(self, shape, t, origin_fn, fade_in=0.25, fade_out=0.45, blur_scale=0.6):
        """origin_fn(particle, r) -> start position (x, y). Returns premultiplied RGBA layer."""
        h, w = shape[:2]
        layer = np.zeros((h, w, 3), np.float32)
        alpha = np.zeros((h, w), np.float32)
        for q in self.p:
            age = t - q["t0"]
            if age < 0 or age > q["life"]:
                continue
            if q["pos"] is None:
                q["pos"] = np.array(origin_fn(q, np.random.default_rng(q["seed"])), np.float64)
            u = age / q["life"]
            a = q["opacity"] * min(1, u / fade_in if fade_in > 0 else 1) * min(1, (1 - u) / fade_out if fade_out > 0 else 1)
            x = q["pos"][0] + q["vel"][0] * age + q["sway"] * np.sin(age * 2.2 + q["ph"])
            y = q["pos"][1] + q["vel"][1] * age
            rad = q["size"]
            pad = int(rad * 3 + 4)
            xi, yi = int(x), int(y)
            if xi + pad < 0 or xi - pad >= w or yi + pad < 0 or yi - pad >= h:
                continue
            ys, xs = np.mgrid[max(yi - pad, 0):min(yi + pad, h), max(xi - pad, 0):min(xi + pad, w)].astype(np.float32)
            d = np.sqrt((xs - x) ** 2 + (ys - y) ** 2)
            disc = np.clip((rad - d) / max(rad * blur_scale, 0.8) + 0.5, 0, 1) * a
            disc += np.exp(-(d / (rad * 2.2)) ** 2) * a * 0.25  # soft halo
            sl = (slice(max(yi - pad, 0), min(yi + pad, h)), slice(max(xi - pad, 0), min(xi + pad, w)))
            layer[sl] += disc[..., None] * np.array(q["color"], np.float32)
            alpha[sl] = 1 - (1 - alpha[sl]) * (1 - np.clip(disc, 0, 1))
        return layer, alpha


def add_light(img, rgb_layer, gain=1.0):
    """Screen-add an emissive RGB layer."""
    return 1 - (1 - img) * (1 - np.clip(rgb_layer * gain, 0, 1))


def ring(shape, center, radius, width, color=(1.0, 0.886, 0.69), strength=0.3):
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    g = np.exp(-((d - radius) / max(width, 1)) ** 2) * strength
    return g[..., None] * np.array(color, np.float32)


def light_leak(shape, t, seed=0, strength=0.5, color=(1.0, 0.70, 0.42), pos=(0.9, 0.15), scale=0.8):
    """Soft organic warm light leak drifting slowly (screen it over the image)."""
    h, w = shape[:2]
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h:4, 0:w:4].astype(np.float32)
    acc = np.zeros_like(xx)
    for i in range(3):
        cx = (pos[0] + 0.12 * np.sin(t * 0.7 + i * 2.1 + r.uniform(0, 6))) * w
        cy = (pos[1] + 0.10 * np.cos(t * 0.5 + i * 1.3 + r.uniform(0, 6))) * h
        rad = scale * w * (0.35 + 0.15 * i)
        acc += np.exp(-(((xx - cx) ** 2 + ((yy - cy) * 0.8) ** 2) / rad ** 2)) * (1.0 - 0.25 * i)
    acc = cv2.resize(acc, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(acc * strength, 0, 1)[..., None] * np.array(color, np.float32)


def value_noise(shape, scale, seed=0, octaves=2):
    """Smooth 2D value noise in [0, 1] (bilinear-upsampled random grids)."""
    h, w = shape[:2]
    r = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = scale / (2 ** o)
        gh, gw = max(int(h / s) + 2, 2), max(int(w / s) + 2, 2)
        g = r.random((gh, gw)).astype(np.float32)
        acc += cv2.resize(g, (int(gw * s), int(gh * s)), interpolation=cv2.INTER_CUBIC)[:h, :w] * amp
        tot += amp
        amp *= 0.5
    return np.clip(acc / tot, 0, 1)


def wave_front(shape, center, radius, feather, noise=None, noise_amp=0.0):
    """Expanding wavefront weight map: 1 inside radius (already transformed), 0 outside.

    `noise` (HxW in [0,1]) perturbs the edge by +/- noise_amp px for an organic, non-geometric front.
    """
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    if noise is not None and noise_amp > 0:
        d = d + (noise - 0.5) * 2 * noise_amp
    return np.clip((radius - d) / max(feather, 1) + 0.5, 0, 1)


def present_grade(img, warmth=0.035, contrast_pts=((0, 0.025), (0.18, 0.165), (0.5, 0.52), (0.82, 0.86), (1, 0.975)),
                  sky=(0.043, 0.07, 0.125), sat=1.06, halation=0.18, bloom_amt=0.10):
    """Warm 'golden present' night grade: gentle warm balance, filmic curve, navy-ink blacks,
    gold-leaning highlights, halation on lights. Skin and the lavender hijab are left near-natural."""
    out = img.copy()
    y = luma(out)[..., None]
    # warm the highlights, keep a hint of teal-ink in the shadows
    out = out + np.array([warmth, warmth * 0.25, -warmth], np.float32) * y - np.array([0.004, -0.002, -0.012], np.float32) * (1 - y)
    out = curves(np.clip(out, 0, 1), list(contrast_pts))
    # deep blacks toward navy ink
    d = np.clip(1 - luma(out) / 0.10, 0, 1)[..., None]
    out = out * (1 - d * 0.6) + np.array(sky, np.float32) * d * 0.6 + out * d * 0.0
    out = saturation(out, sat)
    out = bloom(out, threshold=0.80, radius=22, strength=halation, tint=(1.0, 0.42, 0.25))
    out = bloom(out, threshold=0.90, radius=60, strength=bloom_amt, tint=(1.0, 0.85, 0.7))
    return np.clip(out, 0, 1)


def _hex(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def hue_band(hsv, lo, hi, soft=8.0):
    """Soft membership of HSV hue (degrees) in [lo, hi]; lo > hi wraps through 0."""
    h = hsv[..., 0]
    if lo <= hi:
        return np.clip((h - lo) / soft, 0, 1) * np.clip((hi - h) / soft, 0, 1)
    return np.maximum(np.clip((h - lo) / soft, 0, 1), np.clip((hi - h) / soft, 0, 1))


def hijab_mask(img, hsv=None):
    """Her dusty mauve-pink hijab: hue ~320-5 deg, low-mid saturation (lips/skin are far more saturated)."""
    if hsv is None:
        hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    return (hue_band(hsv, 318, 6, 10) * np.clip((s - 0.07) / 0.04, 0, 1) * np.clip((0.40 - s) / 0.06, 0, 1)
            * np.clip((v - 0.30) / 0.08, 0, 1) * np.clip((0.86 - v) / 0.06, 0, 1)).astype(np.float32)   # not candle wax


def mono_silver(img, mix=(0.40, 0.48, 0.12), pts=((0, 0.04), (0.18, 0.18), (0.45, 0.52), (0.75, 0.80), (1, 0.95)),
                shadow="#1C2026", highlight="#F1E4CF", split=0.12, halation=0.15, local_contrast=0.15, lc_radius=30,
                hijab_darken=0.26):
    """'Silver memory' monochrome: luminous skin (red-leaning 'orange filter' portrait mix), the hijab gently
    darkened by hue so face and scarf separate, cool-ink shadows / warm-paper highlights."""
    y = img[..., 0] * mix[0] + img[..., 1] * mix[1] + img[..., 2] * mix[2]
    if hijab_darken > 0:
        sig = 3.0 * img.shape[1] / 464                       # smooth over compression blocks
        y = y * (1 - hijab_darken * cv2.GaussianBlur(hijab_mask(img), (0, 0), sig))
    if local_contrast > 0:
        r = lc_radius * img.shape[1] / 1080
        y = y + (y - cv2.GaussianBlur(y, (0, 0), r)) * local_contrast
    y = curves(np.clip(y, 0, 1), list(pts))
    base = np.repeat(y[..., None], 3, -1)
    sh, hi = _hex(shadow), _hex(highlight)
    w = y[..., None]
    toned = base * (1 - split) + split * (base * (hi / hi.max()) * w + (sh + base * (1 - sh)) * (1 - w))
    out = np.clip(toned, 0, 1)
    if halation > 0:
        out = bloom(out, threshold=0.85, radius=14, strength=halation, tint=tuple(_hex("#F3D9B0")))
    return out


def flicker_smooth(k, amount=0.012, fps=30.0, seed=4):
    """Exposure flicker as smooth noise below 6 Hz (sum of slow sines), never frame-random."""
    r = np.random.default_rng(seed)
    t = k / fps
    f = r.uniform(1.0, 5.5, 4)
    ph = r.uniform(0, 2 * np.pi, 4)
    v = sum(np.sin(2 * np.pi * fi * t + p) for fi, p in zip(f, ph)) / 2.0
    return 1 + amount * float(v)


def film_gate_rect(img, inset, radius, feather=4.0, outside=(0.039, 0.039, 0.043)):
    """Rounded-rectangle film gate. inset < 0 pushes the frame edge beyond the picture (gate burst)."""
    h, w = img.shape[:2]
    if inset <= -feather * 2 and radius <= 1:
        return img
    m = rounded_rect_mask(img.shape, (inset, inset, w - inset, h - inset), max(radius, 0.5), feather=feather)
    return img * m[..., None] + np.array(outside, np.float32) * (1 - m[..., None])


def present_grade_v2(img, warmth=0.008, pts=((0, 0.025), (0.18, 0.16), (0.5, 0.52), (0.82, 0.86), (1, 0.97)),
                     sky="#0B1220", halation=0.18, bloom_amt=0.12):
    """'Golden present' night grade with hue-qualified protection:
    lavender hijab (hue 270-320) keeps its exact hue/saturation, skin (15-35) +4% sat,
    warm hotel lights (30-60, bright) pushed toward gold +10% sat, sky blacks toward teal-ink."""
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)       # H in degrees, S/V in [0,1]
    hue, sat = hsv[..., 0], hsv[..., 1]
    y0 = luma(img)
    out = img + (np.array([1.0, 0.25, -0.75], np.float32) * warmth) * y0[..., None] \
        + np.array([0.0, 0.01, 0.02], np.float32) * (1 - y0[..., None]) * 0.5
    out = curves(np.clip(out, 0, 1), list(pts))
    # navy/teal ink in the darkest areas (the night sky)
    d = np.clip(1 - luma(out) / 0.08, 0, 1)[..., None]
    out = out * (1 - d * 0.5) + _hex(sky) * d * 0.5
    # hue-qualified saturation
    band = lambda lo, hi, soft=8: np.clip((hue - lo) / soft, 0, 1) * np.clip((hi - hue) / soft, 0, 1)
    skin = band(8, 38) * np.clip((sat - 0.35) / 0.1, 0, 1)
    lights = band(30, 60) * np.clip((y0 - 0.6) / 0.15, 0, 1) * (1 - skin)
    gain = 1 - 0.15 * skin + 0.10 * lights
    yy = luma(out)[..., None]
    out = yy + (out - yy) * gain[..., None]
    # lock the hijab: restore its original chroma relative to the graded lightness
    hij = cv2.GaussianBlur(hijab_mask(img, hsv), (0, 0), 2.0 * img.shape[1] / 464)[..., None]
    if hij.max() > 0:
        y_in = y0[..., None]
        restored = yy + (img - y_in)            # original chroma on graded luma
        out = out * (1 - hij) + restored * hij
    out = np.clip(out, 0, 1)
    out = bloom(out, threshold=0.82, radius=24, strength=halation, tint=tuple(_hex("#FF5A36")))
    out = bloom(out, threshold=0.90, radius=60, strength=bloom_amt, tint=(1.0, 0.86, 0.72))
    return np.clip(out, 0, 1)


def unsharp(img, radius=1.2, amount=0.25):
    b = cv2.GaussianBlur(img, (0, 0), radius)
    return np.clip(img + (img - b) * amount, 0, 1)
