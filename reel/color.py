"""Colour helpers: cross-take colour matching, grading primitives."""
import cv2
import numpy as np


def _corresponding_mask(src, dst):
    g = cv2.cvtColor((dst * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = cv2.GaussianBlur(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), (9, 9), 0)
    ok = grad < np.percentile(grad, 70)
    diff = np.abs(cv2.GaussianBlur(src, (15, 15), 0) - cv2.GaussianBlur(dst, (15, 15), 0)).sum(-1)
    return ok & (diff < np.percentile(diff, 75))


def fit_color_match(src, dst):
    """Conservative take-to-take match of `src` onto `dst` (both RGB float, roughly aligned).

    Lightness: monotonic quantile curve. Chroma (a, b): mean/std linear transfer. Saturated colours
    keep their hue, which a polynomial RGB fit does not guarantee.
    """
    ok = _corresponding_mask(src, dst)
    ls, ld = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)[ok], cv2.cvtColor(dst, cv2.COLOR_RGB2LAB)[ok]
    q = np.linspace(0, 100, 33)
    curve_x, curve_y = np.percentile(ls[:, 0], q), np.percentile(ld[:, 0], q)
    curve_x = np.maximum.accumulate(curve_x + np.arange(len(q)) * 1e-4)
    curve_y = np.maximum.accumulate(curve_y)
    ab = [(ls[:, c].mean(), ls[:, c].std(), ld[:, c].mean(), ld[:, c].std()) for c in (1, 2)]
    return dict(lx=curve_x.tolist(), ly=curve_y.tolist(), ab=ab)


def apply_color_match(img, P, amount=1.0):
    lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)
    out = lab.copy()
    x = np.concatenate([[0], P["lx"], [100]])
    y = np.concatenate([[0], P["ly"], [100]])
    out[..., 0] = np.interp(lab[..., 0], x, y)
    for c, (ms, ss, md, sd) in zip((1, 2), P["ab"]):
        out[..., c] = (lab[..., c] - ms) * np.clip(sd / max(ss, 1e-3), 0.8, 1.25) + md
    out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB).clip(0, 1)
    return out if amount >= 1 else img * (1 - amount) + out * amount
