"""Source frame access with fractional (RIFE-interpolated) positions, plus geometry helpers."""
import functools

import cv2
import numpy as np

from . import paths
from .rife import Rife

SRC_W, SRC_H = 464, 832


def mat3(M):
    M = np.asarray(M, np.float64)
    return M if M.shape == (3, 3) else np.vstack([M, [0, 0, 1]])


def similarity(scale=1.0, rot_deg=0.0, tx=0.0, ty=0.0, cx=0.0, cy=0.0):
    """3x3 similarity: rotate/scale about (cx, cy), then translate."""
    a = np.radians(rot_deg)
    c, s = scale * np.cos(a), scale * np.sin(a)
    return np.array([[c, -s, cx - c * cx + s * cy + tx], [s, c, cy - s * cx - c * cy + ty], [0, 0, 1]])


def decompose(M):
    M = mat3(M)
    s = np.sqrt(abs(np.linalg.det(M[:2, :2])))
    return s, np.degrees(np.arctan2(M[1, 0], M[0, 0])), M[0, 2], M[1, 2]


def lerp_sim(A, B, t, anchor=None):
    """Interpolate two similarity transforms in (log-scale, angle, path-of-anchor) space.

    The anchor point (default: frame centre) travels on a straight line between its two images.
    """
    sa, ra, _, _ = decompose(A)
    sb, rb, _, _ = decompose(B)
    c = np.array([SRC_W / 2, SRC_H / 2, 1.0]) if anchor is None else np.array([anchor[0], anchor[1], 1.0])
    pa, pb = mat3(A) @ c, mat3(B) @ c
    s = np.exp(np.log(sa) * (1 - t) + np.log(sb) * t)
    r = ra * (1 - t) + rb * t
    p = pa * (1 - t) + pb * t
    R = similarity(s, r, cx=0, cy=0)
    R[0, 2], R[1, 2] = p[0] - (R[0, 0] * c[0] + R[0, 1] * c[1]), p[1] - (R[1, 0] * c[0] + R[1, 1] * c[1])
    return R


class Clip:
    """Frames of one clip as float32 RGB; frame(f) accepts fractional f (RIFE in between)."""

    def __init__(self, name, rife=None):
        self.name = name
        self.files = sorted(paths.frames_dir(name).glob("*.png"))
        self.n = len(self.files)
        self.rife = rife
        self._interp = {}

    @functools.lru_cache(maxsize=None)
    def exact(self, i):
        i = int(np.clip(i, 0, self.n - 1))
        return cv2.cvtColor(cv2.imread(str(self.files[i])), cv2.COLOR_BGR2RGB).astype(np.float32) / 255

    @functools.lru_cache(maxsize=None)
    def matte(self, i):
        i = int(np.clip(i, 0, self.n - 1))
        return cv2.imread(str(paths.matte_dir(self.name) / self.files[i].name), 0).astype(np.float32) / 255

    def frame(self, f, ensemble=False):
        f = float(np.clip(f, 0, self.n - 1))
        i0 = int(np.floor(f + 1e-6))
        t = round(f - i0, 4)
        if t < 0.02 or i0 >= self.n - 1:
            return self.exact(i0)
        if t > 0.98:
            return self.exact(i0 + 1)
        key = (i0, round(t, 3), ensemble)
        if key not in self._interp:
            self._interp[key] = self.rife.interpolate(self.exact(i0), self.exact(i0 + 1), t, ensemble=ensemble)
        return self._interp[key]

    def matte_at(self, f):
        f = float(np.clip(f, 0, self.n - 1))
        i0 = int(np.floor(f))
        t = f - i0
        if i0 >= self.n - 1:
            return self.matte(i0)
        return self.matte(i0) * (1 - t) + self.matte(i0 + 1) * t


def bg_track(analysis, clip, f):
    """Background transform frame0 -> frame f (fractional f linearly blended)."""
    bg = analysis[clip]["bg"]
    i0 = int(np.clip(np.floor(f), 0, len(bg) - 1))
    i1 = min(i0 + 1, len(bg) - 1)
    t = float(np.clip(f - i0, 0, 1))
    return lerp_sim(mat3(bg[i0]), mat3(bg[i1]), t)


def stabilizer(analysis, clip, f, ref):
    """Transform taking frame f into the background coordinates of frame `ref` of the same clip."""
    return bg_track(analysis, clip, ref) @ np.linalg.inv(bg_track(analysis, clip, f))


def load_rife():
    return Rife(str(paths.MODELS / "flownet_v4.25.pkl"))
