"""Keyframed curves for time-remapping (speed ramps) and the virtual camera."""
import numpy as np
from scipy.interpolate import PchipInterpolator

from .frames import similarity


class Curve:
    """Monotone-cubic (PCHIP) interpolation through (t, value) keys; clamps outside the key range.

    Used for time remaps: keys (t_out, src_frame) give smooth speed ramps whose speed is the slope.
    """

    def __init__(self, keys):
        keys = sorted(keys)
        self.t = np.array([k[0] for k in keys], np.float64)
        self.v = np.array([k[1] for k in keys], np.float64)
        self.f = PchipInterpolator(self.t, self.v) if len(keys) > 1 else None

    def __call__(self, t):
        if self.f is None:
            return float(self.v[0])
        return float(self.f(np.clip(t, self.t[0], self.t[-1])))

    def speed(self, t, fps=30.0):
        """Source frames advanced per output frame at time t (1.0 = real time)."""
        dt = 1e-3
        return (self(t + dt) - self(t - dt)) / (2 * dt) / fps


class EasedKeys:
    """Piecewise interpolation of vectors with a per-segment easing function (for camera moves)."""

    def __init__(self, keys):
        """keys: list of (t, np.array(values), ease) — `ease` shapes the segment that ENDS at this key."""
        self.keys = sorted(keys, key=lambda k: k[0])

    def __call__(self, t):
        ks = self.keys
        if t <= ks[0][0]:
            return np.asarray(ks[0][1], np.float64)
        for (t0, v0, _), (t1, v1, ease) in zip(ks, ks[1:]):
            if t <= t1:
                u = (t - t0) / max(t1 - t0, 1e-9)
                e = float(ease(u)) if ease is not None else u
                return np.asarray(v0, np.float64) * (1 - e) + np.asarray(v1, np.float64) * e
        return np.asarray(ks[-1][1], np.float64)


def camera_matrix(zoom, cx, cy, roll_deg, src_w, src_h, out_w, out_h):
    """3x3 map from reference (source-pixel) space to output pixels.

    The camera looks at (cx, cy); zoom=1 shows the full source width; roll rotates the picture
    (positive = content turns clockwise on screen).
    """
    s = zoom * out_w / src_w
    T1 = similarity(1, 0, -cx, -cy)
    R = similarity(s, roll_deg)
    T2 = similarity(1, 0, out_w / 2, out_h / 2)
    return T2 @ R @ T1


def min_zoom_for_roll(roll_deg, src_w, src_h, aspect_w=9, aspect_h=16):
    """Smallest zoom whose rotated output rectangle still fits inside the source frame (centred)."""
    a = np.radians(abs(roll_deg))
    out_w, out_h = src_w, src_w * aspect_h / aspect_w
    need_w = out_w * np.cos(a) + out_h * np.sin(a)
    need_h = out_w * np.sin(a) + out_h * np.cos(a)
    return max(need_w / src_w, need_h / src_h)


class SpeedRamp:
    """Time remap defined by playback speed (1.0 = real time) keyed over output time.

    keys: [(t_out, speed), ...] interpolated with a smoothstep between keys (so ramps have no kinks);
    start: (t0, f0) anchors source frame f0 at output time t0. Holds (speed 0) are allowed.
    """

    def __init__(self, keys, start, src_fps=30.0, step=1 / 600):
        self.keys = sorted(keys)
        self.t0, self.f0 = start
        self.src_fps = src_fps
        t_end = self.keys[-1][0] + 5
        self.ts = np.arange(min(self.t0, self.keys[0][0]) - 1, t_end, step)
        sp = np.array([self.speed(t) for t in self.ts])
        f = np.concatenate([[0], np.cumsum((sp[1:] + sp[:-1]) / 2 * np.diff(self.ts))]) * src_fps
        f += self.f0 - np.interp(self.t0, self.ts, f)
        self.fs = f

    def speed(self, t):
        ks = self.keys
        if t <= ks[0][0]:
            return ks[0][1]
        for (ta, sa), (tb, sb) in zip(ks, ks[1:]):
            if t <= tb:
                u = (t - ta) / max(tb - ta, 1e-9)
                u = u * u * (3 - 2 * u)
                return sa + (sb - sa) * u
        return ks[-1][1]

    def __call__(self, t):
        return float(np.interp(t, self.ts, self.fs))

    def time_of(self, f):
        """Output time at which source frame f is shown (first crossing)."""
        return float(np.interp(f, self.fs, self.ts))
