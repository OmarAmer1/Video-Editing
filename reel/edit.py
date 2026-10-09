"""The edit: "Thirty, in One Breath" — 29 -> 30 birthday reel (1080x1920, 30 fps, 340 frames, loops).

Story: the 29 take plays as an antique sepia memory (an old portrait print) where only the candle flames keep
their gold.
One continuous camera push carries us from the farther 29 take to the closer 30 take (two-layer registration:
her foreground is face-locked, the background is background-locked onto a clean plate, so neither jumps).
Mid-blow, both candles still lit, the "29" morphs into the "30" (RIFE flow morph), and the 30 candle is the
first thing in colour. She blows it out on the melody's high note: colour floods the night from the wick,
the old film gate bursts open, and she laughs in slow motion (no end card unless titles are built with one).

All creative timings live in CFG (output frame numbers at 30 fps unless noted).
"""
import functools

import cv2
import numpy as np

from . import fx
from .frames import Clip, lerp_sim, similarity, stabilizer
from .geometry import Geometry, apply_luma_lut, fg_prewarp, flame_alive, flame_tips, luma_lut, pull_back
from .timeline import SpeedRamp, camera_matrix

SRC_W, SRC_H = 464, 832

CFG = dict(
    fps=30, n_frames=340, out_size=(1080, 1920),
    # time remaps: playback speed keyed over output seconds (smoothstep between keys)
    c29_speed=[(0.0, 0.5548), (1.0, 0.5548), (2.4, 0.30), (3.2, 0.067)], c29_anchor=(0.0, 0.0),
    c30_speed=[(3.6667, 0.067), (4.6667, 0.2663), (5.0, 0.30), (5.3333, 0.30), (5.8667, 0.82), (6.6667, 0.45),
               (7.6667, 0.28), (9.1667, 0.18), (10.0, 0.09), (10.958, 0.0)], c30_anchor=(3.6667, 61.0),
    morph=(96, 110), pair=(39, 61),              # morph frames; exact source frames C39 <-> B61
    reg=(0.40, 3.20), prewarp=(1.90, 3.20),      # registration and body pre-warp windows (seconds)
    cam0=(1.08, 232, 404, 0.0), cam1=(1.12, 230, 430, -1.6),
    camB=[(3.20, 1.12, 230, 430, -1.6), (3.6667, 1.12, 230, 430, -1.6), (4.6667, 1.15, 230, 428, -1.6),
          (5.3333, 1.11, 229, 428, -1.6), (6.6667, 1.12, 228, 426, -1.6), (10.6, 1.155, 225, 418, -1.6),
          (11.3333, 1.165, 225, 417, -1.6)],
    settle=(0.40, 1.16),                          # opening settle: duration (s), start scale
    candle_ellipse=((228, 430), (45, 50), 20),    # B-space candle region for the fast digit switch
    glow_peak=103, flare=(132, 136), drop=140,
    bloom1=(110, 120, 110), bloom2=(140, 159, 150, 1250), gate=(140, 152),
    luma_lut=(72, 96),
    defocus=[(0, 10.0), (3, 5.0), (8, 0.0), (72, 0.0), (96, 2.0), (140, 2.0), (148, 0.0), (200, 0.0), (260, 4.0), (340, 4.0)],
    embers=(141, 4), leak_in=(0, 4), leak_out=(330, 339),
)


def _piecewise(keys, x):
    xs, ys = zip(*keys)
    return float(np.interp(x, xs, ys))


class Edit:
    def __init__(self, an, cfg=CFG, titles=None):
        self.an = an
        self.cfg = cfg
        self.fps = cfg["fps"]
        self.n = cfg["n_frames"]
        self.duration = self.n / self.fps
        self.out_size = cfg["out_size"]
        self.W, self.H = self.out_size
        self.s = self.W / 1080.0                       # resolution scale for px-specified effects
        self.geo = Geometry(an)
        self.r29 = SpeedRamp(cfg["c29_speed"], cfg["c29_anchor"])
        self.r30 = SpeedRamp(cfg["c30_speed"], cfg["c30_anchor"])
        self.titles = titles
        self.grain16 = fx.Grain((self.H, self.W), seed=16)
        self.grain35 = fx.Grain((self.H, self.W), seed=35)
        self.eye40 = an["c29"]["faces"][40]["eyes"]
        self._cache = {}

    # ------------------------------------------------------------------ cameras (source -> output pixels)
    def cam(self, z, cx, cy, roll):
        return camera_matrix(z, cx, cy, roll, SRC_W, SRC_H, self.W, self.H)

    def settle(self, t):
        d, s0 = self.cfg["settle"]
        if t >= d:
            return np.eye(3)
        e = 1 - 2 ** (-10 * max(t, 0) / d)
        sc = 1 + (s0 - 1) * (1 - e)
        return similarity(sc, 0, cx=540 * self.s, cy=540 * self.s)

    def e_reg(self, t):
        r0, r1 = self.cfg["reg"]
        return float(fx.ease_in_out_sine((t - r0) / (r1 - r0)))

    def screen29(self, t, M):
        """C40-stabilised C space -> output, phasing from the C-native camera to cam1 . M (M = face or bg lock)."""
        A = self.cam(*self.cfg["cam0"])
        B = self.cam(*self.cfg["cam1"]) @ M
        e = self.e_reg(t)
        L = lerp_sim(A, B, e, anchor=self.eye40) if e < 1 else B
        return self.settle(t) @ L

    def camB(self, t):
        ks = self.cfg["camB"]
        if t <= ks[0][0]:
            return self.cam(*ks[0][1:])
        for a, b in zip(ks, ks[1:]):
            if t <= b[0]:
                u = float(fx.ease_in_out_sine((t - a[0]) / (b[0] - a[0])))
                return self.cam(*(np.array(a[1:]) * (1 - u) + np.array(b[1:]) * u))
        return self.cam(*ks[-1][1:])

    def camera(self, t):
        """Reference-space camera (the generic coverage checker composes it with each layer's T)."""
        return self.camB(t)

    def layers(self, t):
        """Layer transforms for render.coverage(): C foreground and background layers, then the 30 take."""
        k = int(round(t * self.fps))
        m0, m1 = self.cfg["morph"]
        out = []
        if k <= m0:
            f = self.r29(t) if k < m0 else float(self.cfg["pair"][0])
            tt = min(t, m0 / self.fps)
            for M in (self.geo.A_face, self.geo.A_bg):
                T = np.linalg.inv(self.camB(t)) @ self.screen29(tt, M) @ stabilizer(self.an, "c29", f, 40)
                out.append(dict(clip="c29", f=f, T=T))
        if k >= m0:
            f = self.cfg["pair"][1] if k <= m1 else min(self.r30(t), 122.0)
            out.append(dict(clip="c30", f=f, T=stabilizer(self.an, "c30", f, 61)))
        return out

    # ------------------------------------------------------------------ precomputed swap assets
    def _c39_reference(self):
        """C39 two-layer composite in reference (B61) space, plus its foreground matte."""
        geo, an = self.geo, self.an
        c29 = Clip("c29")
        ci = self.cfg["pair"][0]
        S = stabilizer(an, "c29", ci, 40)
        Mf, Mb = geo.A_face @ S, geo.A_bg @ S
        img, m = geo.color29(c29.exact(ci)), c29.matte(ci)
        sz = (SRC_W, SRC_H)
        fg = cv2.warpAffine(img, Mf[:2], sz, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
        afg = cv2.warpAffine(m, Mf[:2], sz)
        bg = cv2.warpAffine(img, Mb[:2], sz, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
        h = cv2.GaussianBlur(cv2.warpAffine(cv2.dilate(m, np.ones((13, 13), np.uint8)), Mb[:2], sz), (0, 0), 3)[..., None]
        bgc = bg * (1 - h) + geo.plate() * h
        return fg * afg[..., None] + bgc * (1 - afg[..., None]), afg

    @functools.cached_property
    def prewarp_C(self):
        """Body pre-warp field in C40-stabilised C space (cake/plate glide onto the 30 take's)."""
        bi = self.cfg["pair"][1]
        c30 = Clip("c30")
        comp, afg = self._c39_reference()
        D_R = fg_prewarp(self.geo, comp, c30.exact(bi), afg, c30.matte(bi), self.an["c30"]["faces"][bi]["chin"])
        return pull_back(D_R, self.geo.A_face)

    @functools.cached_property
    def lut(self):
        """Luma LUT: C39 composite -> B61 inside the person matte (reference space)."""
        bi = self.cfg["pair"][1]
        comp, afg = self._c39_reference()
        return luma_lut(comp, Clip("c30").exact(bi), afg * Clip("c30").matte(bi))

    def e2(self, t):
        p0, p1 = self.cfg["prewarp"]
        return float(fx.smoothstep((t - p0) / (p1 - p0)))

    # ------------------------------------------------------------------ composition on the canvas
    def _remap_fg(self, R, img, matte, f, t, e2):
        """Warp a C frame onto the canvas through the face-lock screen transform, with the body pre-warp."""
        Ainv = np.linalg.inv(R.canvas @ self.screen29(t, self.geo.A_face))   # canvas -> C40-stab space
        X, Y = np.meshgrid(np.arange(R.cw, dtype=np.float32), np.arange(R.ch, dtype=np.float32))
        zx = Ainv[0, 0] * X + Ainv[0, 1] * Y + Ainv[0, 2]
        zy = Ainv[1, 0] * X + Ainv[1, 1] * Y + Ainv[1, 2]
        if e2 > 0:
            d = cv2.remap(self.prewarp_C, zx.astype(np.float32), zy.astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            zx, zy = zx + e2 * d[..., 0], zy + e2 * d[..., 1]
        Sinv = np.linalg.inv(stabilizer(self.an, "c29", f, 40))
        sx = (Sinv[0, 0] * zx + Sinv[0, 1] * zy + Sinv[0, 2]).astype(np.float32)
        sy = (Sinv[1, 0] * zx + Sinv[1, 1] * zy + Sinv[1, 2]).astype(np.float32)
        fg = cv2.remap(img, sx, sy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
        fm = cv2.remap(matte, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return fg, fm

    def _c_composite(self, R, t, f):
        """Two-layer composite of C frame f at time t on the canvas -> (rgb, matte, flame tips in output px)."""
        c29 = R.clips["c29"]
        img = self.geo.color29(c29.frame(f))
        m = c29.matte_at(f)
        fg, fm = self._remap_fg(R, img, m, f, t, self.e2(t))
        l0, l1 = self.cfg["luma_lut"]
        a_lut = float(fx.smoothstep((t * self.fps - l0) / (l1 - l0)))
        if a_lut > 0:
            fg = apply_luma_lut(fg, self.lut, a_lut)
        Sb = R.canvas @ self.screen29(t, self.geo.A_bg)
        Mb = Sb @ stabilizer(self.an, "c29", f, 40)
        size = (R.cw, R.ch)
        bg = cv2.warpAffine(img, Mb[:2], size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
        h = cv2.warpAffine(cv2.dilate(m, np.ones((13, 13), np.uint8)), Mb[:2], size, flags=cv2.INTER_LINEAR)
        h = cv2.GaussianBlur(h, (0, 0), 3 * R.cw / SRC_W)[..., None]
        P = Sb @ np.linalg.inv(self.geo.A_bg)                       # reference-space plate -> canvas
        plate = cv2.warpAffine(self.geo.plate(), P[:2], size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
        bgc = bg * (1 - h) + plate * h
        a = cv2.GaussianBlur(np.clip(fm, 0, 1), (0, 0), 0.6)[..., None]
        out = fg * a + bgc * (1 - a)
        Sf = self.screen29(t, self.geo.A_face) @ stabilizer(self.an, "c29", f, 40)
        tips = [(Sf @ np.array([p[0], p[1], 1.0]))[:2] for p in flame_tips(self.geo, "c29", f)]
        return out, fm, tips

    def _b_frame(self, R, t, f):
        c30 = R.clips["c30"]
        Mo = self.camB(t) @ stabilizer(self.an, "c30", f, 61)
        M = R.canvas @ Mo
        size = (R.cw, R.ch)
        slow = t * self.fps > self.cfg["morph"][1] and self.r30.speed(t) < 0.2
        src = c30.frame(f, ensemble=slow)
        if 66.2 < f < 72.0:
            src = self._unflare(src, f)
        img = cv2.warpAffine(src, M[:2], size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
        m = cv2.warpAffine(c30.matte_at(f), M[:2], size, flags=cv2.INTER_LINEAR)
        tips = [(Mo @ np.array([p[0], p[1], 1.0]))[:2] for p in flame_tips(self.geo, "c30", f)]
        cc = self.geo.candle("c30", f)
        self._centre = (Mo @ np.array([cc[0], cc[1] - 4.0, 1.0]))[:2]
        return img, m, tips

    def _unflare(self, src, f, ref=73):
        """After the blow-out the real flame flickers back for a few frames (B68-B69); the story says it is out,
        so the wick region is replaced by the same region from B73 (flame fully out), aligned on the candle track."""
        c30 = Clip("c30")
        d = self.geo.candle("c30", f) - self.geo.candle("c30", ref)
        refimg = cv2.warpAffine(c30.exact(ref), np.float32([[1, 0, d[0]], [0, 1, d[1]]]), (SRC_W, SRC_H),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        tip = flame_tips(self.geo, "c30", f)[-1]
        Y, X = np.mgrid[0:SRC_H, 0:SRC_W].astype(np.float32)
        dist = np.sqrt((X - tip[0]) ** 2 + ((Y - (tip[1] + 3)) * 0.8) ** 2)
        m = np.clip((15 - dist) / 6 + 0.5, 0, 1)
        ramp = float(np.interp(f, [66.2, 66.8, 71.2, 72.0], [0, 1, 1, 0]))
        # replace only the hot (flame) pixels, so faint smoke around the wick survives
        excess = fx.luma(src) - fx.luma(refimg)
        hot = cv2.GaussianBlur(np.clip((excess - 0.04) / 0.12, 0, 1), (0, 0), 2.0)
        m = (m * np.maximum(hot, 0.15 * m) * ramp)[..., None]
        return src * (1 - m) + refimg * m

    def morph_weight_map(self, R, u):
        """Content weight: smootherstep(u) everywhere, but the candle ellipse switches fast around u=0.5."""
        (cx, cy), (ax, ay), feather = self.cfg["candle_ellipse"]
        M = R.canvas @ self.camB(self.cfg["morph"][0] / self.fps)
        sc = np.hypot(M[0, 0], M[1, 0])
        p = M @ np.array([cx, cy, 1.0])
        Y, X = np.mgrid[0:R.ch, 0:R.cw].astype(np.float32)
        d = np.sqrt(((X - p[0]) / (ax * sc)) ** 2 + ((Y - p[1]) / (ay * sc)) ** 2)
        Rm = np.clip((1.0 - d) * ax / feather + 0.5, 0, 1)
        Rm = cv2.GaussianBlur(Rm, (0, 0), 6 * sc)
        wg = float(fx.smootherstep(u))
        wc = float(fx.smootherstep((u - 0.33) / 0.34))          # digits switch over ~f100.6-f105.4, under the flash
        return (wg * (1 - Rm) + wc * Rm).astype(np.float32)

    def compose(self, t, R):
        k = int(round(t * self.fps))
        m0, m1 = self.cfg["morph"]
        ci, bi = self.cfg["pair"]
        if k < m0:
            f = self.r29(t)
            if k <= 8:   # motion blur through the opening settle (many taps while the zoom is fastest)
                taps = np.linspace(-0.25, 0.25, 9) if k <= 4 else np.array([-1 / 3, 0, 1 / 3])
                imgs = [self._c_composite(R, t + d / self.fps, f) for d in taps]
                out = sum(x[0] for x in imgs) / len(imgs)
                _, fm, tips = imgs[len(imgs) // 2]
            else:
                out, fm, tips = self._c_composite(R, t, f)
            return out, fm, dict(tips=tips, flame=flame_alive("c29", f)), 0.0, [dict(clip="c29", f=f)]
        if k <= m1:
            u = (k - m0) / (m1 - m0)
            if "C39" not in self._cache:
                tc = m0 / self.fps
                self._cache["C39"] = self._c_composite(R, tc, float(ci))
                self._cache["B61"] = self._b_frame(R, tc, float(bi))
            A, mA, tA = self._cache["C39"]
            B, mB, tB = self._cache["B61"]
            if u <= 0:
                return A, mA, dict(tips=tA, flame=1.0), 0.0, [dict(clip="c29", f=float(ci))]
            if u >= 1:
                return B, mB, dict(tips=tB, flame=1.0), 1.0, [dict(clip="c30", f=float(bi))]
            Wm = self.morph_weight_map(R, u)
            img, _, _, (ma, mb) = R.rife.morph(A, B, u, weight=Wm, extra0=mA, extra1=mB)
            flip = lambda x: np.ascontiguousarray(x[:, ::-1])
            img2, _, _, _ = R.rife.morph(flip(A), flip(B), u, weight=flip(Wm), extra0=flip(mA), extra1=flip(mB))
            img = (img + flip(img2)) / 2                                   # flip-TTA ensemble
            w = float(fx.smootherstep(u))
            mm = np.clip(ma * (1 - w) + mb * w, 0, 1)
            return img, mm, dict(tips=list(tA) + list(tB), morph_u=u), w, [dict(clip="c29", f=float(ci)),
                                                                          dict(clip="c30", f=float(bi))]
        f = min(self.r30(t), 122.0)
        img, m, tips = self._b_frame(R, t, f)
        return img, m, dict(tips=tips, flame=flame_alive("c30", f)), 1.0, [dict(clip="c30", f=f)]

    # ------------------------------------------------------------------ look
    def colour_mask(self, k, img, centre, wick):
        """0 = antique memory, 1 = golden present.
        Stage 1 (f110-120): the new '30' candle and the cake-top decoration turn colour as one object (keyed by
        wax brightness / berry red / gold leaf inside a small ellipse; the hijab and hands stay sepia).
        Stage 2 (the drop, f140+): colour radiates from the wick across the whole frame."""
        c = self.cfg
        H, W = img.shape[:2]
        m = np.zeros((H, W), np.float32)
        b0, b1, r1 = c["bloom1"]
        if k >= b0:
            a = float(fx.smoothstep((k - b0) / (b1 - b0)))
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            rr = r1 * self.s
            d = np.sqrt(((xx - centre[0]) / 1.0) ** 2 + ((yy - centre[1]) / 0.85) ** 2)
            ell = np.clip((rr - d) / (0.35 * rr) + 0.5, 0, 1)
            hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)
            sat, val = hsv[..., 1], hsv[..., 2]
            wax = np.clip((val - 0.74) / 0.06, 0, 1) * np.clip((0.45 - sat) / 0.1, 0, 1)
            berry = fx.hue_band(hsv, 340, 14, 6) * np.clip((sat - 0.45) / 0.1, 0, 1)
            gold = fx.hue_band(hsv, 32, 66, 6) * np.clip((sat - 0.35) / 0.1, 0, 1) * np.clip((val - 0.45) / 0.1, 0, 1)
            obj = np.maximum(np.maximum(wax, berry), gold) * ell * (1 - fx.hijab_mask(img, hsv))
            obj = cv2.GaussianBlur(cv2.dilate(obj, np.ones((3, 3), np.uint8)), (0, 0), 1.5 * self.s * 2)
            m = np.maximum(m, np.clip(obj * 1.3, 0, 1) * a)
        d0, d1, ra, rb = c["bloom2"]
        if k >= d0:
            u = min((k - d0) / (d1 - d0), 1.0)
            Rr = (ra + (rb - ra) * float(fx.ease_in_out_sine(u) * 0.35 + fx.ease_out_cubic(u) * 0.65)) * self.s
            m = np.maximum(m, fx.radial_mask(img.shape, wick, Rr, feather=0.45))
        return np.clip(m, 0, 1), None

    def _candle_flash(self, out, e):
        s = self.s
        H, W = out.shape[:2]
        t = self.cfg["morph"][0] / self.fps
        cc = self.geo.candle("c30", float(self.cfg["pair"][1]))
        p = self.camB(t) @ stabilizer(self.an, "c30", float(self.cfg["pair"][1]), 61) @ np.array([cc[0], cc[1], 1.0])
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - p[0]) / (125 * s)) ** 2 + ((yy - p[1]) / (110 * s)) ** 2)
        ell = cv2.GaussianBlur(np.clip((1 - d) / 0.45 + 0.5, 0, 1), (0, 0), 12 * s)[..., None]
        soft = cv2.GaussianBlur(out, (0, 0), max(4 * s * e, 0.1))                       # A: local defocus
        a = min(1.5 * e, 1.0)
        out = out * (1 - ell * a) + soft * ell * a
        warm = fx._hex("#FFE9C8")
        hl = cv2.GaussianBlur(np.clip(fx.luma(out) - 0.55, 0, 1), (0, 0), 20 * s)[..., None]  # B: highlight bloom
        out = fx.add_light(out, hl * warm * 2.0 * e)
        out = fx.add_light(out, ell * warm * 0.6 * e)                                       # C: flat warm lift
        return out

    def wick_at_drop(self):
        """Output-pixel position of the 30 flame tip on the drop frame (deterministic, for chunked renders)."""
        if "wick_drop" not in self._cache:
            t = self.cfg["drop"] / self.fps
            f = min(self.r30(t), 122.0)
            Mo = self.camB(t) @ stabilizer(self.an, "c30", f, 61)
            p = flame_tips(self.geo, "c30", f)[-1]
            self._cache["wick_drop"] = (Mo @ np.array([p[0], p[1], 1.0]))[:2] * 1.0
        return self._cache["wick_drop"]

    def _flames(self, img, tips, px):
        m = np.zeros(img.shape[:2], np.float32)
        for p in tips:
            m = np.maximum(m, fx.flame_mask(img, (p[0], p[1] + 14 * px), px=px))
        return m

    def look(self, img, ctx):
        c = self.cfg
        k, t = ctx["k"], ctx["t"]
        H, W = img.shape[:2]
        s = self.s
        info = ctx["candle"]          # dict built by compose()
        tips = info["tips"]
        px = W / SRC_W * 1.12         # output px per source px (approx.)

        # flames: re-lit as warm emitters around the tracked tips
        if "morph_u" in info:
            u = info["morph_u"]
            wB = float(fx.smootherstep((u - 0.3) / 0.4))
            flame = np.maximum(self._flames(img, tips[:2], px) * (1 - wB), self._flames(img, tips[2:], px) * wB)
        else:
            flame = self._flames(img, tips, px) * info.get("flame", 1.0)
        f0, f1 = c["flare"]
        surge = 1 + 0.6 * fx.ramp(k, f0 - 2, f0) * (1 - fx.ramp(k, f1, f1 + 3))

        # colour masks are centred on the candles / the 30 wick (output px)
        wick = np.array(tips[-1]) if tips else np.array([W / 2, H / 2])
        centre = getattr(self, "_centre", None)
        if centre is None or k < c["morph"][1]:
            centre = np.mean(np.array(tips), axis=0) + np.array([0, 25 * s]) if tips else wick
        if k >= c["drop"]:
            wick = self.wick_at_drop()      # the flame is gone: the bloom/embers originate where it died
        cmask, ring = self.colour_mask(k, img, centre, wick)

        # defocus: lens defocus on the opening settle; matte-normalised background blur later
        dfc = _piecewise(c["defocus"], k) * s
        img = fx.unsharp(img, 1.2 * s * 2, 0.25)
        base = img
        if k <= 8:
            if dfc > 0.3:
                base = cv2.GaussianBlur(img, (0, 0), dfc)
        elif dfc > 0.3:
            src = img
            if k >= 200:   # real hotel lights become gold bokeh: boost highlights before the blur
                boost = 0.5 * fx.ramp(k, 200, 260)
                src = img + (np.clip(fx.luma(img) - 0.7, 0, 1) * boost * 1.5)[..., None]
            bl = fx.bg_blur(src, ctx["matte"], dfc)
            m3 = ctx["matte"][..., None]
            base = np.clip(img * m3 + bl * (1 - m3), 0, 1)

        # the two worlds
        cmin, cmax = float(cmask.min()), float(cmask.max())
        if cmin < 1:
            past = fx.mono_antique(base)
            past = fx.flame_glow(past, flame, strength=1.0 * surge, px=px)
        if cmax > 0:
            present = fx.present_grade_v2(base)
            present = fx.flame_glow(present, flame, strength=0.30 * surge, px=px)
        if cmax <= 0:
            out = past
        elif cmin >= 1:
            out = present
        else:
            out = past * (1 - cmask[..., None]) + present * cmask[..., None]
        # warm light-wrap from the flames onto her chin and hands while they burn
        lw = float(flame.max()) if flame.size else 0.0
        if lw > 0.05 and tips:
            gc = np.mean(np.array(tips), axis=0)
            out = fx.add_light(out, fx.radial_glow(img.shape, gc, 220 * s, color=tuple(fx._hex("#FFB060")),
                                                   strength=0.08 * min(lw, 1.0)))
        if ring is not None:
            out = fx.add_light(out, ring)

        # glow swell over the candles, peaking on the 50/50 morph frame
        gp, m0 = c["glow_peak"], c["morph"][0]
        if m0 <= k <= gp + 20:
            g = float(fx.smootherstep((k - m0) / (gp - m0))) if k <= gp else float(np.exp(-((k - gp) / self.fps) / 0.15))
            if g > 0.01:
                gc = np.mean(np.array(tips), axis=0) if tips else wick
                out = fx.add_light(out, fx.radial_glow(img.shape, gc, 260 * s, color=(1.0, 0.788, 0.541), strength=0.30 * g))
                gain = 1 + 0.45 * g * fx.radial_glow(img.shape, gc, 110 * s, color=(1, 1, 1), strength=1.0)
                out = np.clip(out * gain, 0, 1)

        # candle flash: the digits switch inside a burst of candle light sized to the digit pair, so no
        # in-between number (39 / 20) is readable; it lands on the glass-shimmer hit at f103
        if 100 <= k <= 106:
            e = float(np.interp(k, [100, 102, 103, 104, 106], [0, 0.8, 1.0, 0.8, 0]))
            if e > 0:
                out = self._candle_flash(out, e)

        # embers rising from the extinguished wick
        e0, ne = c["embers"]
        if k >= e0:
            if not hasattr(self, "_embers"):
                self._embers = fx.Particles(
                    ne, 3, spawn=lambda r, i: e0 / self.fps + 0.05 + 0.12 * i, life=lambda r: r.uniform(0.8, 1.0),
                    vel=lambda r: (r.uniform(-14, 14) * s, -r.uniform(45, 65) * s),
                    size=lambda r: r.uniform(2.0, 3.5) * s, color=lambda r: (1.0, 0.69, 0.38),
                    opacity=lambda r: r.uniform(0.7, 1.0), sway=5 * s)
                self._ember_origin = (wick[0], wick[1] - 6 * s)
            o = self._ember_origin
            rgb, _ = self._embers.render(img.shape, t, lambda q, r: (o[0] + r.uniform(-6, 6) * s, o[1] + r.uniform(-4, 4) * s))
            halo = cv2.GaussianBlur(rgb, (0, 0), 6 * s) * np.array(fx._hex("#FFB060")) * 2.5
            out = fx.add_light(out, rgb * 1.3 + halo)

        # typography (before grain, so it lives in the film)
        if self.titles is not None:
            nc = getattr(self.titles, "numeral_centre", None)
            bl = 0.0
            if nc is not None:
                x, y = int(np.clip(nc[0] * s, 0, W - 1)), int(np.clip(nc[1] * s, 0, H - 1))
                bl = float(cmask[y, x])
            L = self.titles.layer(k, bloom=bl)
            if L is not None:
                out = out * (1 - L[..., 3:4]) + L[..., :3]

        # loop seam: warm candle-coloured leak from lower-left, decaying on f0-4 and building on f330-339
        li0, li1 = c["leak_in"]
        lo0, lo1 = c["leak_out"]
        lk = 0.8 * max(1 - k / 3.0, 0) ** 2 + 0.8 * max((k - 334) / 5.0, 0) ** 2
        if lk > 0:
            out = fx.add_light(out, fx.light_leak(img.shape, t, seed=4, strength=lk, color=(1.0, 0.70, 0.42),
                                                  pos=(0.35, 0.6), scale=1.6))
        flash = 0.35 * max(1 - k / 2.0, 0) ** 2 + 0.35 * max((k - 336) / 3.0, 0) ** 2
        if flash > 0:
            out = fx.add_light(out, np.ones_like(out[..., :1]) * np.array(fx._hex("#FFC48A")) * flash)

        # film: smooth flicker, vignette, grain 16mm -> 35mm through the colour mask, then the gate
        cm3 = cmask[..., None]
        out = out * ((1 - cm3) * fx.flicker_smooth(k, 0.012) + cm3)
        out = fx.vignette(out, 0.45) * (1 - cm3) + fx.vignette(out, 0.25) * cm3
        g16 = self.grain16.apply(out, k, amount=0.035, size=1.7 * s, chroma=0.0)
        g35 = self.grain35.apply(out, k, amount=0.018, size=1.2 * s, chroma=0.2)
        out = g16 * (1 - cm3) + g35 * cm3
        g0, g1 = c["gate"]
        go = float(1 - 2 ** (-10 * np.clip((k - g0) / (g1 - g0), 0, 1))) if k >= g0 else 0.0
        if go < 0.999:
            if k < g0:
                dx, dy = fx.gate_weave(k, amp=0.8 * s)
                out = cv2.warpAffine(out, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REFLECT101)
            gi = float(fx.ease_out_cubic(k / 5.0)) if k < 5 else 1.0      # gate closes in at the loop restart
            go = 1 - (1 - go) * gi
            out = fx.film_gate_rect(out, inset=26 * s * (1 - go) - 90 * s * go, radius=56 * s * (1 - go), feather=4 * s)
        # +-1 LSB triangular dither
        r = np.random.default_rng(1000 + k)
        out = out + (r.random(out.shape, dtype=np.float32) - r.random(out.shape, dtype=np.float32)) / 255.0
        return np.clip(out, 0, 1)
