"""Reel typography: kicker 'one last wish at', the '29' -> '30' odometer numeral, and the end card.

Implements final_edl.json["typography"] (plus the TXT notes of the timeline):

  kicker   'one last wish at'  Instrument Serif Italic 50 px, #EDE6DA @ 92 %, x 74, baseline 280.
           Word-by-word in at f6/f10/f14/f18 (12 fr each: opacity 0->0.92, blur 10->0, rise 12->0, easeOutCubic);
           out f78-f90 (blur 0->8, opacity ->0, drift up 6 px, easeInCubic).
  numeral  '29' -> '30'        Instrument Serif Regular 200 px, #F2EDE4 @ 92 %, x 70, baseline 452.
           Rack focus in f20-f38 (blur 18->0, scale 1.06->1.00 about the glyph centre, opacity 0->0.92, easeOutCubic).
           Odometer on fixed digit cells from the '30' layout: units '9'->'0' f106-f113, tens '2'->'3' f107-f114 (rolling together, so '20' never reads),
           easeInOutBack (overshoot 1.2), 12 px vertical motion blur at peak speed, clipped to the digit cell
           (figure height + 24 px). Recolour ivory -> candle gold #E8C78E gated by the colour-bloom value at the
           glyph centre and rate-limited to the f142-f150 ramp (the bloom front alone crosses it in ~1 frame), with an
           inner glow #FFB866 (10 px) pulsing 25 % -> 45 % -> 25 %. Exit f186-f198 (blur 0->12, opacity ->0,
           scale 1.0->0.97).
  end card 'happy birthday,'    Italic 56 px, #F4EBDD @ 95 %, x 74, baseline 284; 'happy' f176-f190, 'birthday,' f184-f198.
           'my love.'           Italic 92 px, gold #E8C78E + inner glow #FFB866 (10 px, 25 %), x 72, baseline 420;
                                'my' f200-f214, 'love.' f207-f221; glow pulse 25 % -> 45 % -> 25 % centred on f260.
  Everything fades out f328-f339.

Rendering: glyphs are rasterised by FreeType at 2x, every per-frame transform (rise, scale, odometer roll, motion
blur, cell clip) is applied on the 2x mask, then area-downsampled; focus blur, the soft shadow (0, 2 px, 12 px blur,
black 45 %) and the inner glow are applied at output resolution. Only the text's bounding region is processed.
All shadows are composited under all glyphs (one text layer, one drop shadow). Resting positions sit on the 2x pixel
grid so settled text is an exact copy of the FreeType raster.
All geometry is authored for 1080x1920 and scaled by out_size[0] / 1080.

Conventions: blur values are Gaussian sigmas in px (as in reel/text.py, whose default shadow is (0, 2, 12, 0.45)).
"""
import functools
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import paths

FONT_WORDS = "InstrumentSerif-Italic.ttf"
FONT_NUMERALS = "InstrumentSerif-Regular.ttf"
SS = 2                                  # supersampling factor


def _hex(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


KICKER_COL = _hex("#EDE6DA")
IVORY = _hex("#F2EDE4")
END_COL = _hex("#F4EBDD")
GOLD = _hex("#E8C78E")
GLOW_COL = _hex("#FFB866")

# Every number below is from the approved spec (1080x1920 px, output frames at 30 fps).
SPEC = dict(
    shadow=dict(dx=0.0, dy=2.0, blur=12.0, alpha=0.45),
    inner_glow_px=10.0,
    kicker=dict(words=("one", "last", "wish", "at"), size=50, x=74, baseline=280, colour=KICKER_COL, opacity=0.92,
                starts=(6, 10, 14, 18), dur=12, blur_in=10.0, rise=12.0, out=(78, 90), blur_out=8.0, drift=6.0),
    numeral=dict(size=200, x=70, baseline=452, opacity=0.92, rack=(20, 38), rack_blur=18.0, rack_scale=1.06,
                 units=(106, 113), tens=(107, 114), overshoot=1.2, motion_blur=12.0, cell_margin=12.0,
                 cell_feather=4.0, recolour_fallback=(142, 150), pulse_centre=154, pulse_half=14,
                 exit=(186, 198), exit_blur=12.0, exit_scale=0.97),
    end1=dict(words=("happy", "birthday,"), size=56, x=74, baseline=284, colour=END_COL, opacity=0.95,
              windows=((176, 190), (184, 198)), blur_in=10.0, rise=12.0),
    end2=dict(words=("my", "love."), size=92, x=72, baseline=420, colour=GOLD, opacity=0.95,
              windows=((200, 214), (207, 221)), blur_in=10.0, rise=12.0, max_right=345.0,
              pulse_centre=260, pulse_half=30),
    glow=(0.25, 0.45),                   # inner-glow strength: rest, pulse peak
    fade=(328, 339),
    safe_top=220,
)


# ----------------------------------------------------------------------------- easing
def _c01(u):
    return float(min(max(u, 0.0), 1.0))


def prog(k, f0, f1):
    return _c01((k - f0) / float(f1 - f0))


def ease_out_cubic(u):
    return 1.0 - (1.0 - u) ** 3


def ease_in_cubic(u):
    return u ** 3


def ease_in_out_cubic(u):
    return 4 * u ** 3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2


def smoothstep(u):
    u = _c01(u)
    return u * u * (3 - 2 * u)


def ease_in_out_back(u, overshoot=1.2):
    c2 = overshoot * 1.525
    if u < 0.5:
        return (2 * u) ** 2 * ((c2 + 1) * 2 * u - c2) / 2
    return ((2 * u - 2) ** 2 * ((c2 + 1) * (u * 2 - 2) + c2) + 2) / 2


def _d_ease_in_out_back(u, overshoot=1.2, h=1e-4):
    return (ease_in_out_back(min(u + h, 1.0), overshoot) - ease_in_out_back(max(u - h, 0.0), overshoot)) / (
        min(u + h, 1.0) - max(u - h, 0.0))


def bump(k, centre, half):
    """Raised cosine: 1 at centre, 0 at |k - centre| >= half."""
    x = _c01(abs(k - centre) / float(half))
    return 0.5 * (1 + math.cos(math.pi * x))


# ----------------------------------------------------------------------------- glyphs
@functools.lru_cache(maxsize=None)
def _font(file, size):
    return ImageFont.truetype(str(paths.FONTS / file), size)   # float sizes are fine in Pillow >= 10.1


@functools.lru_cache(maxsize=512)
def _glyph(text, file, size, pad=4):
    """FreeType raster of `text` -> (mask float32 0..1, (ox, oy)): mask top-left relative to pen (left, baseline)."""
    f = _font(file, size)
    l, t, r, b = f.getbbox(text, anchor="ls")
    w = int(math.ceil(r - l)) + 2 * pad
    h = int(math.ceil(b - t)) + 2 * pad
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=f, fill=255, anchor="ls")
    return np.asarray(img, np.float32) / 255.0, (float(l - pad), float(t - pad))


def _ink(M, thresh=0.02):
    ys, xs = np.nonzero(M > thresh)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _shift(M, tx, ty, size):
    """Bilinear translate a mask into a canvas of `size` (w, h)."""
    return cv2.warpAffine(M, np.float32([[1, 0, tx], [0, 1, ty]]), size, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _union(a, b):
    return a + b - a * b


def _box_kernel(L):
    """Normalised 1D box of (fractional) length L px, centred."""
    half = max(L, 1.0) / 2
    n = int(math.ceil(half - 0.5))
    i = np.arange(-n, n + 1, dtype=np.float32)
    w = np.clip(np.minimum(i + 0.5, half) - np.maximum(i - 0.5, -half), 0, None)
    return (w / w.sum()).astype(np.float32)


class _Sprite:
    """A 2x-supersampled coverage mask placed at absolute 2x output coords `tl2` (top-left) at rest."""
    __slots__ = ("M", "tl2", "ink2")

    def __init__(self, M, tl2):
        self.M = M
        self.tl2 = (float(tl2[0]), float(tl2[1]))
        self.ink2 = _ink(M)

    def ink_box(self, a=1.0, tx=0.0, ty=0.0):
        """Ink bbox in output px after X -> a*X + t (output-px affine)."""
        if self.ink2 is None:
            return None
        x0, y0, x1, y1 = self.ink2
        X0 = a * (x0 + self.tl2[0]) / SS + tx
        X1 = a * (x1 + self.tl2[0]) / SS + tx
        Y0 = a * (y0 + self.tl2[1]) / SS + ty
        Y1 = a * (y1 + self.tl2[1]) / SS + ty
        return X0, Y0, X1, Y1

    def centre(self):
        b = self.ink_box()
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


# ----------------------------------------------------------------------------- titles
class Titles:
    """All on-screen text of the reel as a premultiplied RGBA layer per output frame."""

    def __init__(self, out_size=(1080, 1920), name=None):
        self.W, self.H = int(out_size[0]), int(out_size[1])
        self.s = self.W / 1080.0
        self.name = name
        self.nudge = {}                       # element -> (dx, dy, scale) from fit_to_matte(); 1080-space px
        s2 = self.s * SS
        sp = SPEC

        # kicker words and end-card line 1: word sprites at their pen positions
        self._kicker = self._word_sprites(sp["kicker"]["words"], sp["kicker"]["size"], sp["kicker"]["x"],
                                          sp["kicker"]["baseline"])
        self._end1 = self._word_sprites(sp["end1"]["words"], sp["end1"]["size"], sp["end1"]["x"],
                                        sp["end1"]["baseline"])

        # end-card line 2: 'my love.' or her first name (fitted to end before x 345)
        e2 = sp["end2"]
        if name:
            nm = name.strip()
            if nm and nm[-1] not in ".!?,":
                nm += "."
            words2 = tuple(nm.split())
        else:
            words2 = e2["words"]
        size2 = e2["size"]
        f = _font(FONT_WORDS, size2 * s2)
        avail = (e2["max_right"] - e2["x"]) * s2
        wid = f.getbbox(" ".join(words2), anchor="ls")[2]
        if wid > avail:
            size2 = size2 * avail / wid
        self.end2_words = words2
        self.end2_size = size2
        self._end2 = self._word_sprites(words2, size2, e2["x"], e2["baseline"])
        n2 = len(words2)
        if n2 == len(e2["windows"]):
            self._end2_windows = e2["windows"]
        else:                                  # spread any word count over the 'my'..'love.' window
            a0, a1 = e2["windows"][0][0], e2["windows"][-1][0]
            d = e2["windows"][0][1] - e2["windows"][0][0]
            self._end2_windows = tuple((a0 + (a1 - a0) * (i / max(n2 - 1, 1)), a0 + (a1 - a0) * (i / max(n2 - 1, 1)) + d)
                                       for i in range(n2))

        self._build_numeral()

    # ---------------------------------------------------------------- construction
    def _word_sprites(self, words, size, x, baseline):
        s2 = self.s * SS
        f = _font(FONT_WORDS, size * s2)
        out = []
        for i, w in enumerate(words):
            pen = x * s2 + (f.getlength(" ".join(words[:i]) + " ") if i else 0.0)
            M, (ox, oy) = _glyph(w, FONT_WORDS, size * s2)
            # snap the resting position to the 2x pixel grid (<= 0.25 output px): at rest the warp is then an exact
            # integer copy, so every word is equally crisp (a fractional rest offset bilinear-softens a word)
            out.append(_Sprite(M, (round(pen + ox), round(baseline * s2 + oy))))
        return out

    def _build_numeral(self):
        n = SPEC["numeral"]
        s2 = self.s * SS
        size = n["size"] * s2
        f = _font(FONT_NUMERALS, size)
        x0, base = n["x"] * s2, n["baseline"] * s2
        # fixed digit cells from the '30' layout
        pen = {"3": x0, "0": x0 + f.getlength("30") - f.getlength("0")}
        g = {d: _glyph(d, FONT_NUMERALS, size) for d in "2930"}
        ink = {d: _ink(g[d][0]) for d in g}

        def ink_cx(d, penx):
            M, (ox, oy) = g[d]
            return penx + ox + (ink[d][0] + ink[d][2]) / 2

        cell_cx = [ink_cx("3", pen["3"]), ink_cx("0", pen["0"])]
        # figure extent (all four digits) relative to the baseline, in 2x px
        top = min(g[d][1][1] + ink[d][1] for d in g)
        bot = max(g[d][1][1] + ink[d][3] for d in g)
        self.fig_height = (bot - top) / s2                         # in 1080-space px
        m2 = n["cell_margin"] * s2
        cell_top, cell_bot = base + top - m2, base + bot + m2
        self.pitch2 = cell_bot - cell_top                          # drum pitch = digit-cell height

        # numeral canvas (2x, absolute): generous margin so nothing is cut before the cell clip
        lft = min(pen["3"] + g[d][1][0] for d in g) - 8 * s2
        rgt = max(pen["0"] + g[d][1][0] + g[d][0].shape[1] for d in g) + 8 * s2
        cx0, cy0 = math.floor(lft), math.floor(cell_top - 4 * s2)
        cw, ch = int(math.ceil(rgt - cx0)), int(math.ceil(cell_bot + 4 * s2 - cy0))
        self._ncan = (cx0, cy0, cw, ch)

        # each digit pre-placed (horizontally) in its cell, on the baseline, as a canvas-sized mask
        placed = {}
        for d, cell in (("2", 0), ("3", 0), ("9", 1), ("0", 1)):
            M, (ox, oy) = g[d]
            px = cell_cx[cell] - (ink[d][0] + ink[d][2]) / 2 - ox     # pen x that centres the ink in the cell
            # integer 2x offsets: a resting digit is copied exactly (a half-pixel offset softened the '2' and '0')
            placed[d] = (M, float(round(px + ox - cx0)), float(round(base + oy - cy0)))
        self._placed = placed

        # vertical clip of the digit cell, feathered inside its 12 px margin
        y = np.arange(ch, dtype=np.float32) + 0.5 + cy0
        fe = n["cell_feather"] * s2
        clip = np.clip((y - cell_top) / fe, 0, 1) * np.clip((cell_bot - y) / fe, 0, 1)
        self._clip = clip[:, None].astype(np.float32)
        self.cell_box = (cx0 / s2, cell_top / s2, (cx0 + cw) / s2, cell_bot / s2)   # 1080-space

        # peak |dE/du| of the odometer ease, to scale motion blur to 12 px at peak speed
        us = np.linspace(0, 1, 2001)
        self._vmax = max(abs(_d_ease_in_out_back(u, n["overshoot"])) for u in us)

        # frames on which each odometer digit visibly 'clicks' over (first frame with eased progress >= 0.5), for the
        # sound design: units f110, tens f114 (the spec's ticks are f108 / f114; f108 is 2 frames before the
        # units digit moves: the back-ease is still in its dip at f107-f108)
        def click(w):
            return next(k for k in range(w[0], w[1] + 1) if ease_in_out_back(prog(k, *w), n["overshoot"]) >= 0.5)
        self.odometer_clicks = {"units": click(n["units"]), "tens": click(n["tens"])}

        self._num_cache = {}
        n29 = self._numeral_sprite(0.0, 0.0, 0.0, 0.0)
        n30 = self._numeral_sprite(1.0, 1.0, 0.0, 0.0)
        c29, c30 = n29.centre(), n30.centre()
        # numeral centre in 1080-space px (edit.py samples the colour-bloom mask there, times out_w / 1080)
        self.numeral_centre = (c30[0] / self.s, c30[1] / self.s)
        self._c29, self._c30 = c29, c30

    def _cell(self, d_out, d_in, p, blur2):
        """One odometer cell: d_out rolls up and away, d_in rolls in from below (p = eased progress)."""
        cx0, cy0, cw, ch = self._ncan
        acc = None
        for d, off in ((d_out, -self.pitch2 * p), (d_in, self.pitch2 * (1.0 - p))):
            if abs(off) >= self.pitch2 + blur2:                       # entirely outside the cell
                continue
            M, tx, ty = self._placed[d]
            m = _shift(M, tx, ty + off, (cw, ch))
            acc = m if acc is None else _union(acc, m)
        if acc is None:
            return np.zeros((ch, cw), np.float32)
        if blur2 > 1.0:
            acc = cv2.filter2D(acc, -1, _box_kernel(blur2).reshape(-1, 1), borderType=cv2.BORDER_CONSTANT)
        if (p != 0.0 and p != 1.0) or blur2 > 0:
            # any displacement (incl. the back-ease dip p < 0 and the overshoot p > 1) is seen through the cell window
            acc = acc * self._clip
        return acc

    def _numeral_sprite(self, pu, pt, bu, bt):
        """Numeral mask for units/tens progress pu/pt (eased; may over/undershoot) and motion-blur lengths (2x px)."""
        key = (round(pu, 5), round(pt, 5), round(bu, 3), round(bt, 3))
        spr = self._num_cache.get(key)
        if spr is None:
            tens = self._cell("2", "3", pt, bt)
            units = self._cell("9", "0", pu, bu)
            cx0, cy0, _, _ = self._ncan
            spr = _Sprite(_union(tens, units), (cx0, cy0))
            if len(self._num_cache) > 64:
                self._num_cache.clear()
            self._num_cache[key] = spr
        return spr

    # ---------------------------------------------------------------- per-frame plan
    def _plan(self, k, bloom=None):
        """List of draw items for frame k: dict(name, spr, a, tx, ty, blur, op, col, glow) in output px."""
        sp, s = SPEC, self.s
        fade = 1.0 - smoothstep(prog(k, *sp["fade"]))
        items = []
        if fade <= 0:
            return items

        def add(name, spr, op, blur=0.0, dy=0.0, sc=1.0, centre=None, col=None, glow=0.0):
            if op < 1.0 / 512:
                return
            if centre is None:
                centre = (0.0, 0.0)
            a = sc
            tx = centre[0] - sc * centre[0]
            ty = centre[1] - sc * centre[1] + dy * s
            nd = self.nudge.get(name)
            if nd is not None:                  # pre-transform: scale about the line's pen origin, then shift
                ndx, ndy, nsc, (ax, ay) = nd
                ax, ay = ax * s, ay * s
                # X -> a*(nsc*(X - A) + A + nd) + t
                tx, ty = a * (ax - nsc * ax - ndx * s) + tx, a * (ay - nsc * ay - ndy * s) + ty
                a = a * nsc
            items.append(dict(name=name, spr=spr, a=a, tx=tx, ty=ty, blur=blur * s, op=op, col=col, glow=glow))

        # 1) kicker
        kk = sp["kicker"]
        if k < kk["out"][1]:
            q = ease_in_cubic(prog(k, *kk["out"]))
            for spr, t0 in zip(self._kicker, kk["starts"]):
                p = ease_out_cubic(prog(k, t0, t0 + kk["dur"]))
                add("kicker", spr, kk["opacity"] * p * (1 - q) * fade,
                    blur=kk["blur_in"] * (1 - p) + kk["blur_out"] * q,
                    dy=kk["rise"] * (1 - p) - kk["drift"] * q, col=kk["colour"])

        # 2) numeral
        n = sp["numeral"]
        if n["rack"][0] <= k < n["exit"][1]:
            p_in = ease_out_cubic(prog(k, *n["rack"]))
            q = ease_in_out_cubic(prog(k, *n["exit"]))
            uu, ut = prog(k, *n["units"]), prog(k, *n["tens"])
            pu, pt = ease_in_out_back(uu, n["overshoot"]), ease_in_out_back(ut, n["overshoot"])
            mb2 = n["motion_blur"] * self.s * SS / self._vmax
            bu = mb2 * abs(_d_ease_in_out_back(uu, n["overshoot"])) if 0 < uu < 1 else 0.0
            bt = mb2 * abs(_d_ease_in_out_back(ut, n["overshoot"])) if 0 < ut < 1 else 0.0
            spr = self._numeral_sprite(pu, pt, bu, bt)
            sc = (n["rack_scale"] + (1 - n["rack_scale"]) * p_in) * (1 + (n["exit_scale"] - 1) * q)
            centre = self._c29 if k < n["units"][0] else self._c30
            m = self._bloom(k, bloom)
            col = IVORY + (GOLD - IVORY) * m
            g0, g1 = sp["glow"]
            glow = m * (g0 + (g1 - g0) * bump(k, n["pulse_centre"], n["pulse_half"]))
            add("numeral", spr, n["opacity"] * p_in * (1 - q) * fade,
                blur=n["rack_blur"] * (1 - p_in) + n["exit_blur"] * q, sc=sc, centre=centre, col=col, glow=glow)

        # 3) end card
        e1 = sp["end1"]
        if k >= e1["windows"][0][0]:
            for spr, (a0, a1) in zip(self._end1, e1["windows"]):
                p = ease_out_cubic(prog(k, a0, a1))
                add("end1", spr, e1["opacity"] * p * fade, blur=e1["blur_in"] * (1 - p), dy=e1["rise"] * (1 - p),
                    col=e1["colour"])
        e2 = sp["end2"]
        if k >= self._end2_windows[0][0]:
            g0, g1 = sp["glow"]
            glow = g0 + (g1 - g0) * bump(k, e2["pulse_centre"], e2["pulse_half"])
            for spr, (a0, a1) in zip(self._end2, self._end2_windows):
                p = ease_out_cubic(prog(k, a0, a1))
                add("end2", spr, e2["opacity"] * p * fade, blur=e2["blur_in"] * (1 - p), dy=e2["rise"] * (1 - p),
                    col=e2["colour"], glow=glow)
        return items

    @staticmethod
    def _bloom(k, bloom):
        """Ivory -> gold amount of the numeral.

        The spec wants the recolour 'as the bloom passes the glyph centre, ~f142-f150'. The bloom value gates it
        (never gold before the colour has reached the glyph) and the f142-f150 ramp limits its rate, so it can never
        pop. With a pure easeOutCubic drop bloom (R 190 -> 2300 px, 45 % feather, wick ~(585, 970)) the front crosses
        the numeral centre in ~1 frame (mask 0.00 at f141, 0.89 at f142), i.e. a one-frame ivory -> gold pop without
        the ramp; with edit.py's 0.35 sine + 0.65 cubic easing it crosses f145-f150 and the bloom itself drives."""
        ramp = smoothstep(prog(k, *SPEC["numeral"]["recolour_fallback"]))
        if bloom is None:                    # no mask value supplied: the bloom front passes about f142-f150
            return ramp
        b = float(bloom)
        if not math.isfinite(b):             # a NaN mask sample must never poison the layer
            b = 0.0
        return min(smoothstep(min(max(b, 0.0), 1.0)), ramp)

    # ---------------------------------------------------------------- rasterisation
    def _draw_parts(self, it):
        """Rasterise one draw item -> (premultiplied fill RGBA crop, black-shadow alpha crop, x0, y0) or None."""
        spr, a, tx, ty = it["spr"], it["a"], it["tx"], it["ty"]
        s = self.s
        sh = SPEC["shadow"]
        M = spr.M
        h, w = M.shape
        # 2x mapping: X2 = a*(u + tl2) + SS*t
        bx = a * spr.tl2[0] + SS * tx
        by = a * spr.tl2[1] + SS * ty
        x0f, x1f = bx / SS, (bx + a * w) / SS
        y0f, y1f = by / SS, (by + a * h) / SS
        sb = sh["blur"] * s
        margin = int(math.ceil(3.0 * it["blur"] + 3.0 * sb + abs(sh["dy"]) * s + 2))
        X0, Y0 = max(int(math.floor(x0f)) - margin, 0), max(int(math.floor(y0f)) - margin, 0)
        X1, Y1 = min(int(math.ceil(x1f)) + margin, self.W), min(int(math.ceil(y1f)) + margin, self.H)
        if X1 <= X0 or Y1 <= Y0:
            return None
        cw, ch = X1 - X0, Y1 - Y0
        Mt = cv2.warpAffine(M, np.float32([[a, 0, bx - SS * X0], [0, a, by - SS * Y0]]), (SS * cw, SS * ch),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        A = cv2.resize(Mt, (cw, ch), interpolation=cv2.INTER_AREA)
        if it["blur"] > 0.05:
            A = cv2.GaussianBlur(A, (0, 0), it["blur"])
        op = it["op"]
        T = A * op
        # soft shadow
        S = cv2.warpAffine(A, np.float32([[1, 0, sh["dx"] * s], [0, 1, sh["dy"] * s]]), (cw, ch),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        S = cv2.GaussianBlur(S, (0, 0), max(sb, 0.1)) * (sh["alpha"] * op)
        # colour + inner glow (screen-blended toward the glow colour inside the glyph edges)
        col = np.asarray(it["col"], np.float32)
        if it["glow"] > 1e-4:
            Gb = cv2.GaussianBlur(A, (0, 0), SPEC["inner_glow_px"] * 0.5 * s)
            e = np.clip(2.0 * (1.0 - Gb), 0.0, 1.0) * it["glow"]
            scr = 1.0 - (1.0 - col) * (1.0 - GLOW_COL)
            rgb = col[None, None, :] + (scr - col)[None, None, :] * e[..., None]
            rgb = rgb * T[..., None]
        else:
            rgb = T[..., None] * col[None, None, :]
        fill = np.empty((ch, cw, 4), np.float32)
        fill[..., :3] = rgb
        fill[..., 3] = T
        return fill, S, X0, Y0

    def _draw(self, it):
        """Rasterise one draw item -> (premultiplied RGBA crop incl. its own shadow, x0, y0) or None."""
        r = self._draw_parts(it)
        if r is None:
            return None
        fill, S, X0, Y0 = r
        fill[..., 3] = fill[..., 3] + S * (1.0 - fill[..., 3])
        return fill, X0, Y0

    # ---------------------------------------------------------------- public API
    def render(self, k, bloom=None):
        """Text for frame k as (premultiplied RGBA crop, x0, y0), or None when no text is on screen.

        Cheaper than layer() for compositing: out[y0:y0+h, x0:x0+w] = out*(1-A) + RGB over the crop only."""
        crops = [c for c in (self._draw_parts(it) for it in self._plan(k, bloom)) if c is not None]
        if not crops:
            return None
        X0 = min(c[2] for c in crops)
        Y0 = min(c[3] for c in crops)
        X1 = max(c[2] + c[0].shape[1] for c in crops)
        Y1 = max(c[3] + c[0].shape[0] for c in crops)
        reg = np.zeros((Y1 - Y0, X1 - X0, 4), np.float32)
        sha = np.zeros((Y1 - Y0, X1 - X0), np.float32)
        # all shadows sit UNDER all glyphs (as one text layer with one drop shadow): a word's 12 px shadow must not
        # darken its neighbour's letters (per-word 'over' darkened glyph edges by up to 5/255)
        for fill, S, x, y in crops:
            sub = sha[y - Y0:y - Y0 + S.shape[0], x - X0:x - X0 + S.shape[1]]
            sub += S * (1.0 - sub)
            sub = reg[y - Y0:y - Y0 + fill.shape[0], x - X0:x - X0 + fill.shape[1]]
            sub *= (1.0 - fill[..., 3:4])
            sub += fill
        reg[..., 3] += sha * (1.0 - reg[..., 3])
        return reg, X0, Y0

    def layer(self, k, bloom=None):
        """HxWx4 float32 premultiplied RGBA layer for output frame k.

        bloom: colour-bloom mask value (0..1) at numeral_centre; drives the ivory -> gold recolour of the numeral.
        None = time-based fallback (the drop bloom crossing the numeral at about f142-f150)."""
        out = np.zeros((self.H, self.W, 4), np.float32)
        r = self.render(k, bloom)
        if r is not None:
            reg, x, y = r
            out[y:y + reg.shape[0], x:x + reg.shape[1]] = reg
        return out

    def named_boxes(self, k, bloom=None, min_opacity=0.02):
        """{element: (x0, y0, x1, y1)} ink boxes (output px, after rise/scale/roll; excluding blur and shadow)."""
        boxes = {}
        for it in self._plan(k, bloom):
            if it["op"] < min_opacity:
                continue
            b = it["spr"].ink_box(it["a"], it["tx"], it["ty"])
            if b is None:
                continue
            o = boxes.get(it["name"])
            boxes[it["name"]] = b if o is None else (min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3]))
        return {nm: (int(math.floor(b[0])), int(math.floor(b[1])), int(math.ceil(b[2])), int(math.ceil(b[3])))
                for nm, b in boxes.items()}

    def boxes(self, k, bloom=None):
        """List of (x0, y0, x1, y1) text bounding boxes on frame k (output px)."""
        return list(self.named_boxes(k, bloom).values())

    # ---------------------------------------------------------------- matte collision (spec: dilate 24, nudge <= 12)
    def collisions(self, k, matte, dilate=24, thresh=0.5):
        """{element: overlap_px} for text boxes that touch the person matte (HxW at output res) dilated by 24 px."""
        Md = self._dilated(matte, dilate, thresh)
        out = {}
        for nm, (x0, y0, x1, y1) in self.named_boxes(k).items():
            sub = Md[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)]
            if sub.any():
                cols = np.nonzero(sub.any(axis=0))[0]
                out[nm] = int(sub.shape[1] - cols.min())          # px of the box right part inside the matte
        return out

    def _dilated(self, matte, dilate, thresh):
        r = max(int(round(dilate * self.s)), 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return cv2.dilate((np.asarray(matte) > thresh).astype(np.uint8), ker) > 0

    def fit_to_matte(self, matte_fn, frames=None, dilate=24, max_nudge=12, shrink=0.92):
        """Choose ONE constant nudge per element (temporally stable): left by up to 12 px, else up, else scale -8 %.

        matte_fn(k) -> HxW person matte at output resolution. Stores and returns self.nudge
        ({element: (dx_left, dy_up, scale, anchor)} in 1080-space px)."""
        anchors = {"kicker": (SPEC["kicker"]["x"], SPEC["kicker"]["baseline"]),
                   "numeral": (SPEC["numeral"]["x"], SPEC["numeral"]["baseline"]),
                   "end1": (SPEC["end1"]["x"], SPEC["end1"]["baseline"]),
                   "end2": (SPEC["end2"]["x"], SPEC["end2"]["baseline"])}
        frames = range(SPEC["fade"][1] + 1) if frames is None else frames
        self.nudge = {}
        per = {}
        dil = {}
        for k in frames:
            nb = self.named_boxes(k)
            if nb:
                dil[k] = self._dilated(matte_fn(k), dilate, 0.5)
            for nm, b in nb.items():
                per.setdefault(nm, []).append((k, b))

        def overlap(nm, f):
            n = 0
            for k, b in per[nm]:
                x0, y0, x1, y1 = f(b)
                n += int(dil[k][max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)].sum())
            return n

        def shrunk(nm, b, dd=0):
            ax, ay = anchors[nm][0] * self.s, anchors[nm][1] * self.s
            return (int(math.floor(ax + (b[0] - ax) * shrink)) - dd, int(math.floor(ay + (b[1] - ay) * shrink)),
                    int(math.ceil(ax + (b[2] - ax) * shrink)) - dd, int(math.ceil(ay + (b[3] - ay) * shrink)))

        self.unresolved = {}
        for nm in per:
            if overlap(nm, lambda b: b) == 0:
                continue
            cands = []
            for d in range(1, max_nudge + 1):                       # 1) nudge left
                dd = int(round(d * self.s))
                cands.append(((float(d), 0.0, 1.0), lambda b, dd=dd: (b[0] - dd, b[1], b[2] - dd, b[3])))
            for d in range(1, max_nudge + 1):                       # 2) nudge up
                dd = int(round(d * self.s))
                cands.append(((0.0, float(d), 1.0), lambda b, dd=dd: (b[0], b[1] - dd, b[2], b[3] - dd)))
            cands.append(((0.0, 0.0, shrink), lambda b, nm=nm: shrunk(nm, b)))          # 3) scale -8 %
            dd = int(round(max_nudge * self.s))
            cands.append(((float(max_nudge), 0.0, shrink), lambda b, nm=nm, dd=dd: shrunk(nm, b, dd)))
            best, best_n = None, None
            for choice, f in cands:
                n = overlap(nm, f)
                if n == 0:
                    best, best_n = choice, 0
                    break
                if best_n is None or n < best_n:
                    best, best_n = choice, n
            self.nudge[nm] = best + (anchors[nm],)
            if best_n:
                self.unresolved[nm] = best_n                      # px still inside the dilated matte (summed)
        return self.nudge
