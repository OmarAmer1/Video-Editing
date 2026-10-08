"""Cross-take geometry: maps both takes into one reference space (the "30" take, frame REF_30).

* c30 frames are background-stabilised onto frame REF_30.
* c29 frames are background-stabilised onto frame REF_29, then mapped by the face transform
  REF_29 -> REF_30 (face Procrustes on 468 landmarks). That transform is what fixes the
  "she is further away / lower / tilted in the 29 take" problem: it is ~1.09x scale, ~5.7 deg roll.
  It is phased in progressively (identity -> full) so it reads as a slow camera push, and is exactly
  complete when the morph begins.
"""
import cv2
import numpy as np

from . import analyze, color, paths
from .frames import Clip, lerp_sim, mat3, stabilizer

REF_29, REF_30 = 40, 61


class Geometry:
    def __init__(self, an, ref29=REF_29, ref30=REF_30):
        self.an = an
        self.ref29, self.ref30 = ref29, ref30
        self.A_face = mat3(analyze.face_align(an["c29"]["faces"], ref29, an["c30"]["faces"], ref30))
        self.I = np.eye(3)
        # colour match of the 29 take onto the 30 take, fitted on the aligned reference pair
        a = cv2.warpAffine(Clip("c29").exact(ref29), self.A_face[:2], (464, 832), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REFLECT101)
        self.cmatch = color.fit_color_match(a, Clip("c30").exact(ref30))
        # background-only alignment of the 29 take onto the 30 take (SIFT on background pixels)
        self.A_bg = mat3(analyze.bg_align("c29", ref29, "c30", ref30))
        self._plate = None

    def T29(self, f, e=1.0):
        """c29 frame f -> reference space, with registration amount e (0 = untouched, 1 = face-locked)."""
        eyes = self.an["c29"]["faces"][self.ref29]["eyes"]
        A = lerp_sim(self.I, self.A_face, float(np.clip(e, 0, 1)), anchor=eyes) if e < 1 else self.A_face
        return A @ stabilizer(self.an, "c29", f, self.ref29)

    def T29_bg(self, f, e=1.0):
        """c29 frame f -> reference space aligning its BACKGROUND (used for the background layer)."""
        eyes = self.an["c29"]["faces"][self.ref29]["eyes"]
        A = lerp_sim(self.I, self.A_bg, float(np.clip(e, 0, 1)), anchor=eyes) if e < 1 else self.A_bg
        return A @ stabilizer(self.an, "c29", f, self.ref29)

    def plate(self):
        """Clean background plate (person removed) in reference space, median of all frames of both takes."""
        if self._plate is not None:
            return self._plate
        cache = paths.WORK / "plate.npy"
        if cache.exists():
            self._plate = np.load(cache)
            return self._plate
        W, H = 464, 832
        c29, c30 = Clip("c29"), Clip("c30")
        stack = []
        for clip, idxs in (("c30", range(0, c30.n, 2)), ("c29", range(0, c29.n, 3))):
            for i in idxs:
                if clip == "c30":
                    T, img, m = self.T30(i), c30.exact(i), c30.matte(i)
                else:
                    T, img, m = self.T29_bg(i), self.color29(c29.exact(i)), c29.matte(i)
                wi = cv2.warpAffine(img, T[:2], (W, H), flags=cv2.INTER_LINEAR)
                valid = cv2.warpAffine(np.ones((H, W), np.float32), T[:2], (W, H), flags=cv2.INTER_LINEAR)
                mm = cv2.warpAffine(cv2.dilate(m, np.ones((9, 9), np.uint8)), T[:2], (W, H),
                                    flags=cv2.INTER_LINEAR, borderValue=1)
                ok = (valid > 0.99) & (mm < 0.05)
                stack.append(np.where(ok[..., None], wi, np.nan).astype(np.float32))
        st = np.stack(stack)
        with np.errstate(all="ignore"):
            import warnings

            warnings.simplefilter("ignore")
            plate = np.nanmedian(st, axis=0)
        holes = ~np.isfinite(plate[..., 0])
        plate = np.nan_to_num(plate)
        plate = cv2.inpaint((plate * 255).astype(np.uint8), holes.astype(np.uint8) * 255, 9,
                            cv2.INPAINT_TELEA).astype(np.float32) / 255
        np.save(cache, plate)
        self._plate = plate
        return plate

    def T30(self, f):
        return stabilizer(self.an, "c30", f, self.ref30)

    def candle(self, clip, f):
        tr = self.an["candle"][clip]
        i0 = int(np.clip(np.floor(f), 0, len(tr) - 1))
        i1 = min(i0 + 1, len(tr) - 1)
        u = float(np.clip(f - i0, 0, 1))
        return np.array(tr[i0]) * (1 - u) + np.array(tr[i1]) * u

    def color29(self, img, amount=1.0):
        return color.apply_color_match(img, self.cmatch, amount)


# Measured flame life (source frames): the 29 candles die at C41, the 30 flame dies at B66.
FLAME_OUT = {"c29": (40.3, 41.0), "c30": (65.4, 66.0)}


def flame_alive(clip, f):
    """1 while the clip's candle flame burns, easing to 0 across its measured death."""
    a, b = FLAME_OUT[clip]
    u = float(np.clip((f - a) / (b - a), 0, 1))
    return 1 - u * u * (3 - 2 * u)


# Flame tips relative to the tracked candle centre (source px), measured on C39/B61:
# the 29 take's big '2' flame and small '9' flame, the 30 take's single '0' flame.
FLAME_OFFSETS = {"c29": [(-18.0, -11.0), (22.0, -12.0)], "c30": [(21.0, -18.0)]}


def flame_tips(geo, clip, f):
    c = geo.candle(clip, f)
    return [c + np.array(o) for o in FLAME_OFFSETS[clip]]


def fg_prewarp(geo, c39comp_R, b61, m_c39_R, m_b61, chin_b61, sigma=10.0):
    """Foreground pre-warp field so the 29 take's cake/plate/body glide onto the 30 take's (face untouched).

    c39comp_R: C39 foreground composited in reference space; b61: B61 frame (reference space).
    Returns D in reference space: B61(x) ~ C39comp(x + D(x)) on the body below the chin.
    """
    g = lambda x: cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fl = dis.calc(g(b61), g(c39comp_R), None)
    U = np.clip(cv2.dilate(np.maximum(m_b61, m_c39_R), np.ones((31, 31), np.uint8)), 0, 1)
    yy = np.arange(b61.shape[0], dtype=np.float32)[:, None]
    ramp = np.clip((yy - (chin_b61[1] + 8)) / 30.0, 0, 1)
    Wt = (U * ramp).astype(np.float32)
    num = cv2.GaussianBlur(fl * Wt[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(Wt, (0, 0), sigma) + 1e-4
    return (num / den[..., None] * Wt[..., None]).astype(np.float32)


def pull_back(D_R, M):
    """Express a reference-space displacement field in the source space of a layer mapped by similarity M:
    D_src(z) = L^-1 . D_R(M z)."""
    h, w = D_R.shape[:2]
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx = (M[0, 0] * X + M[0, 1] * Y + M[0, 2]).astype(np.float32)
    my = (M[1, 0] * X + M[1, 1] * Y + M[1, 2]).astype(np.float32)
    Ds = cv2.remap(D_R, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    Linv = np.linalg.inv(M[:2, :2])
    return np.einsum("ij,hwj->hwi", Linv, Ds).astype(np.float32)


def luma_lut(src, dst, mask, n=33):
    """1D lightness LUT (quantile match inside mask) taking src toward dst; returns (x, y) knots in LAB L."""
    ok = mask > 0.5
    ls = cv2.cvtColor(src.astype(np.float32), cv2.COLOR_RGB2LAB)[..., 0][ok]
    ld = cv2.cvtColor(dst.astype(np.float32), cv2.COLOR_RGB2LAB)[..., 0][ok]
    q = np.linspace(0, 100, n)
    x = np.maximum.accumulate(np.percentile(ls, q) + np.arange(n) * 1e-4)
    y = np.maximum.accumulate(np.percentile(ld, q))
    return np.concatenate([[0], x, [100]]), np.concatenate([[0], y, [100]])


def apply_luma_lut(img, lut, amount=1.0):
    if amount <= 0:
        return img
    lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    lab[..., 0] = L + (np.interp(L, *lut) - L) * amount
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
