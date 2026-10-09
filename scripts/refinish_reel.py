"""Refinish an already-rendered reel when the source clips are not at hand (they were never on this Mac).

Two client notes on the 2026-10-08 render, applied to its pixels:

  * "the black and white makes her look like an alien; make it more of an old black and white":
    the silver memory (f0-f159, and the bloom's outer edge after it) is re-toned to the antique sepia look of
    fx.mono_antique. Per pixel the silver tone curve is inverted, the face is re-shaded from face landmarks
    (the silver's red-leaning mix had washed skin, lips and brows into one pale grey: skin goes a touch darker
    than the hijab, lips and brows back to a natural ratio to the skin around them), then clarity, a soft-focus
    glow, the antique curve and sepia toning. Only the silver -> antique difference is added, weighted by how
    much of the pixel is the past (the colour bloom's geometry from edit.CFG), so flames, glows, the colour
    candle, the titles and the film gate keep their pixels.
  * "delete the happy birthday my love": the end card is lifted off f176-f339. reel.titles re-renders the exact
    text layer; where it is translucent (its shadow) the frame is un-composited, under the glyphs the night sky
    is inpainted and given fresh grain; the odometer numeral that shares f176-f198 is composited back.
    Its chime (soundtrack.title_glint, f204) is subtracted from both soundtracks.

python scripts/refinish_reel.py [--src REEL.mp4] [--src-sfx REEL_NO_MUSIC.mp4] [--workers 6] [--reuse-frames]
The sources default to the 10-08 render as committed in git (the deliverables now hold the refinished reel).
Outputs in output/: reel.mp4, reel_sfx_only.mp4, their 2x cuts, reel_share.mp4 (CRF 21, smaller), reel_cover.jpg.
"""
import argparse
import inspect
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reel import fx, paths  # noqa: E402
from reel.edit import CFG  # noqa: E402

W, H, N, FPS = 1080, 1920, CFG["n_frames"], CFG["fps"]
REC709 = np.array([0.2126, 0.7152, 0.0722], np.float32)
WORK = paths.WORK / "refinish"
END = ("end1", "end2")

# colour-bloom origin on the drop frame (output px), fitted to the rendered f141-f146 bloom
# (edit.Edit.wick_at_drop() needs the clip analysis, which is not available without the clips)
WICK = (562.0, 912.0)
FIRST_CUT = "9d34dd8"                      # last commit of the 10-08 render (the sources of this refinish)

# the silver look being undone: fx.mono_silver's own defaults
_SIL = {k: v.default for k, v in inspect.signature(fx.mono_silver).parameters.items() if k != "img"}
_ANT = {k: v.default for k, v in inspect.signature(fx.mono_antique).parameters.items() if k != "img"}

# Antique curve on the silver mix (fx.mono_antique's own curve expects its red-shy mix, which is darker on skin):
# faded blacks, a lower-mid face, creamy rolled-off highlights.
CURVE = ((0, 0.08), (0.15, 0.145), (0.45, 0.42), (0.75, 0.69), (0.9, 0.80), (1, 0.90))
SKIN, LIPS, BROWS = 0.90, 0.82, 0.80      # skin gain; lips/brows target ratio to the skin around them
CLARITY, CLARITY_PX = 0.25, 22.0
DIFFUSION, DIFFUSION_PX = _ANT["diffusion"], _ANT["diffusion_px"]

# mediapipe face-mesh indices
LIP_OUT = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIP_IN = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
BROW_L = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_R = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
OVAL = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
        356, 389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162, 127]


# ------------------------------------------------------------------------------------------- frames i/o
def from_git(src):
    """'git:REV:PATH' -> that file as committed, extracted once into WORK; any other value is a plain path."""
    if not src.startswith("git:"):
        return src
    rev, path = src[4:].split(":", 1)
    out = WORK / f"{rev}_{Path(path).name}"
    if not out.exists():
        WORK.mkdir(parents=True, exist_ok=True)
        out.write_bytes(subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:{path}"], capture_output=True,
                                       check=True).stdout)
    return str(out)


def decode(src, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(list(out_dir.glob("*.png"))) == N:
        return
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf",
                    "scale=in_color_matrix=bt709:in_range=tv:flags=spline+accurate_rnd+full_chroma_int,format=rgb48le",
                    "-fps_mode", "passthrough", str(out_dir / "%04d.png")], check=True)


def load(d, k):
    return cv2.cvtColor(cv2.imread(str(d / f"{k + 1:04d}.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype(np.float32) / 65535


def save(d, k, img):
    cv2.imwrite(str(d / f"{k + 1:04d}.png"),
                cv2.cvtColor((np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16), cv2.COLOR_RGB2BGR))


# ------------------------------------------------------------------------------------------- silver -> antique
def silver_rgb(y):
    """fx.mono_silver's split toning of its post-curve value y."""
    sh, hi = fx._hex(_SIL["shadow"]), fx._hex(_SIL["highlight"])
    base = np.repeat(np.asarray(y, np.float32)[..., None], 3, -1)
    w = base[..., :1]
    return base * (1 - _SIL["split"]) + _SIL["split"] * (base * (hi / hi.max()) * w + (sh + base * (1 - sh)) * (1 - w))


_Y = np.linspace(0, 1, 4096).astype(np.float32)
_L_OF_Y = silver_rgb(_Y) @ REC709                                   # post-curve value -> displayed luma
_X = np.linspace(0, 1, 4096)
_Y_OF_X = PchipInterpolator(*zip(*_SIL["pts"]))(_X)                 # silver curve: mix -> post-curve value
_CURVE = PchipInterpolator(*zip(*CURVE))


def antique_rgb(y):
    sh, hi = np.array(_ANT["shadow"], np.float32), np.array(_ANT["highlight"], np.float32)
    sh, hi = sh / (sh @ REC709), hi / (hi @ REC709)
    w = np.asarray(y, np.float32)[..., None]
    return np.clip(w * (1 + (sh * (1 - w) + hi * w - 1) * _ANT["tone"]), 0, 1)


def silver_mix(img):
    """Displayed pixel -> (silver post-curve value y, silver pre-curve mix x)."""
    y = np.interp(img @ REC709, _L_OF_Y, _Y).astype(np.float32)
    return y, np.interp(y, _Y_OF_X, _X).astype(np.float32)


def regrade(img, xmul, p):
    y, x = silver_mix(img)
    x = x * xmul
    x = x + (x - cv2.GaussianBlur(x, (0, 0), CLARITY_PX)) * CLARITY
    b = cv2.GaussianBlur(x, (0, 0), DIFFUSION_PX)
    x = x * (1 - DIFFUSION) + np.maximum(x, b) * DIFFUSION
    new = antique_rgb(_CURVE(np.clip(x, 0, 1)).astype(np.float32))
    return img + p[..., None] * (new - silver_rgb(y))


# ------------------------------------------------------------------------------------------- where the past is
def bloom_radius(k):
    d0, d1, ra, rb = CFG["bloom2"]
    u = min((k - d0) / (d1 - d0), 1.0)
    return ra + (rb - ra) * float(fx.ease_in_out_sine(u) * 0.35 + fx.ease_out_cubic(u) * 0.65)


def pastness(k):
    """1 = silver memory, 0 = colour (edit.colour_mask's drop bloom; the stage-1 candle is caught by its colour)."""
    if k < CFG["bloom2"][0]:
        return np.ones((H, W), np.float32)
    return 1 - fx.radial_mask((H, W), WICK, bloom_radius(k), feather=0.45)


def gate_mask(k):
    """Inside of edit.look()'s rounded film gate (1 = picture)."""
    g0, g1 = CFG["gate"]
    go = float(1 - 2 ** (-10 * np.clip((k - g0) / (g1 - g0), 0, 1))) if k >= g0 else 0.0
    if go >= 0.999:
        return np.ones((H, W), np.float32)
    gi = float(fx.ease_out_cubic(k / 5.0)) if k < 5 else 1.0
    go = 1 - (1 - go) * gi
    inset, radius = 26 * (1 - go) - 90 * go, 56 * (1 - go)
    if inset <= -8 and radius <= 1:
        return np.ones((H, W), np.float32)
    return fx.rounded_rect_mask((H, W), (inset, inset, W - inset, H - inset), max(radius, 0.5), feather=4.0)


def colourful(img):
    c = img.max(-1) - img.min(-1)
    return np.clip((c - 0.07) / 0.08, 0, 1) ** 2 * (3 - 2 * np.clip((c - 0.07) / 0.08, 0, 1))


# ------------------------------------------------------------------------------------------- text layers
def text_layers(T, k, names, bloom):
    """(premultiplied RGB, alpha incl. shadow, fill alpha) of the named title items, composited as Titles.render."""
    rgb = np.zeros((H, W, 3), np.float32)
    a = np.zeros((H, W), np.float32)
    sha = np.zeros((H, W), np.float32)
    for it in T._plan(k, bloom):
        if it["name"] not in names:
            continue
        r = T._draw_parts(it)
        if r is None:
            continue
        fill, S, x, y = r
        sl = (slice(y, y + fill.shape[0]), slice(x, x + fill.shape[1]))
        sha[sl] += S * (1 - sha[sl])
        rgb[sl] = rgb[sl] * (1 - fill[..., 3:4]) + fill[..., :3]
        a[sl] = a[sl] * (1 - fill[..., 3]) + fill[..., 3]
    fill_a = a.copy()
    return rgb, a + sha * (1 - a), fill_a


def leak_flash(k):
    """edit.look()'s loop-seam light leak + flash (screened over the frame after the titles)."""
    t = k / FPS
    lk = 0.8 * max(1 - k / 3.0, 0) ** 2 + 0.8 * max((k - 334) / 5.0, 0) ** 2
    out = np.zeros((H, W, 3), np.float32)
    if lk > 0:
        out = fx.add_light(out, fx.light_leak((H, W), t, seed=4, strength=lk, color=(1.0, 0.70, 0.42),
                                              pos=(0.35, 0.6), scale=1.6))
    fl = 0.35 * max(1 - k / 2.0, 0) ** 2 + 0.35 * max((k - 336) / 3.0, 0) ** 2
    if fl > 0:
        out = fx.add_light(out, np.ones((H, W, 1), np.float32) * fx._hex("#FFC48A") * fl)
    return out


def inpaint(img, mask, radius=7):
    """Telea inpainting of a float RGB image (via 16 bit: OpenCV's Telea is wrong on float32 input)."""
    u16 = (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16)
    return np.dstack([cv2.inpaint(np.ascontiguousarray(u16[..., c]), mask, radius, cv2.INPAINT_TELEA)
                      for c in range(3)]).astype(np.float32) / 65535


def lift_end_card(img, k, T, vig):
    """Remove the end card from a present-world frame (vignette 0.25 there), keeping the numeral.

    Where the end card has ink (fill alpha, including its blurred fade-ins) the night sky is painted in from around
    it: subtracting bright text from a dark sky leaves dark ghosts at the smallest model error. Where only its soft
    shadow lies, the shadow is divided out. Leak, flash and vignette are undone first and re-applied after, so any
    error in modelling them cancels."""
    keep = ("kicker", "numeral")
    C_num, A_num, F_num = text_layers(T, k, keep, 1.0)
    C_all, A_all, _ = text_layers(T, k, keep + END, 1.0)
    _, A_end, F_end = text_layers(T, k, END, 1.0)
    if A_end.max() <= 0.002:
        return img
    ys, xs = np.nonzero(A_end > 0.002)
    pad = 24
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad, H), max(xs.min() - pad, 0), min(xs.max() + pad, W)
    sl = (slice(y0, y1), slice(x0, x1))
    l = leak_flash(k)[sl]
    v = vig[sl]
    X = (img[sl] / v - l) / np.maximum(1 - l, 1e-3)                      # the composite before leak and vignette
    A_n, C_n = A_num[sl][..., None], C_num[sl]
    B = (X - C_all[sl]) / np.maximum(1 - A_all[sl], 0.05)[..., None]      # sky under all text (shadow divided out)
    hole = cv2.dilate((F_end[sl] > 0.01).astype(np.uint8), np.ones((7, 7), np.uint8))
    unknown = hole | cv2.dilate((F_num[sl] > 0.05).astype(np.uint8), np.ones((5, 5), np.uint8))   # no numeral smear
    B_fill = inpaint(B, unknown)
    hs = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 1.0)[..., None]
    out_X = (B_fill * hs + B * (1 - hs)) * (1 - A_n) + C_n
    out = v * (out_X * (1 - l) + l)
    # fresh grain where the sky was painted in (edit.look()'s 35 mm grain: amount 0.018, size 1.2, chroma 0.2)
    g = fx.Grain(out.shape, seed=77)
    noise = g.apply(np.full_like(out, 0.5), k, amount=0.018, size=1.2, chroma=0.2) - 0.5
    y = out @ REC709
    out = out + noise * (4 * y * (1 - y) * 0.8 + 0.2)[..., None] * hs
    M = np.clip((A_end[sl] - 0.002) / 0.018, 0, 1)[..., None]
    res = img.copy()
    res[sl] = img[sl] * (1 - M) + out * M
    return res


# ------------------------------------------------------------------------------------------- faces
def detect_faces(src_dir, frames):
    """Face-mesh landmarks (output px) per frame; the washed-out silver face needs a crop (and a gamma/CLAHE retry)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(paths.MODELS / "face_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_faces=1, min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3))
    X0, Y0 = 140, 100

    def det(rgb):
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        if not res.face_landmarks:
            return None
        h, w = rgb.shape[:2]
        return [[p.x * w + X0, p.y * h + Y0] for p in res.face_landmarks[0][:478]]

    rows = []
    for k in frames:
        rgb = (load(src_dir, k) * 255 + 0.5).astype(np.uint8)[Y0:1100, X0:940]
        r = det(rgb)
        if r is None:
            r = det(((rgb.astype(np.float32) / 255) ** 2.2 * 255).astype(np.uint8))
        if r is None:
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            lab[..., 0] = cv2.createCLAHE(3.0, (8, 8)).apply(lab[..., 0])
            r = det(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
        rows.append(r)
    # fill misses from the nearest detected frame, then smooth 1-2-1 in time (no landmark jitter on the lips)
    known = [i for i, r in enumerate(rows) if r is not None]
    arr = np.array([rows[min(known, key=lambda j: abs(j - i))] for i in range(len(rows))], np.float32)
    sm = arr.copy()
    sm[1:-1] = (arr[:-2] + 2 * arr[1:-1] + arr[2:]) / 4
    return sm


def poly_mask(pts, idx, feather):
    m = np.zeros((H, W), np.float32)
    p = np.array([pts[i] for i in idx], np.float32)
    cv2.fillPoly(m, [np.round(p * 8).astype(np.int32)], 1.0, lineType=cv2.LINE_AA, shift=3)
    return cv2.GaussianBlur(m, (0, 0), feather) if feather > 0 else m


def face_masks(pts):
    iod = float(np.linalg.norm(pts[[33, 133]].mean(0) - pts[[362, 263]].mean(0)))
    f = iod / 10
    skin = poly_mask(pts, OVAL, 1.6 * f)
    lips = np.clip(poly_mask(pts, LIP_OUT, 0.25 * f) - poly_mask(pts, LIP_IN, 0.2 * f), 0, 1)
    brows = np.maximum(poly_mask(pts, BROW_L, 0.25 * f), poly_mask(pts, BROW_R, 0.25 * f))
    return f, skin, lips, brows


def ring_mean(x, pts, idx, inner, outer):
    m = poly_mask(pts, idx, 0)
    k1 = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outer + 1, 2 * outer + 1)))
    k0 = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inner + 1, 2 * inner + 1)))
    return float(x[(k1 > 0.5) & (k0 < 0.5)].mean())


def feature_gains(img, pts):
    """Lips / brows gains that bring their ratio to the surrounding skin to LIPS / BROWS (never lighter)."""
    _, x = silver_mix(img)
    f, _, _, _ = face_masks(pts)
    lip_core = np.clip(poly_mask(pts, LIP_OUT, 0) - poly_mask(pts, LIP_IN, 0), 0, 1) > 0.5
    brow_core = np.maximum(poly_mask(pts, BROW_L, 0), poly_mask(pts, BROW_R, 0)) > 0.5
    r_lip = float(x[lip_core].mean()) / ring_mean(x, pts, LIP_OUT, int(0.4 * f), int(1.4 * f))
    r_brow = float(x[brow_core].mean()) / ring_mean(x, pts, BROW_L, int(0.4 * f), int(1.2 * f))
    return float(np.clip(LIPS / r_lip, 0.72, 1.0)), float(np.clip(BROWS / r_brow, 0.75, 1.0))


def face_xmul(pts, g_lip, g_brow):
    _, skin, lips, brows = face_masks(pts)
    return (1 - (1 - SKIN) * skin) * (1 - (1 - g_lip) * lips) * (1 - (1 - g_brow) * brows)


# ------------------------------------------------------------------------------------------- per frame
_STATE = {}


def _init(src_dir, out_dir, faces, gains):
    from reel.titles import Titles

    _STATE.update(src=src_dir, out=out_dir, faces=faces, gains=gains,
                  T=Titles((W, H), end_card=True),
                  vig25=fx.vignette(np.ones((H, W, 3), np.float32), 0.25))


def process(k):
    st = _STATE
    img = load(st["src"], k)
    p = pastness(k)
    if p.max() > 0.003:
        _, _, fill = text_layers(st["T"], k, ("kicker", "numeral"), None)
        pe = p * gate_mask(k) * (1 - fill) * (1 - colourful(img))
        xmul = np.ones((H, W), np.float32)
        if k < len(st["faces"]):
            xmul = face_xmul(st["faces"][k], *st["gains"][k])
        img = regrade(img, xmul, pe)
    if k >= 170:
        img = lift_end_card(img, k, st["T"], st["vig25"])
    save(st["out"], k, img)
    return k


# ------------------------------------------------------------------------------------------- sound
def decode_audio(src):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-vn", "-ac", "2", "-ar", "48000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)


def remove_glint(x):
    """Subtract the end-card chime: re-synthesise it (deterministic), find its exact offset, then fit its gain on
    40 ms windows (the bus compressor and the limiter move it a little) and subtract."""
    from reel import soundtrack as snd

    g = snd.dc_block(snd.title_glint() * snd.dbg(4.0), 15.0)        # as sfx_bus placed it, through the mix's DC block
    n = min(len(g), len(x) - snd.S(204) - 400)
    g = g[:n]
    s0 = snd.S(204)
    seg = x[s0 - 400:s0 + 4800 + 400]
    best = max(range(-400, 401), key=lambda d: float(np.sum(seg[400 + d:400 + d + 4800] * g[:4800])))
    a0 = s0 + best
    head = slice(0, int(0.6 * 48000))                                  # the loud part fixes the overall gain
    g0 = float(np.sum(g[head] * x[a0:a0 + n][head]) / np.sum(g[head] * g[head]))
    win = 1920                                                         # 40 ms: the compressor/limiter ride it a little
    e_peak = max(float(np.sum(g[i:i + win] ** 2)) for i in range(0, n, win))
    gains = []
    for i in range(0, n, win):
        gg, ob = g[i:i + win], x[a0 + i:a0 + i + min(win, n - i)]
        e = float(np.sum(gg * gg))
        gains.append(float(np.clip(np.sum(gg * ob) / e, 0.8 * g0, 1.2 * g0)) if e > 0.01 * e_peak else g0)
    gl = np.convolve(np.pad(np.array(gains), 2, mode="edge"), np.ones(5) / 5, mode="valid")
    y = x.copy()
    y[a0:a0 + n] -= g * np.repeat(gl, win)[:n, None]
    return y, dict(offset=best, gain=round(g0, 4))


def write_wav(path, x):
    import soundfile

    soundfile.write(str(path), x.astype(np.float32), 48000, subtype="FLOAT")


def band_db(x, a, b, f0, f1):
    seg = x[a:b].mean(1)
    sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    fr = np.fft.rfftfreq(len(seg), 1 / 48000)
    return 20 * np.log10(np.sqrt(np.mean(sp[(fr >= f0) & (fr <= f1)] ** 2)) + 1e-12)


# ------------------------------------------------------------------------------------------- encode
X264 = ["-c:v", "libx264", "-preset", "slow", "-tune", "film", "-pix_fmt", "yuv420p", "-profile:v", "high",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
TO_YUV = "scale=out_color_matrix=bt709:out_range=tv:flags=spline+accurate_rnd+full_chroma_int"


def encode(frames_dir, out, crf=16, every_other=False):
    """Picture only (as build_reel.py: encode once, then mux each soundtrack onto a copy of the stream)."""
    vf = TO_YUV if not every_other else "select=not(mod(n\\,2)),setpts=N/30/TB," + TO_YUV
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-start_number", "1",
                    "-i", str(frames_dir / "%04d.png"), "-vf", vf, *X264, "-crf", str(crf), "-movflags", "+faststart",
                    str(out)], check=True)


def mux(video, wav, out, tempo=None):
    """Video copied; both streams kept whole (the WAV is exactly the picture's length: -shortest drops a frame)."""
    af = ["-af", f"atempo={tempo}"] if tempo else []
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", *af, "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-movflags", "+faststart",
                    str(out)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"git:{FIRST_CUT}:deliverables/reel_29_to_30.mp4")
    ap.add_argument("--src-sfx", default=f"git:{FIRST_CUT}:deliverables/reel_29_to_30_no_music.mp4")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--frames", default=None, help="a:b (picture only, for previews)")
    ap.add_argument("--reuse-frames", action="store_true", help="keep work/refinish/frames, only encode and mux")
    a = ap.parse_args()
    src_dir, out_dir = WORK / "src", WORK / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    a.src, a.src_sfx = from_git(a.src), from_git(a.src_sfx)
    print("decoding", a.src)
    decode(a.src, src_dir)

    fpath = WORK / "faces.json"
    n_face = CFG["bloom2"][1] + 1
    if fpath.exists():
        faces = np.array(json.loads(fpath.read_text()), np.float32)
    else:
        print("face landmarks...")
        faces = detect_faces(src_dir, range(n_face))
        fpath.write_text(json.dumps(np.round(faces, 2).tolist()))
    gains = np.array([feature_gains(load(src_dir, k), faces[k]) for k in range(n_face)])
    for c in range(2):                                      # steady lips/brows: median 5, then mean 5
        g = gains[:, c]
        med = np.array([np.median(g[max(i - 2, 0):i + 3]) for i in range(len(g))])
        gains[:, c] = np.convolve(np.pad(med, 2, mode="edge"), np.ones(5) / 5, mode="valid")

    ks = list(range(N))
    if a.frames:
        s, e = map(int, a.frames.split(":"))
        ks = list(range(s, e))
    if a.reuse_frames and len(list(out_dir.glob("*.png"))) == N:
        ks = []
    with Pool(a.workers, initializer=_init, initargs=(src_dir, out_dir, faces, gains)) as pool:
        for i, k in enumerate(pool.imap_unordered(process, ks)):
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(ks)} frames", flush=True)
    if a.frames:
        return

    out = paths.OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    wavs = {}
    for name, src in (("full", a.src), ("sfx_only", a.src_sfx)):
        x = decode_audio(src)
        from reel.soundtrack import S
        before = band_db(x, S(204), S(204) + 24000, 3400, 3650)
        y, info = remove_glint(x)
        import pyloudnorm

        meter = pyloudnorm.Meter(48000)                      # keep the master's loudness (the chime had a share of it)
        y = y * 10 ** ((meter.integrated_loudness(x) - meter.integrated_loudness(y)) / 20)
        after = band_db(y, S(204), S(204) + 24000, 3400, 3650)
        ref = band_db(x, S(190), S(190) + 24000, 3400, 3650)
        print(f"{name}: chime band {before:.1f} -> {after:.1f} dB (same band just before it: {ref:.1f} dB)", info)
        wavs[name] = WORK / f"audio_{name}.wav"
        write_wav(wavs[name], y)

    pics = {"": (16, False), "_2x": (16, True), "_share": (21, False)}
    for tag, (crf, half) in pics.items():
        encode(out_dir, WORK / f"picture{tag}.mp4", crf=crf, every_other=half)
    mux(WORK / "picture.mp4", wavs["full"], out / "reel.mp4")
    mux(WORK / "picture.mp4", wavs["sfx_only"], out / "reel_sfx_only.mp4")
    mux(WORK / "picture_2x.mp4", wavs["full"], out / "reel_2x.mp4", tempo=2.0)
    mux(WORK / "picture_2x.mp4", wavs["sfx_only"], out / "reel_sfx_only_2x.mp4", tempo=2.0)
    mux(WORK / "picture_share.mp4", wavs["full"], out / "reel_share.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out / "reel.mp4"), "-vf", "select=eq(n\\,300)",
                    "-frames:v", "1", "-q:v", "2", str(out / "reel_cover.jpg")], check=True)
    print("done:", *sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
