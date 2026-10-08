"""Analysis pass: extract frames, person mattes (RVM), face landmarks (mediapipe), background camera motion.

Writes everything under WORK/. Run: python -m reel.analyze
"""
import json

import cv2
import numpy as np

from . import paths

CLIPS = {"c29": paths.CLIP_29, "c30": paths.CLIP_30}


def extract_frames(clip, src):
    import subprocess

    d = paths.frames_dir(clip)
    d.mkdir(parents=True, exist_ok=True)
    if not any(d.glob("*.png")):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vsync", "0", str(d / "%04d.png")], check=True)
    return sorted(d.glob("*.png"))


def load_rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def run_matting(clip, files):
    """Robust Video Matting (ResNet50). A short warm-up pass stabilises the recurrent state."""
    import onnxruntime as ort

    d = paths.matte_dir(clip)
    d.mkdir(parents=True, exist_ok=True)
    if len(list(d.glob("*.png"))) == len(files):
        return
    sess = ort.InferenceSession(str(paths.MODELS / "rvm_resnet50_fp32.onnx"), providers=["CPUExecutionProvider"])
    rec = [np.zeros([1, 1, 1, 1], np.float32)] * 4
    for passno, seq in enumerate([files[:15], files]):
        for f in seq:
            src = load_rgb(f).transpose(2, 0, 1)[None]
            _, pha, *rec = sess.run(None, {"src": src, "r1i": rec[0], "r2i": rec[1], "r3i": rec[2], "r4i": rec[3],
                                           "downsample_ratio": np.array([0.5], np.float32)})
            if passno == 1:
                cv2.imwrite(str(d / f.name), (pha[0, 0] * 255).clip(0, 255).astype(np.uint8))


def run_faces(files):
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(paths.MODELS / "face_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True)
    lm = vision.FaceLandmarker.create_from_options(opts)
    rows = []
    for f in files:
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if not res.face_landmarks:
            rows.append(None)
            continue
        L = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
        le, re = L[[33, 133]].mean(0), L[[362, 263]].mean(0)
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
        rows.append(dict(
            eyes=((le + re) / 2).tolist(), iod=float(np.linalg.norm(re - le)),
            roll=float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0]))),
            chin=L[152].tolist(), mouth=L[[13, 14]].mean(0).tolist(),
            pucker=bs.get("mouthPucker", 0.0),
            smile=(bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2,
            lm=L[:468].round(2).tolist()))
    return rows


def bg_mask(clip, name):
    m = cv2.imread(str(paths.matte_dir(clip) / name), 0)
    return (cv2.dilate(m, np.ones((31, 31), np.uint8)) < 20).astype(np.uint8) * 255


def run_bg_motion(clip, files):
    """Accumulated background similarity transform of every frame relative to frame 0 (3x3, frame0 -> frame i)."""
    acc = np.eye(3)
    out = [acc.tolist()]
    prev = cv2.cvtColor(cv2.imread(str(files[0])), cv2.COLOR_BGR2GRAY)
    for f in files[1:]:
        g = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev, 400, 0.01, 8, mask=bg_mask(clip, files[files.index(f) - 1].name))
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None)
        ok = st[:, 0] == 1
        M, _ = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
        acc = np.vstack([M, [0, 0, 1]]) @ acc
        out.append(acc.tolist())
        prev = g
    return out


def bg_align(clip_a, ia, clip_b, ib):
    """Similarity mapping background of clip_a frame ia onto clip_b frame ib (SIFT on background only)."""
    fa = paths.frames_dir(clip_a) / f"{ia + 1:04d}.png"
    fb = paths.frames_dir(clip_b) / f"{ib + 1:04d}.png"
    a = cv2.cvtColor(cv2.imread(str(fa)), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.imread(str(fb)), cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(3000)
    ka, da = sift.detectAndCompute(a, bg_mask(clip_a, fa.name))
    kb, db = sift.detectAndCompute(b, bg_mask(clip_b, fb.name))
    good = [x for x, y in cv2.BFMatcher().knnMatch(da, db, k=2) if x.distance < 0.75 * y.distance]
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    M, _ = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    return M


def face_align(faces_a, ia, faces_b, ib):
    """Similarity mapping the face in frame ia onto the face in frame ib (468 mesh landmarks, LMedS)."""
    A = np.float32(faces_a[ia]["lm"])
    B = np.float32(faces_b[ib]["lm"])
    M, _ = cv2.estimateAffinePartial2D(A, B, method=cv2.LMEDS)
    return M


def load():
    return json.loads((paths.WORK / "analysis.json").read_text())


def main():
    paths.WORK.mkdir(parents=True, exist_ok=True)
    out = {}
    for clip, src in CLIPS.items():
        files = extract_frames(clip, src)
        print(f"[{clip}] {len(files)} frames; matting...")
        run_matting(clip, files)
        print(f"[{clip}] faces...")
        faces = run_faces(files)
        print(f"[{clip}] background motion...")
        out[clip] = dict(n=len(files), faces=faces, bg=run_bg_motion(clip, files))
    (paths.WORK / "analysis.json").write_text(json.dumps(out))
    print("wrote", paths.WORK / "analysis.json")


if __name__ == "__main__":
    main()


def track_point(clip, ref, pt, box=26):
    """Track a point (e.g. the candle) through a clip with median LK flow of features around it."""
    files = sorted(paths.frames_dir(clip).glob("*.png"))
    gray = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY) for f in files]
    out = {ref: np.array(pt, np.float64)}
    for step in (1, -1):
        p = np.array(pt, np.float64)
        i = ref
        while 0 <= i + step < len(gray):
            m = np.zeros_like(gray[i])
            x, y = int(p[0]), int(p[1])
            m[max(y - box, 0):y + box, max(x - box, 0):x + box] = 255
            p0 = cv2.goodFeaturesToTrack(gray[i], 60, 0.01, 3, mask=m)
            if p0 is None:
                out[i + step] = p.copy()
                i += step
                continue
            p1, st, _ = cv2.calcOpticalFlowPyrLK(gray[i], gray[i + step], p0, None, winSize=(21, 21), maxLevel=3)
            ok = st[:, 0] == 1
            if ok.sum() >= 3:
                p = p + np.median((p1 - p0)[ok, 0], axis=0)
            i += step
            out[i] = p.copy()
    return [out[i].tolist() for i in range(len(gray))]


# Candle centres picked on reference frames (source pixel coords): c29 frame 40, c30 frame 61.
CANDLE_REF = {"c29": (40, (233.0, 352.0)), "c30": (61, (228.0, 436.0))}


def candle_tracks():
    return {clip: track_point(clip, ref, pt) for clip, (ref, pt) in CANDLE_REF.items()}
