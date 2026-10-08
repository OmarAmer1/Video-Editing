"""QA report for a rendered reel: contact sheets, motion/flicker metrics, loudness.

Usage: python scripts/qa.py output/reel.mp4 [outdir]
Writes sheet_all.png (every 6th frame), sheet_transition.png (every frame in a window), metrics.txt.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def frames(path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f


def sheet(imgs, cols, w=180, label0=0, step=1):
    tiles = []
    for i, f in enumerate(imgs):
        h = int(f.shape[0] * w / f.shape[1])
        t = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        cv2.putText(t, str(label0 + i * step), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else src.parent / (src.stem + "_qa"))
    out.mkdir(parents=True, exist_ok=True)
    fs = list(frames(src))
    n = len(fs)
    cv2.imwrite(str(out / "sheet_all.png"), sheet(fs[::6], 10, label0=0, step=6))
    win = json.loads(sys.argv[3]) if len(sys.argv) > 3 else [100, 140]
    cv2.imwrite(str(out / "sheet_transition.png"), sheet(fs[win[0]:win[1]], 8, w=200, label0=win[0]))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    lines = []
    prev = None
    for k, f in enumerate(fs):
        g = cv2.cvtColor(cv2.resize(f, (270, 480)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            fl = dis.calc(prev, g, None)
            mag = np.linalg.norm(fl, axis=2)
            diff = np.abs(g.astype(np.float32) - prev.astype(np.float32)).mean()
            lines.append(f"{k:4d} t={k / 30:6.3f} luma={g.mean():6.1f} diff={diff:6.2f} flow_med={np.median(mag):5.2f} "
                         f"flow_p95={np.percentile(mag, 95):5.2f}")
        prev = g
    loud = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(src), "-af", "ebur128=peak=true", "-f", "null", "-"],
                          capture_output=True, text=True).stderr
    summ = loud[loud.rfind("Summary:"):] if "Summary:" in loud else "no audio"
    (out / "metrics.txt").write_text(f"frames={n}\n" + "\n".join(lines) + "\n\nLOUDNESS\n" + summ)
    print(f"wrote {out} ({n} frames)")


if __name__ == "__main__":
    main()
