"""Render selected output frames (proxy or full) into a labelled contact sheet for quick review.
python scripts/frames_sheet.py OUT.png 0 12 96 103 ... [--full] [--titles]"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reel import analyze, edit, frames as fr, paths, render  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full = "--full" in sys.argv
    out, ks = args[0], [int(a) for a in args[1:]]
    cfg = dict(edit.CFG)
    if not full:
        cfg["out_size"] = (540, 960)
    titles = None
    if "--titles" in sys.argv:
        from reel.titles import Titles

        titles = Titles(cfg["out_size"])
    E = edit.Edit(analyze.load(), cfg, titles=titles)
    rife = fr.load_rife()
    clips = {"c29": fr.Clip("c29", rife), "c30": fr.Clip("c30", rife)}
    up = None
    if full and "--gan" in sys.argv:
        from reel.upscale import Upscaler

        up = Upscaler(str(paths.MODELS / "realesr-general-x4v3.pth"), str(paths.MODELS / "realesr-general-wdn-x4v3.pth"),
                      denoise=0.5)
    R = render.Renderer(E, clips, rife, upscaler=up, canvas_scale=0.45 if full else 0.86)
    tiles = []
    for k in ks:
        img = (R.frame(k) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if full:
            cv2.imwrite(out.replace(".png", f"_{k:03d}.png"), img)
        t = cv2.resize(img, (270, 480), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"f{k}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(t)
    cols = min(8, len(tiles))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


if __name__ == "__main__":
    main()
