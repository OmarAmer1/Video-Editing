"""Render the reel.  python scripts/render_reel.py [--proxy] [--frames a:b] [--out path] [--audio wav]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reel import analyze, edit, frames as fr, paths, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", action="store_true", help="540x960, Lanczos upscale (fast preview)")
    ap.add_argument("--frames", default=None, help="a:b output frame range")
    ap.add_argument("--out", default=None)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--gan-detail", type=float, default=0.7)
    ap.add_argument("--no-titles", action="store_true", help="clean master without typography")
    ap.add_argument("--end-card", action="store_true", help="add the 'happy birthday, my love.' end card")
    ap.add_argument("--name", default=None, help="end card with her name instead of 'my love.' (implies --end-card)")
    ap.add_argument("--png-dir", default=None, help="write lossless PNG frames here instead of encoding")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    a = ap.parse_args()
    if a.threads:
        import torch

        torch.set_num_threads(a.threads)
    cfg = dict(edit.CFG)
    if a.proxy:
        cfg["out_size"] = (540, 960)
    titles = None
    if not a.no_titles:
        from reel.titles import Titles

        titles = Titles(cfg["out_size"], name=a.name, end_card=a.end_card or a.name is not None)
    E = edit.Edit(analyze.load(), cfg, titles=titles)
    rife = fr.load_rife()
    clips = {"c29": fr.Clip("c29", rife), "c30": fr.Clip("c30", rife)}
    up = None
    if not a.proxy:
        from reel.upscale import Upscaler

        up = Upscaler(str(paths.MODELS / "realesr-general-x4v3.pth"),
                      str(paths.MODELS / "realesr-general-wdn-x4v3.pth"), denoise=0.5)
    R = render.Renderer(E, clips, rife, upscaler=up, gan_detail=a.gan_detail,
                        canvas_scale=0.86 if a.proxy else 0.4)
    frames = None
    if a.frames:
        s, e = map(int, a.frames.split(":"))
        frames = range(s, e)
    if a.png_dir:
        R.render_png(a.png_dir, list(frames if frames is not None else range(E.n)), log_every=10)
        return
    paths.OUTPUT.mkdir(parents=True, exist_ok=True)
    out = a.out or str(paths.OUTPUT / ("proxy.mp4" if a.proxy else "reel_noaudio.mp4"))
    R.render(out, frames=frames, audio=a.audio, log_every=30)
    print("wrote", out)


if __name__ == "__main__":
    main()
