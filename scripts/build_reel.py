"""Build the deliverables: full-quality picture, soundtrack variants, muxed reels and a cover frame.

python scripts/build_reel.py [--name NAME] [--skip-video] [--frames a:b]
Outputs in output/: reel.mp4 (score + sound design), reel_sfx_only.mp4 (no music: add an Instagram sound),
reel_cover.jpg (suggested cover, frame 300).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reel import paths  # noqa: E402


def mux(video, audio, out):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-shortest",
                    "-movflags", "+faststart", str(out)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--workers", type=int, default=2, help="parallel render processes (CPU cores are split)")
    ap.add_argument("--resume", action="store_true", help="keep already-rendered frames in work/final_frames")
    a = ap.parse_args()
    out = paths.OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    video = out / "reel_picture.mp4"
    if not a.skip_video:
        import os

        import shutil

        pngs = paths.WORK / "final_frames"
        if pngs.exists() and not a.resume:
            shutil.rmtree(pngs)
        n, w = 340, max(a.workers, 1)
        bounds = [round(n * i / w) for i in range(w + 1)]
        threads = max((os.cpu_count() or 4) // w, 1)
        procs = []
        for i in range(w):
            cmd = [sys.executable, str(ROOT / "scripts" / "render_reel.py"), "--png-dir", str(pngs),
                   "--frames", f"{bounds[i]}:{bounds[i + 1]}", "--threads", str(threads)]
            if a.name:
                cmd += ["--name", a.name]
            procs.append(subprocess.Popen(cmd))
        if any([p.wait() for p in procs]):
            raise SystemExit("a render worker failed")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", str(pngs / "%04d.png"),
                        "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-tune", "film", "-pix_fmt", "yuv420p",
                        "-profile:v", "high", "-color_primaries", "bt709", "-color_trc", "bt709",
                        "-colorspace", "bt709", "-movflags", "+faststart", str(video)], check=True)
    from reel import soundtrack

    for variant, name in (("full", "reel.mp4"), ("sfx_only", "reel_sfx_only.mp4")):
        wav = paths.WORK / f"soundtrack_{variant}.wav"
        # the no-music variant sits quietly so an Instagram song can go on top of it
        info = soundtrack.build(str(wav), variant=variant, target_lufs=-14.0 if variant == "full" else -20.0)
        print(variant, info)
        mux(video, wav, out / name)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out / "reel.mp4"), "-vf", "select=eq(n\\,300)",
                    "-frames:v", "1", "-q:v", "2", str(out / "reel_cover.jpg")], check=True)
    print("done:", *sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
