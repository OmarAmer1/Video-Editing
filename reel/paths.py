"""Project paths. Override the work/model dirs with REEL_WORK / REEL_MODELS env vars."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("REEL_WORK", ROOT / "work"))
MODELS = Path(os.environ.get("REEL_MODELS", ROOT / "models"))
FONTS = Path(os.environ.get("REEL_FONTS", ROOT / "fonts"))
INPUT = Path(os.environ.get("REEL_INPUT", ROOT / "input"))
OUTPUT = Path(os.environ.get("REEL_OUTPUT", ROOT / "output"))

# Raw clips: the "29" take (further from camera) and the "30" take (closer).
CLIP_29 = INPUT / "29.mov"
CLIP_30 = INPUT / "30.mov"


def frames_dir(clip):
    return WORK / "frames" / clip


def matte_dir(clip):
    return WORK / "matte" / clip
