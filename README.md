# Thirty, in One Breath — a 29 → 30 birthday reel

**Deliverables** (`deliverables/`):
- `reel_29_to_30.mp4`: the final reel (1080×1920, 30 fps, 11.3 s, original score, −14 LUFS).
- `reel_29_to_30_no_music.mp4`: same picture with sound design only (−20 LUFS), for adding a song inside Instagram.
- `reel_cover.jpg`: suggested Instagram cover (frame 300).

A reproducible pipeline that turns two short phone clips (blowing out a **29** candle, then a **30** candle,
filmed from different distances) into a 1080×1920 Instagram Reel with a seamless 29 → 30 transformation.

**The story (11.3 s, loops):** the 29 plays as a silver black-and-white memory where only the candle flames
keep their gold, under *"one last wish at 29"*. One slow camera push carries us from the farther 29 take to the
closer 30 take. Mid-blow, with both candles still lit, the "29" candle morphs into "30" (and the on-screen
numeral rolls over like an odometer). The new 30 candle is the first thing in colour. She blows it out on the
melody's high note, colour floods the night from the wick, the old film gate bursts open, and she laughs in
slow motion under *"happy birthday, my love."*

## How the "she's further away in 29" problem is solved

Measured on the footage (face landmarks + background SIFT):

| | 29 take → 30 take |
|---|---|
| her face | 1.085× larger, +5.7° tilt, ~49 px lower |
| the background | 1.069× larger, +3.1° tilt, offset (+25, −8) px |

Face and background disagree by ~30–45 px, so no single zoom can line both up. A hard cut (the
original edit) jumps; a face-locked cross-fade makes the hotel slide. Instead:

1. **Two-layer progressive registration.** Over 2.8 s the 29 take's *foreground* (her + cake, from a
   Robust-Video-Matting matte) is eased onto the face transform while its *background* is eased onto the
   background transform; areas uncovered behind her are filled from a **clean plate** (median of both takes with
   the person removed). On screen it reads as one continuous dolly-in.
2. **Body pre-warp.** A smoothed optical-flow field (below the chin only) glides the cake/plate ~20 px onto the
   30 take's position during the push, so the cake doesn't jump either.
3. **Lit-to-lit morph.** The swap uses the best-matching pair of frames (C39 ↔ B61: both mid-blow, both candles
   lit, face residual < 1 px) and a RIFE v4.25 flow-morph (flip-ensembled). The candle region switches over
   3 frames under a warm glow so no in-between digits can be read; she then blows the 30 out on screen
   (the original edit cut from an already-out 29 to a lit 30).
4. **Colour separated from geometry.** Nothing changes colour while geometry moves: the 30 candle is born in
   colour after the morph, and the full colour bloom happens on the real flame-out.

## How it was made

1. **Analysis** of both takes: face landmarks, person mattes, camera shake, candle and flame tracking, and the
   exact frames where each flame dies (29: C41, 30: B66).
2. **Design panel**: four independent edit concepts (cinematic, trend, editorial, magical) scored by three judges
   (client intent, Instagram craft, technical feasibility) and merged into one shot list.
3. **Build** of the pipeline below, with the soundtrack and typography modules each built and independently verified.
4. **Adversarial QA** of a proxy by four reviewers (transition, craft, artefacts, A/V sync), every major finding
   re-verified before fixing. Fixes included linear RIFE slow motion (recursive midpoints), amber flame re-lighting,
   a digit-sized candle flash over the 29→30 hand-off, painting out the 30 wick's post-blow-out re-flare, skin/hijab
   colour protection, a BT.709-correct encode, and frame-accurate sound sync.

## Pipeline

```
reel/analyze.py     frames, RVM mattes, mediapipe face landmarks, background camera tracks, candle tracks
reel/geometry.py    take-to-take transforms, colour match, clean plate, body pre-warp, luma LUT
reel/rife.py        RIFE v4.25 interpolation + flow-morph with custom blend weights (CPU)
reel/edit.py        the edit: time remaps (speed ramps), cameras, two-layer composite, morph, look
reel/fx.py          grades (silver memory / golden present), flame re-lighting, blooms, grain, gate, particles
reel/titles.py      animated typography (Instrument Serif), odometer numeral
reel/soundtrack.py  original score (public-domain "Happy Birthday" arrangement) + synthesised sound design
reel/render.py      canvas warp → Real-ESRGAN upscale → look → ffmpeg
scripts/            setup, render, QA helpers
```

## Run it

```bash
scripts/setup.sh                          # deps, models (RIFE, RVM, mediapipe, Real-ESRGAN), fonts
mkdir -p input && cp <29 clip> input/29.mov && cp <30 clip> input/30.mov
python -m reel.analyze                    # ~1 min
python scripts/render_reel.py --proxy     # 540x960 preview, ~10 min on 4 CPU cores
python scripts/build_reel.py              # full 1080x1920 master + audio variants (~1 h on 4 CPU cores)
```

Options: `--name "Sara"` replaces "my love." with a name; `--no-titles` renders a clean master.
The audio has a `sfx_only` variant (no music) for adding a trending sound inside Instagram.

## Credits & licences

RIFE v4.25 architecture via [vs-rife](https://github.com/HolyWu/vs-rife) (MIT, `reel/third_party`),
[Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) ONNX model (GPL-3.0, downloaded, not
vendored), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) general-x4v3 (BSD-3), MediaPipe face landmarker
(Apache-2.0), Instrument Serif (SIL OFL), FluidR3_GM soundfont (MIT). "Happy Birthday to You" is in the public
domain; the arrangement and all sound design are original.
