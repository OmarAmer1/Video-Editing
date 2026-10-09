"""The reel's soundtrack: an original score on 'Happy Birthday to You' plus frame-locked sound design.

Implements the 'music_and_sound' section of the approved spec (final_edl.json) and the AUD notes of
its timeline entries.

  python -m reel.soundtrack OUT.wav [full|sfx_only]

GRID      3/4 at 90 BPM, beat = 20 frames, bar = 60 frames; sample = frame * 1600 at 48 kHz.
          Bars: pickup f0, f20, f80, f140 (DROP, sample 224000), f200 (laugh), f260, f320 (final).
FORM      phrase 1 in F (music box, past) -> pivot (D7) -> phrases 3 + 4 in G (felt grand + celesta 8va).
PAST      music box + felt piano + E4 pad.  HPF 180 Hz, LPF 3.2 kHz (eases to 4.5 kHz on the pivot and
          back for the suck-in), plate 2.0 s 25 %, tape wow +-8 cents at 0.55 Hz, width 0.2, ~-20 LUFS.
PRESENT   felt grand (one channel of the FluidR3 stereo pair: mono-safe) + celesta 8va, harp waltz chords,
          pizzicato bass, slow strings (DC-blocked), warm pad.
          Hall 2.4 s 20 %; LPF opens 3.2 -> 18 kHz and width 0.2 -> 1.0 over f140-f149; ~-12.5 LUFS.
SFX       needle tick + projector clack f0, settle whoosh, crackle bed, inhale, pucker-driven breath bed,
          riser, harp gliss, glass shimmer f103, odometer ticks, reverse-celesta swells, pre-drop duck,
          snuff + G1 sub f140, bloom whoosh, ember pings, title glint f200 (only with the end card),
          loop reverse swell, end dip.
AMBIENCE  29.mov (HPF/LPF 2.5 kHz, -38 LUFS, f0-f140) crossfading into 30.mov (from B61, natural
          speed, full band, -34 LUFS) over f136-f146.

'Happy Birthday to You' (Mildred & Patty Hill, 1893) is in the public domain; the arrangement, voicings
and sound design here are original.  Instruments are rendered from MIDI by FluidSynth with FluidR3_GM;
everything else is synthesised in numpy.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from . import audio_dsp as a
from . import paths
from .score import hz, midi, pan

SR = a.SR                      # 48 kHz
FPS = 30
SPF = SR // FPS                # 1600 samples per frame
N_FRAMES = 340
N = N_FRAMES * SPF             # 544000 samples = 11.333 s
NR = 13 * SR                   # internal render length (tails), trimmed to N at the end
DROP = 140
PREROLL = 24000                # FluidSynth pre-roll (a multiple of its 64-sample block)
FS_BLOCK = 64                  # an event at sample s first sounds at ceil64(s) + 65 -> schedule one block early

# General MIDI programs (FluidR3_GM)
MUSIC_BOX, PIANO, CELESTA, HARP, PIZZ, SLOW_STRINGS = 10, 0, 8, 46, 45, 49

FELT = (38, 55)                # felt-piano velocity range (spec)

# notes on these frames are hit points: no timing humanisation, onset on the exact frame sample
HIT_FRAMES = {0, 20, 40, 80, 140, 200, 260, 320}


def S(f):
    """frame -> sample index."""
    return int(round(f * SPF))


def T(f):
    """frame -> seconds."""
    return f / FPS


def dbg(x):
    return 10.0 ** (np.asarray(x, np.float64) / 20.0)


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def keys(points, n=NR, log=False, ease=smoothstep):
    """Per-sample automation from [(frame, value), ...]: held before/after, eased between keys."""
    out = np.empty(n)
    fr = np.arange(n) / SPF
    pts = sorted(points)
    vals = [np.log(v) if log else v for _, v in pts]
    out[:] = vals[0]
    for (f0, _), (f1, _), v0, v1 in zip(pts, pts[1:], vals, vals[1:]):
        m = (fr >= f0) & (fr < f1)
        out[m] = v0 + (v1 - v0) * ease((fr[m] - f0) / max(f1 - f0, 1e-9))
    out[fr >= pts[-1][0]] = vals[-1]
    return np.exp(out) if log else out


def rcos(n):
    """Raised-cosine 0 -> 1 ramp of n samples."""
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / max(n, 1))


def fade(x, fin=0, fout=0):
    """Raised-cosine fades (in samples) on both edges of a clip."""
    x = np.array(x, np.float64, copy=True)
    if fin > 0:
        x[:fin] *= rcos(fin)[:, None] if x.ndim == 2 else rcos(fin)
    if fout > 0:
        r = rcos(fout)[::-1]
        x[-fout:] *= r[:, None] if x.ndim == 2 else r
    return x


def st(x):
    x = np.asarray(x, np.float64)
    return np.stack([x, x], -1) if x.ndim == 1 else x


def place(bus, clip, s, gain=1.0):
    """Mix clip into bus starting exactly at sample s."""
    clip = st(clip)
    if s < 0:
        clip, s = clip[-s:], 0
    m = min(len(clip), len(bus) - s)
    if m > 0:
        bus[s:s + m] += clip[:m] * gain
    return bus


def place_end(bus, clip, s_end, gain=1.0):
    """Mix clip into bus so that its last sample lands just before sample s_end."""
    return place(bus, clip, s_end - len(clip), gain)


def peak_to(x, dbfs):
    return x * (dbg(dbfs) / (np.abs(x).max() + 1e-12))


def rms_db(x):
    return 10 * np.log10(np.mean(np.asarray(x) ** 2) + 1e-20)


def lufs(x):
    import pyloudnorm as pyln

    x = st(x)
    if len(x) < int(0.5 * SR):
        x = np.pad(x, ((0, int(0.5 * SR) - len(x)), (0, 0)))
    v = pyln.Meter(SR).integrated_loudness(x)
    return v if np.isfinite(v) else -120.0


def lufs_to(x, target, region=None):
    ref = x if region is None else x[region[0]:region[1]]
    return x * dbg(target - lufs(ref))


def width(x, w):
    """Mid/side width; w scalar or per-sample array (0 = mono, 1 = unchanged)."""
    w = np.asarray(w, np.float64)
    if w.ndim == 1:
        w = w[:len(x)]
    m = (x[:, 0] + x[:, 1]) / 2
    s = (x[:, 0] - x[:, 1]) / 2 * w
    return np.stack([m + s, m - s], -1)


def rng(seed):
    return np.random.default_rng(seed)


def pink(n, ch=2, seed=0):
    w = rng(seed).standard_normal((n, ch))
    W = np.fft.rfft(w, axis=0)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    x = np.fft.irfft(W / np.sqrt(f)[:, None], n, axis=0)
    return x / (x.std() + 1e-12)


# ----------------------------------------------------------------------------------------- filters


def filt(x, f, kind="low", order=2):
    """Zero-phase Butterworth. kind: low | high | band (f = (lo, hi))."""
    if kind == "low" and f >= 0.45 * SR:
        return np.array(x, np.float64)
    sos = signal.butter(order, f, btype=kind, fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def dc_block(x, fc=20.0, order=2):
    """Causal Butterworth high-pass: strips DC / infrasonic drift without pre-ringing, so onsets stay put.
    (FluidR3's Slow Strings samples carry a DC offset: unfiltered it summed to +0.013 FS of DC over f140-f340
    and a DC step at the loop wrap.)"""
    sos = signal.butter(order, fc, btype="high", fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def mono_coherent(x, ch=0):
    """One channel of a FluidSynth stereo sample pair as a mono source.

    FluidR3's grand piano is a spaced stereo pair whose channels are up to ~180 deg apart on many notes
    (L/R correlation down to -0.55 on D5, B4, F#4, E4, C5): the mono fold-down (phone speakers) lost
    4-6.5 dB note by note and the past bus (width 0.2 = mostly mid) combed the felt piano. One channel is a
    complete, phase-coherent piano; the hall/plate IRs (decorrelated per channel) supply the width."""
    return st(np.asarray(x, np.float64)[:, ch])


def _bank_member(x, fc, kind, bw_oct):
    if kind == "band":
        k = 2 ** (bw_oct / 2)
        return filt(x, (max(fc / k, 20.0), min(fc * k, 0.45 * SR)), "band")
    return filt(x, fc, kind)


def morph_filter(x, fc, kind="low", bw_oct=1.0, step_oct=1 / 6, margin_s=0.25):
    """Time-varying zero-phase filter with a per-sample cutoff/centre `fc`.

    A bank of static zero-phase filters at log-spaced frequencies is crossfaded sample by sample. All
    members share zero phase, so the crossfade is seamless (no block seams, no clicks)."""
    fc = np.broadcast_to(np.asarray(fc, np.float64), (len(x),))
    lo, hi = float(fc.min()), float(fc.max())
    if hi / lo < 1.005:
        return _bank_member(x, lo, kind, bw_oct)
    nb = int(np.ceil(np.log2(hi / lo) / step_oct)) + 1
    bank = lo * (hi / lo) ** (np.arange(nb) / (nb - 1))
    pos = np.log(fc / lo) / np.log(hi / lo) * (nb - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, nb - 2)
    fr = pos - i0
    out = np.zeros_like(x, dtype=np.float64)
    m = int(margin_s * SR)
    for b in range(nb):
        w = np.where(i0 == b, 1 - fr, 0.0) + np.where(i0 == b - 1, fr, 0.0)
        nz = np.flatnonzero(w > 1e-7)
        if not len(nz):
            continue
        s0, s1 = max(nz[0] - m, 0), min(nz[-1] + m + 1, len(x))
        y = _bank_member(x[s0:s1], bank[b], kind, bw_oct)
        out[s0:s1] += y * (w[s0:s1, None] if x.ndim == 2 else w[s0:s1])
    return out


# ----------------------------------------------------------------------------------------- reverb / wow


def make_ir(rt60, predelay, hf=0.55, lf=1.15, early=0, seed=0, build=0.012):
    """Stereo, decorrelated, frequency-dependent decay IR (energy-normalised per channel)."""
    r = rng(seed)
    n = int(rt60 * 1.15 * SR)
    t = np.arange(n) / SR
    nz = r.standard_normal((n, 2))
    low = filt(nz, 450, "low")
    high = filt(nz, 3800, "high")
    mid = nz - low - high

    def env(rt):
        return np.exp(-6.91 * t / rt)[:, None]

    tail = low * env(rt60 * lf) + mid * env(rt60) + high * env(rt60 * hf)
    tail *= (1 - np.exp(-t / build))[:, None]            # diffusion build-up, no hard onset
    pre = int(predelay * SR)
    ir = np.zeros((pre + n, 2))
    ir[pre:] = tail
    for k in range(early):                                 # sparse early reflections
        dt = r.uniform(0.004, predelay + 0.045)
        g = 0.55 * np.exp(-dt / 0.05) * r.uniform(0.5, 1.0)
        i = int(dt * SR)
        p = r.uniform(0.15, 0.85)
        ir[i, 0] += g * np.sqrt(1 - p)
        ir[i, 1] += g * np.sqrt(p)
    ir = filt(ir, 9000 if hf < 0.7 else 12000, "low", 1)
    ir /= np.sqrt((ir ** 2).sum(0, keepdims=True))
    return ir


def reverb(x, ir, mix):
    wet = np.stack([signal.fftconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], -1)
    return x * (1 - mix) + wet * mix


PLATE = dict(rt60=2.0, predelay=0.006, hf=0.8, lf=1.0, early=0, seed=21, build=0.006)
HALL = dict(rt60=2.4, predelay=0.024, hf=0.5, lf=1.2, early=10, seed=22, build=0.02)
SFX_ROOM = dict(rt60=1.5, predelay=0.012, hf=0.6, lf=1.0, early=6, seed=23, build=0.01)


def tape_wow(x, cents=8.0, rate=0.55):
    """Pitch wow of +-cents at `rate` Hz via a modulated delay d(t) = A(1 - cos 2*pi*f*t).
    d(0) = 0, so the f0 downbeat is not displaced; the delay never exceeds 2A (~2.7 ms)."""
    n = len(x)
    t = np.arange(n) / SR
    A = (2 ** (cents / 1200) - 1) / (2 * np.pi * rate)
    d = A * (1 - np.cos(2 * np.pi * rate * t)) * SR
    idx = np.arange(n) - d
    base = np.arange(n)
    return np.stack([np.interp(idx, base, x[:, c], left=0.0) for c in range(2)], -1)


# ----------------------------------------------------------------------------------------- MIDI parts


class Part:
    """One GM instrument line. Notes in frames; hit-point notes are exact, others humanised by a few ms."""

    def __init__(self, program, seed, vel_range=(1, 127)):
        self.program = program
        self.vel_range = vel_range   # e.g. the felt piano stays inside vel 38-55 (spec)
        self.r = rng(seed)
        self.notes = []              # [t_s, dur_s, midi, vel]
        self.hits = []               # frames scheduled exactly

    def note(self, f, frames, pitch, vel, ring=0.25, exact=None, jitter=(-3.0, 6.0), vj=2):
        exact = (f in HIT_FRAMES) if exact is None else exact
        t = T(f)
        if exact:
            self.hits.append(f)
        else:
            t += self.r.uniform(*jitter) / 1000.0
            vel += int(self.r.integers(-vj, vj + 1))
        p = midi(pitch) if isinstance(pitch, str) else int(pitch)
        self.notes.append([t, T(frames) + ring, p, int(np.clip(vel, *self.vel_range))])
        return self

    def chord(self, f, frames, pitches, vel, roll_ms=0.0, ring=0.3, exact=None, top_soft=0):
        """Lowest note first; a roll spreads the others upward by roll_ms each (+ 0-2 ms human spread)."""
        exact = (f in HIT_FRAMES) if exact is None else exact
        for i, p in enumerate(pitches):
            off = 0.0 if i == 0 else (i * roll_ms + self.r.uniform(0, 2.0)) / 1000.0
            v = vel - (top_soft * i) // max(len(pitches) - 1, 1)
            if i == 0 and exact:
                self.note(f, frames, p, v, ring=ring, exact=True)
            else:
                self.note(f + off * FPS, frames - off * FPS, p, v, ring=ring, exact=True if off else False,
                          jitter=(-2.0, 2.0))
        return self

    def render(self, n=NR, gain=0.5):
        if not self.notes:
            return np.zeros((n, 2))
        notes = sorted([list(x) for x in self.notes])
        by_pitch = {}
        for x in notes:
            by_pitch.setdefault(x[2], []).append(x)
        for lst in by_pitch.values():               # a repeated pitch must release before it re-strikes
            for p, q in zip(lst, lst[1:]):
                if p[0] + p[1] > q[0] - 0.012:
                    p[1] = max(0.03, q[0] - 0.012 - p[0])
        lead = (PREROLL - FS_BLOCK) / SR
        ev = [(lead + t, d, m, v) for t, d, m, v in notes]
        x = a.render_notes(ev, program=self.program, gain=gain, tail=1.0, sample_format="float").astype(np.float64)
        need = PREROLL + n
        if len(x) < need:
            x = np.pad(x, ((0, need - len(x)), (0, 0)))
        return dc_block(x)[PREROLL:need]


# ----------------------------------------------------------------------------------------- synth pad


def pad_voice(f, n, cutoff, r, detune_c=6.0):
    """Band-limited detuned saw stack (wavetable, so no aliasing), three voices spread L/C/R."""
    L = 4096
    k = np.arange(1, max(2, int(min(cutoff * 4, 12000) / f)) + 1)
    amp = (1 / k) / np.sqrt(1 + (k * f / cutoff) ** 4)
    ph = r.uniform(0, 2 * np.pi, len(k))
    tbl = (amp[:, None] * np.sin(2 * np.pi * k[:, None] * np.arange(L)[None] / L + ph[:, None])).sum(0)
    tbl = np.append(tbl / np.abs(tbl).max(), tbl[0] / np.abs(tbl).max())
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for dc, pl in ((-detune_c, 0.78), (0.0, 0.5), (detune_c, 0.22)):
        lfo = (2 ** (r.uniform(1.0, 2.5) / 1200) - 1) * np.sin(2 * np.pi * r.uniform(0.07, 0.23) * t + r.uniform(0, 6.28))
        inst = f * 2 ** (dc / 1200) * (1 + lfo)
        phase = (r.uniform() + np.cumsum(inst) / SR) % 1.0
        y = np.interp(phase * L, np.arange(L + 1), tbl)
        out[:, 0] += y * np.sqrt(pl)
        out[:, 1] += y * np.sqrt(1 - pl)
    return out / 3


def pad(chords, cutoff, seed, n=NR):
    """chords: [(f_start, f_end, [pitches], attack_s, release_s, gain_db), ...]."""
    out = np.zeros((n, 2))
    r = rng(seed)
    for f0, f1, pitches, att, rel, g in chords:
        s0, s1 = S(f0), S(f1)
        for p in pitches:
            m = min(s1 - s0 + int(rel * SR), n - s0)
            x = pad_voice(hz(midi(p)), m, cutoff, r)
            env = np.ones(m)
            na, nr = int(att * SR), int(rel * SR)
            env[:na] = rcos(na)
            hold = s1 - s0
            if hold < m:
                env[hold:] = rcos(nr)[::-1][:m - hold]
            out[s0:s0 + m] += x * env[:, None] * dbg(g)
    return out


# ----------------------------------------------------------------------------------------- the score


def compose_past():
    """F major memory: music box melody, felt piano harmony, E4 pad, pivot over D7. Dry stems."""
    mb = Part(MUSIC_BOX, 101)
    mb.note(0, 15, "C4", 60)            # 'Hap-'
    mb.note(15, 5, "C4", 50)            # 'py'
    mb.note(20, 20, "D4", 68)           # 'birth'
    mb.note(40, 20, "C4", 60)           # 'day'   (the '29' lands sharp here)
    mb.note(60, 20, "F4", 65)           # 'to'
    mb.note(80, 40, "E4", 66, ring=0.4)  # 'you' - the morph happens inside this hold
    mb.note(120, 15, "D4", 58)          # pivot 'Hap-'
    mb.note(135, 5, "D4", 47, ring=0.15)  # 'py'

    pf = Part(PIANO, 102, FELT)          # felt piano: vel 38-55, low-passed at 3 kHz below
    pf.chord(20, 60, ["F2", "C3", "G3", "A3"], 40, roll_ms=10)               # Fadd9, 30 ms roll
    pf.chord(80, 40, ["C2", "Bb2", "G3"], 42, roll_ms=12)                    # C7sus4 ...
    pf.note(80 + 0.036 * FPS, 20 - 0.036 * FPS, "F3", 41, exact=True, ring=0.05)
    pf.note(100, 20, "E3", 38, ring=0.3)                                      # ... resolving to C7
    pf.chord(120, 20, ["D2", "A2", "C3", "F#3"], 40, roll_ms=10, ring=0.35)   # D7 pivot
    pf.note(120, 15, "D4", 34)                                                # felt doubling of the pickup
    pf.note(135, 5, "D4", 30, ring=0.15)

    gl = Part(HARP, 103)                 # f96-f108: up through C7 chord tones, one frame apart
    gl_notes = ["C4", "E4", "G4", "Bb4", "C5", "E5", "G5", "Bb5", "C6", "E6", "G6", "Bb6", "C7"]
    for i, p in enumerate(gl_notes):
        v = int(38 + 14 * np.sin(np.pi * (i + 1) / (len(gl_notes) + 1)))
        gl.note(96 + i, 30, p, v, exact=True, ring=0.6)

    e4pad = pad([(80, 120, ["E4"], 0.55, 0.35, 0.0), (80, 120, ["E3"], 0.7, 0.35, -9.0)], cutoff=900, seed=104)
    return dict(music_box=mb, piano_past=pf, gliss=gl, e4pad=e4pad)


def compose_present():
    """G major present: felt grand + celesta 8va melody, harp waltz, pizz, slow strings, warm pad."""
    mel = [(140, 20, "D5", 55), (160, 20, "B4", 47), (180, 20, "G4", 45),     # 'BIRTH day dear'
           (200, 20, "F#4", 52), (220, 20, "E4", 44),                         # 'Na- me' (the laugh)
           (240, 15, "C5", 49), (255, 5, "C5", 41),                           # 'Hap- py'
           (260, 20, "B4", 50), (280, 20, "G4", 44), (300, 20, "A4", 46),     # 'birth day to'
           (320, 100, "G4", 44)]                                              # 'you', let ring
    pm = Part(PIANO, 201, FELT)
    ce = Part(CELESTA, 202)
    for f, ln, p, v in mel:
        ring = 0.35 if ln < 100 else 0.5
        pm.note(f, ln, p, v, ring=ring)
        ce.note(f, ln, midi(p) + 12, int(v * 0.92), ring=ring, jitter=(0.0, 7.0))
    # celesta sparkle on the final chord (Gmaj9 tones, after the harp roll)
    for f, p, v in ((325, "F#6", 31), (328, "A6", 28), (331, "D7", 25)):
        ce.note(f, 12, p, v, ring=1.2)

    pc = Part(PIANO, 203, FELT)          # felt left hand
    pc.chord(140, 60, ["G2", "D3", "B3"], 46, roll_ms=12)                      # G (the drop)
    pc.note(180, 20, "F3", 37)                                                 # -> G7
    pc.chord(200, 60, ["C3", "G3", "B3", "E4"], 43, roll_ms=10)                # Cmaj7 (laugh)
    pc.chord(260, 40, ["D3", "G3", "B3", "D4"], 41, roll_ms=10)                # G/D
    pc.chord(300, 20, ["D3", "F#3", "C4"], 39, roll_ms=10)                     # D7
    pc.chord(320, 110, ["G1", "G2", "D3", "F#3", "A3", "B3"], 44, roll_ms=14, top_soft=6)  # Gmaj9

    hp = Part(HARP, 204)                 # waltz chords on beats 2-3 from f160, vel 45
    for f, ch in ((160, ["G3", "B3", "D4"]), (180, ["B3", "D4", "F4"]),
                  (220, ["G3", "B3", "E4"]), (240, ["B3", "E4", "G4"]),
                  (280, ["G3", "B3", "D4"]), (300, ["A3", "C4", "F#4"])):
        hp.chord(f, 20, ch, 45, roll_ms=9, ring=0.7)
    roll = ["G2", "D3", "B3", "F#4", "A4", "D5", "F#5", "A5", "D6", "F#6", "A6"]
    t_ms = 0.0
    for i, p in enumerate(roll):                                               # rolled arpeggio to A6
        hp.note(320 + t_ms / 1000 * FPS, 60, p, int(50 - 12 * i / (len(roll) - 1)), exact=True, ring=1.5)
        t_ms += 34 + 4 * i

    pz = Part(PIZZ, 205)                 # beat 1 from f200
    pz.note(200, 20, "C2", 74, ring=0.6)
    pz.note(260, 20, "D2", 68, ring=0.6)
    pz.note(320, 30, "G1", 72, ring=0.8)

    sg = Part(SLOW_STRINGS, 206)         # enters on G at the drop; common tones are tied
    _tied_chords(sg, [(140, ["G2", "D3", "B3", "D4"]), (180, ["G2", "F3", "B3", "D4"]),
                      (200, ["C3", "G3", "B3", "E4"]), (260, ["D3", "G3", "B3", "D4"]),
                      (300, ["D3", "F#3", "A3", "C4"]), (320, ["G2", "D3", "F#3", "A3", "B3"])],
                 end=430, vel=52)

    warm = pad([(140, 200, ["G3", "B3", "D4"], 0.4, 0.45, 0.0),
                (200, 260, ["G3", "B3", "E4"], 0.3, 0.45, 0.0),
                (260, 300, ["G3", "B3", "D4"], 0.3, 0.4, 0.0),
                (300, 320, ["F#3", "A3", "C4"], 0.25, 0.4, 0.0),
                (320, 430, ["G2", "D3", "A3", "B3"], 0.3, 0.8, 0.0)], cutoff=1400, seed=207)
    return dict(piano_mel=pm, celesta=ce, piano_chd=pc, harp=hp, pizz=pz, strings=sg, pad=warm)


def _tied_chords(part, seq, end, vel):
    """Chord sequence where pitches common to consecutive chords are held rather than re-struck."""
    seq = [(f, [midi(p) for p in ps]) for f, ps in seq] + [(end, [])]
    active = {}
    for (f, ps), (fn, _) in zip(seq, seq[1:]):
        for p in list(active):
            if p not in ps:
                s = active.pop(p)
                part.note(s, f - s, p, vel, ring=0.12, exact=s in HIT_FRAMES)
        for p in ps:
            active.setdefault(p, f)
    for p, s in active.items():
        part.note(s, end - s, p, vel, ring=0.12, exact=s in HIT_FRAMES)


# ----------------------------------------------------------------------------------------- buses


def past_bus(parts):
    mb = parts["music_box"].render()
    pf = filt(mono_coherent(parts["piano_past"].render()), 3000, "low")   # felt
    mb = lufs_to(mb, -18.0)
    pf = lufs_to(pf, -23.5)
    e4 = lufs_to(parts["e4pad"], -27.0)
    x = mb + pf + e4
    x = filt(x, 180, "high")
    x = reverb(x, make_ir(**PLATE), 0.25)
    lp = keys([(120, 3200.0), (134, 4500.0), (140, 3200.0)], log=True)
    x = morph_filter(x, lp, "low")
    x = tape_wow(x, 8.0, 0.55)
    x = width(x, 0.2)
    return lufs_to(x, -20.0, (0, S(134)))


def gliss_bus(gl):
    x = gl.render()
    x = filt(filt(x, 180, "high"), 5000, "low")
    x = reverb(x, make_ir(**PLATE), 0.3)
    x = tape_wow(x, 8.0, 0.55)
    x = width(x, 0.5)
    return peak_to(x, -26.0)


def present_bus(parts):
    st_ = {k: parts[k].render() for k in ("piano_mel", "piano_chd", "celesta", "harp", "pizz", "strings")}
    st_["piano_mel"] = filt(mono_coherent(st_["piano_mel"]), 3000, "low")
    st_["piano_chd"] = filt(mono_coherent(st_["piano_chd"]), 3000, "low")
    targets = dict(piano_mel=-17.0, piano_chd=-23.0, celesta=-23.0, harp=-25.0, pizz=-25.5, strings=-21.5)
    placement = dict(piano_mel=0.0, piano_chd=-0.08, celesta=0.25, harp=-0.35, pizz=-0.05, strings=0.0)
    x = np.zeros((NR, 2))
    for k, v in st_.items():
        y = lufs_to(v, targets[k])
        if k == "piano_mel":                       # felt hammers: tame the attack peaks (~3 dB)
            y = lufs_to(compressor(y, -15.0, 3.0, 6.0, attack=0.002, release=0.12, detect_ms=4.0)[0], targets[k])
        if k == "strings":
            y = y * dbg(keys([(196, 0.0), (232, 3.0), (256, 3.0), (296, 1.0)]))[:, None]   # bar-4 swell
        x += pan(y, placement[k])
    x += lufs_to(parts["pad"], -25.0)
    x = compressor(x, -19.0, 2.0, 6.0, attack=0.012, release=0.25, detect_ms=20.0)[0]   # bus glue
    x = reverb(x, make_ir(**HALL), 0.20)
    x = morph_filter(x, keys([(DROP, 3200.0), (149, 18000.0)], log=True), "low")
    x = width(x, keys([(DROP, 0.2), (149, 1.0)]))
    return lufs_to(x, -12.5, (S(DROP), S(330)))


# ----------------------------------------------------------------------------------------- SFX


def needle_drop(seed=301):
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    r = rng(seed)
    click = r.standard_normal(n) * np.exp(-t / 0.0007)
    click = filt(click, (1500, 7000), "band")
    thump = np.sin(2 * np.pi * 78 * t) * np.exp(-t / 0.022) * 0.55
    scr = np.zeros(n)
    for _ in range(9):                                   # a few stylus scratches as it settles
        i = int(r.uniform(0.004, 0.09) * SR)
        m = int(r.integers(12, 60))
        scr[i:i + m] += r.standard_normal(m) * np.exp(-np.arange(m) / (m / 4)) * r.uniform(0.1, 0.35)
    scr = filt(scr, (1200, 8000), "band")
    x = click + thump + scr
    x = fade(x, 12, int(0.05 * SR))
    return width(st(x) + np.stack([scr * 0.15, -scr * 0.15], -1), 0.6)


def projector_clack(seed=302):
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    r = rng(seed)
    x = np.zeros(n)
    for t0, g, k in ((0.0, 1.0, 1.0), (0.038, 0.45, 1.07)):
        tt = np.clip(t - t0, 0, None)
        on = (t >= t0).astype(float)
        for f, tau, amp in ((1150, 0.009, 1.0), (2480, 0.006, 0.7), (3900, 0.004, 0.5), (620, 0.014, 0.4)):
            x += on * g * amp * np.sin(2 * np.pi * f * k * tt + r.uniform(0, 6.28)) * np.exp(-tt / tau) \
                * np.minimum(tt / 0.0004, 1)
    x = fade(x, 8, int(0.03 * SR))
    return pan(st(x), 0.2)


def settle_whoosh(seed=303):
    n = S(12)
    u = np.arange(n) / n
    fc = 3000 * (900 / 3000) ** (1 - 2 ** (-10 * u))      # easeOutExpo, like the picture settle
    x = morph_filter(pink(n, 2, seed), fc, "band", bw_oct=1.2)
    env = 2 ** (-5.5 * u)
    x *= env[:, None]
    x = width(x, 0.8 - 0.5 * u)
    return fade(x, int(0.008 * SR), int(0.08 * SR))


def crackle_bed(n, seed=304):
    """Vinyl surface: hiss + Poisson crackle in four resonant bands + a once-per-revolution tick."""
    r = rng(seed)
    hiss = filt(pink(n, 2, seed + 1), (400, 7000), "band") * 0.02
    bands = [(1500, 3000), (2500, 4500), (3500, 6500), (5000, 9000)]
    exc = [np.zeros((n, 2)) for _ in bands]
    dur = n / SR
    for rate, lo, hi in ((40, 0.04, 0.12), (8, 0.12, 0.25), (0.6, 0.25, 0.35)):
        for _ in range(r.poisson(rate * dur)):
            i = int(r.integers(0, n - 300))
            if rate < 40 and (S(94) <= i < S(114) or S(132) <= i < S(140)):
                continue                     # no loud pops while the designed sync sounds play
            m = int(r.integers(4, 40))
            b = int(r.integers(0, len(bands)))
            amp = r.uniform(lo, hi)
            p = r.uniform(0.15, 0.85)
            burst = r.standard_normal(m) * np.exp(-np.arange(m) / (m / 3)) * amp
            exc[b][i:i + m, 0] += burst * np.sqrt(1 - p)
            exc[b][i:i + m, 1] += burst * np.sqrt(p)
    x = hiss + sum(filt(e, bd, "band") for e, bd in zip(exc, bands))
    rev = 60 / (100 / 3)                                     # 33 1/3 rpm
    for k in range(int(dur / rev) + 1):                      # soft once-a-revolution tick
        i = int((0.31 + k * rev) * SR)
        if i + 400 < n:
            m = 300
            x[i:i + m] += st(filt(r.standard_normal(m), (600, 2500), "band") * np.exp(-np.arange(m) / 60) * 0.05)
    return filt(x, 9000, "low")


def inhale(seed=305):
    n = S(32) - S(22)
    u = np.arange(n) / n
    x = filt(pink(n + 2000, 2, seed), (600, 3000), "band")[1000:1000 + n]
    x = x * 0.75 + filt(x, (1000, 1700), "band") * 0.6       # slight hollow formant
    env = np.where(u < 0.88, (u / 0.88) ** 1.8, 1.0)
    x *= env[:, None]
    x = width(x, 0.4)
    return fade(x, int(0.01 * SR), int(0.035 * SR))


def pucker_curve():
    tl = json.loads((paths.WORK / "timeline.json").read_text())
    p = np.array([fr.get("pucker") if fr.get("pucker") is not None else np.nan for fr in tl["frames"]], float)
    k = np.arange(len(p))
    good = np.isfinite(p)
    return np.interp(k, k[good], p[good])


def breath_bed(seed=306):
    """Pink noise BP 500-2000 Hz following the per-frame pucker of the frame on screen; it peaks with
    the real flame flare f128-f137."""
    f0, f1 = 32, 137
    s0, s1 = S(f0), S(f1)
    n = s1 - s0
    pk = pucker_curve()
    fr = (s0 + np.arange(n)) / SPF
    p = np.interp(fr, np.arange(len(pk)), pk)
    p = uniform_filter1d(p, int(0.04 * SR), mode="nearest")
    pn = p / p.max()
    emph = keys([(112, 0.6), (126, 1.85), (137, 1.85)], n=s1)[s0:s1]   # flare emphasis
    env = pn ** 0.8 * emph
    env /= env.max()
    r = rng(seed)
    tn = filt(r.standard_normal(n), 7, "low")
    turb = 1 + 0.18 * tn / (tn.std() + 1e-9)
    c, l, rr = pink(n, 1, seed)[:, 0], pink(n, 1, seed + 1)[:, 0], pink(n, 1, seed + 2)[:, 0]
    x = np.stack([0.85 * c + 0.45 * l, 0.85 * c + 0.45 * rr], -1)
    lo = filt(x, (500, 1100), "band")
    hi = filt(x, (1100, 2000), "band")
    wob = 0.5 + 0.5 * np.sin(2 * np.pi * 0.9 * np.arange(n) / SR + 1.0)
    x = lo * (0.8 + 0.2 * wob)[:, None] + hi * (0.8 + 0.2 * (1 - wob))[:, None]
    x *= (env * np.clip(turb, 0.5, 1.6))[:, None]
    x = fade(x, int(0.06 * SR), int(0.033 * SR))
    peak_frame = f0 + int(np.argmax(np.abs(x).max(1))) / SPF
    return peak_to(x, -22.0), peak_frame


def riser(seed=307):
    f0, f1 = 80, 139
    n = S(f1) - S(f0)
    u = np.arange(n) / n
    fc = 300 * (5000 / 300) ** (u ** 1.15)
    x = morph_filter(pink(n, 2, seed), fc, "band", bw_oct=0.9)
    x /= np.sqrt(uniform_filter1d((x ** 2).mean(1), int(0.05 * SR), mode="nearest") + 1e-12)[:, None]  # flat level
    x *= dbg(-18.0 * (1 - u ** 1.6))[:, None]                # -36 -> -18 relative, accelerating
    x = width(x, 0.4 + 0.6 * u)
    x = fade(x, int(0.12 * SR), int(0.004 * SR))             # cut at f139 without a click
    return peak_to(x, -18.0)


def harp_gliss_sfx_only(seed=308):
    """sfx_only stand-in for the harp gliss: 13 inharmonic glass pings, one frame apart (no pitch centre)."""
    out = np.zeros((S(13) + int(1.4 * SR), 2))
    r = rng(seed)
    for i in range(13):
        f = 900 * (4200 / 900) ** (i / 12) * r.uniform(0.98, 1.02)
        m = int(1.2 * SR)
        t = np.arange(m) / SR
        y = sum(g * np.sin(2 * np.pi * f * q * t + r.uniform(0, 6.28)) * np.exp(-t * (4 + 3 * q))
                for q, g in ((1.0, 1.0), (2.76, 0.35), (5.4, 0.12)))
        y = fade(y, int(0.002 * SR), int(0.2 * SR)) * (0.6 + 0.4 * np.sin(np.pi * (i + 1) / 14))
        place(out, pan(st(y), -0.5 + i / 12), S(i))
    out = reverb(out, make_ir(**SFX_ROOM), 0.3)
    return peak_to(out, -26.0)


def glass_shimmer(seed=309, partials=(2093.0, 2637.0, 3136.0, 4186.0), rt=1.8):
    n = int((rt + 0.6) * SR)
    t = np.arange(n) / SR
    r = rng(seed)
    out = np.zeros((n, 2))
    for i, f in enumerate(partials):
        amp = (1.0, 0.8, 0.65, 0.5)[i]
        decay = np.exp(-6.91 * t / (rt * (1 - 0.12 * i)))
        y = sum(np.sin(2 * np.pi * (f + d) * t + r.uniform(0, 6.28)) for d in (-0.8, 0.85)) / 2
        y = y * decay * amp
        out += pan(st(y), -0.6 + 1.2 * i / (len(partials) - 1))
    out = fade(out, int(0.004 * SR), int(0.4 * SR))
    out = reverb(out, make_ir(**SFX_ROOM), 0.25)
    return peak_to(out, -22.0)


def odometer_tick(seed=317):
    """Broadband mechanical click (clears the 2 kHz shimmer partials): 1.5 ms of 3-7 kHz noise + a 900 Hz body."""
    m = int(0.012 * SR)
    t = np.arange(m) / SR
    nz = filt(rng(seed).standard_normal(m + 2000), (3000, 7000), "band")[1000:1000 + m] * np.exp(-t / 0.0015)
    body = np.sin(2 * np.pi * 900 * t) * np.exp(-t / 0.004) * 0.5
    return peak_to(st(fade(nz / (np.abs(nz).max() + 1e-9) + body, 24, int(0.004 * SR))), -26.0)


def reverse_celesta(pitches, n_frames, peak_db, seed, vel=72):
    """Reversed celesta + reverb tail; returns a clip whose last sample is the original attack."""
    p = Part(CELESTA, seed)
    for q in pitches:
        p.note(0, 75, q, vel, exact=True, ring=0.5)
    x = p.render(n=int(4.5 * SR))
    # no predelay / fast build: the tail is fullest right after the attack, so the reversal is a
    # monotonic crescendo that peaks on the hit sample
    x = reverb(x, make_ir(rt60=3.0, predelay=0.0, hf=0.6, lf=1.0, early=0, seed=seed, build=0.003), 0.5)
    rev = x[::-1]
    m = S(n_frames)
    clip = rev[-m:]
    u = np.arange(m) / m
    clip = clip * (u ** 1.2)[:, None]
    clip = fade(clip, int(0.02 * SR), int(0.0015 * SR))
    return peak_to(width(clip, 0.7), peak_db)


def reverse_air(n_frames, peak_db, seed):
    """sfx_only stand-in for the reverse-celesta swells: reversed air burst + reverb (no pitch)."""
    m0 = int(0.06 * SR)
    burst = filt(pink(m0, 2, seed), (900, 8000), "band") * np.hanning(m0)[:, None]
    x = np.zeros((int(3.2 * SR), 2))
    place(x, burst, 0)
    x = reverb(x, make_ir(rt60=2.6, predelay=0.0, hf=0.7, lf=1.0, early=0, seed=seed, build=0.003), 0.75)
    m = S(n_frames)
    clip = x[::-1][-m:] * ((np.arange(m) / m) ** 1.3)[:, None]
    clip = fade(clip, int(0.02 * SR), int(0.0015 * SR))
    return peak_to(clip, peak_db)


def snuff(seed=310):
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    x = filt(pink(n + 4000, 2, seed), (700, 2500), "band")[2000:2000 + n]
    env = np.exp(-t / 0.028) * (1 + 0.25 * np.exp(-((t - 0.022) / 0.01) ** 2))
    x = width(x * env[:, None], 0.3)
    return peak_to(fade(x, int(0.0015 * SR), int(0.02 * SR)), -16.0)


def sub_thump():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = hz(midi("G1")) * (1 + 0.12 * np.exp(-t / 0.012))          # 49 Hz with a short pitch settle
    ph = 2 * np.pi * np.cumsum(f) / SR - 2 * np.pi * f[0] / SR      # phase 0 at the hit sample
    y = np.sin(ph) * np.exp(-t / 0.13)
    y = fade(y, int(0.003 * SR), int(0.09 * SR))
    return peak_to(st(y), -16.0)


def bloom_whoosh(seed=311):
    n = S(165) - S(140)
    u = np.arange(n) / n
    eb = 1 - (1 - np.clip(u * 25 / 16, 0, 1)) ** 3                   # bloom radius, easeOutCubic f140-f156
    fc = 500 * (6000 / 500) ** eb
    x = morph_filter(pink(n, 2, seed), fc, "band", bw_oct=1.3)
    env = np.minimum(u / 0.22, 1) ** 1.5 * (1 - u) ** 1.4
    x *= env[:, None]
    x = width(x, 0.3 + 0.7 * eb)
    return peak_to(fade(x, int(0.01 * SR), int(0.06 * SR)), -28.0)


def ember_pans():
    """Pan each ping to its ember: replays the ember particle RNG of edit.py (fx.Particles, seed 3)."""
    r = rng(3)
    vx = []
    for _ in range(4):
        r.uniform(0.8, 1.0)
        r.integers(1 << 30)
        vx.append(r.uniform(-14, 14))
        r.uniform(45, 65)
        r.uniform(2.0, 3.5)
        r.uniform(0.7, 1.0)
        r.uniform(0, 2 * np.pi)
        r.uniform(0.5, 1.5)
    wick = (585 - 540) / 540
    return [float(np.clip(wick + v / 14 * 0.5, -0.6, 0.6)) for v in vx]


def ember_ping(freq, p):
    m = int(0.08 * SR)
    t = np.arange(m) / SR
    y = np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.018) + 0.12 * np.sin(2 * np.pi * freq * 2.01 * t) * np.exp(-t / 0.008)
    return pan(st(fade(y, int(0.001 * SR), int(0.015 * SR))), p)


def title_glint(seed=312):
    n = int(2.2 * SR)
    t = np.arange(n) / SR
    fc = 3520.0                                                      # A7, ~3.5 kHz
    idx = 1.4 * np.exp(-t / 0.06)
    bell = np.sin(2 * np.pi * fc * t + idx * np.sin(2 * np.pi * fc * 1.41 * t)) * np.exp(-6.91 * t / 1.7)
    bell += 0.18 * np.sin(2 * np.pi * fc * 2.76 * t) * np.exp(-t / 0.08)
    air = filt(pink(n, 2, seed), (5500, 14000), "band")
    aenv = np.minimum(t / 0.02, 1) * np.exp(-t / 0.12)
    x = pan(st(bell), 0.1) + width(air * aenv[:, None], 0.9) * 0.35
    x = fade(x, int(0.0012 * SR), int(0.3 * SR))
    x = reverb(x, make_ir(**SFX_ROOM), 0.2)
    return peak_to(x, -28.0)


def sfx_bus(variant, glint=False):
    """All SFX as separate stems (each NR long). Levels are peak dBFS except the crackle bed (RMS).
    glint: the chime under the end card's 'my love.' (off with the end card)."""
    z = lambda: np.zeros((NR, 2))  # noqa: E731
    out = {}
    tonal = variant == "full"

    x = z()
    place(x, peak_to(needle_drop(), -24.0), 0)
    place(x, peak_to(projector_clack(), -26.0), 0)
    out["needle_clack"] = x

    out["settle_whoosh"] = place(z(), peak_to(settle_whoosh(), -32.0), 0)

    cr = crackle_bed(S(147))
    cr *= dbg(-36.0 - rms_db(cr[:S(140)]))
    if not tonal:
        cr = np.tanh(cr / dbg(-34.0)) * dbg(-34.0)       # soft-cap pops: nothing masks them in this variant
    cr = fade(cr, int(0.002 * SR), 0)
    out["crackle"] = place(z(), cr, 0)

    out["inhale"] = place(z(), peak_to(inhale(), -30.0), S(22))
    br, br_peak = breath_bed()
    out["breath"] = place(z(), br, S(32))
    out["_breath_peak_frame"] = br_peak
    out["riser"] = place(z(), riser(), S(80))

    out["shimmer"] = place(z(), glass_shimmer(), S(103))
    x = z()
    from .titles import Titles

    clicks = Titles((1080, 1920)).odometer_clicks       # the frames the numeral's digits visibly flip
    place(x, pan(odometer_tick(317), 0.05), S(clicks["units"]))
    place(x, pan(odometer_tick(318), -0.05), S(clicks["tens"]))
    out["odometer"] = x

    if tonal:
        rs = reverse_celesta(["D5", "G5", "B5"], DROP - 118, -18.0, 313)
        loop = reverse_celesta(["C5"], N_FRAMES - 326, -26.0, 314, vel=66)
    else:
        rs = reverse_air(DROP - 118, -18.0, 315)
        loop = reverse_air(N_FRAMES - 326, -26.0, 316)
        out["gliss_sub"] = place(z(), harp_gliss_sfx_only(), S(96))
    out["rev_swell"] = place_end(z(), rs, S(DROP))
    out["loop_swell"] = place_end(z(), loop, N)

    x = z()
    place(x, snuff(), S(DROP - 2))                     # the flame collapses on f138: puff, then the drop
    place(x, sub_thump(), S(DROP))
    out["snuff_sub"] = x
    out["bloom_whoosh"] = place(z(), bloom_whoosh(), S(DROP))

    x = z()
    pans = ember_pans()
    sfx_verb = make_ir(**SFX_ROOM)
    for i, (f, fq) in enumerate(zip((149, 153, 157, 160), (6272.0, 5274.0, 5920.0, 6272.0))):   # embers airborne
        place(x, peak_to(ember_ping(fq, pans[i]), -32.0), S(f))
    out["embers"] = peak_to(reverb(x, sfx_verb, 0.18), -32.0)       # -32 dBFS including the room send
    if glint:
        out["glint"] = place(z(), title_glint() * dbg(4.0), S(204))
    return out


# ----------------------------------------------------------------------------------------- ambience


def clip_audio(path):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)


def mirror_extend(x, n):
    """Extend an ambience bed by ping-pong (forward/reversed) concatenation: continuous, no seams."""
    out, fwd = [x], False
    while sum(len(s) for s in out) < n:
        out.append(x if fwd else x[::-1])
        fwd = not fwd
    return np.concatenate(out)[:n]


def xfade_chain(segs, xf):
    """Concatenate uncorrelated noise segments with equal-power crossfades of xf samples."""
    out = segs[0]
    for s in segs[1:]:
        g = np.sin(np.pi / 2 * np.arange(xf) / xf)[:, None]
        mid = out[-xf:] * np.cos(np.pi / 2 * np.arange(xf) / xf)[:, None] + s[:xf] * g
        out = np.concatenate([out[:-xf], mid, s[xf:]])
    return out


def ambience():
    """C (29.mov) bed f0-f146 and B (30.mov) bed from f136; equal-power crossfade f136-f146."""
    c = clip_audio(paths.CLIP_29)
    b = clip_audio(paths.CLIP_30)
    # C: skip AAC priming and the mid-band event after 2.35 s; HPF 150 Hz (wind rumble) + LPF 2.5 kHz
    cbed = mirror_extend(c[int(0.06 * SR):int(2.30 * SR)], S(147) + SR // 2)
    cbed = filt(filt(cbed, 150, "high"), 2500, "low", 4)[:S(147)]     # filtered after mirroring: smooth joints
    cbed = width(cbed, 0.4)
    cbed = lufs_to(cbed, -38.0, (0, S(140)))
    cbed = fade(cbed, int(0.004 * SR), 0)
    # B: from B61 at natural speed, placed at f136; continued with crossfaded B segments (no blow)
    bsrc = filt(b, 30, "high")
    s61 = int(round(61 / 30 * SR))
    main = bsrc[s61:]
    rmain = np.sqrt(np.mean(main[int(0.7 * SR):] ** 2))
    segs = [main]
    pool = [bsrc[int(2.7 * SR):][::-1], bsrc[int(0.08 * SR):int(1.85 * SR)], bsrc[int(2.7 * SR):]]
    k = 0
    while sum(len(s) for s in segs) < NR - S(136) + SR:
        s = pool[k % len(pool)]
        segs.append(s * rmain / (np.sqrt(np.mean(s ** 2)) + 1e-12))
        k += 1
    bbed = xfade_chain(segs, int(0.2 * SR))[:NR - S(136)]
    full_b = place(np.zeros((NR, 2)), bbed, S(136))
    full_b = lufs_to(full_b, -34.0, (S(146), N))
    full_c = place(np.zeros((NR, 2)), cbed, 0)
    u = keys([(136, 0.0), (146, 1.0)], ease=lambda v: np.clip(v, 0, 1))
    full_c *= np.cos(np.pi / 2 * u)[:, None]
    full_b *= np.sin(np.pi / 2 * u)[:, None]
    return full_c, full_b


# ----------------------------------------------------------------------------------------- mix + master


def duck_curve(release_to=None):
    """Pre-drop suck-in: 0 -> -10 dB over f134-f139.5; optionally releases back by `release_to`."""
    pts = [(134, 0.0), (139.5, -10.0)]
    if release_to:
        pts += [(140, -10.0), (release_to, 0.0)]
    return dbg(keys(pts))


def compressor(x, thresh_db, ratio=2.0, knee_db=6.0, attack=0.025, release=0.3, detect_ms=30.0):
    """Feed-forward RMS compressor with a soft knee; returns (y, max gain reduction dB)."""
    p = (x ** 2).mean(1)
    al = np.exp(-1 / (detect_ms / 1000 * SR))
    env = signal.lfilter([1 - al], [1, -al], p)
    lvl = 10 * np.log10(env + 1e-12)
    over = lvl - thresh_db
    gr = np.where(over <= -knee_db / 2, 0.0,
                  np.where(over >= knee_db / 2, (1 / ratio - 1) * over,
                           (1 / ratio - 1) * (over + knee_db / 2) ** 2 / (2 * knee_db)))
    dec = 8
    g = gr[::dec]
    aa, ar = np.exp(-dec / (attack * SR)), np.exp(-dec / (release * SR))
    sm = np.empty_like(g)
    cur = 0.0
    for i, v in enumerate(g):
        cur = aa * cur + (1 - aa) * v if v < cur else ar * cur + (1 - ar) * v
        sm[i] = cur
    gs = np.interp(np.arange(len(x)), np.arange(len(g)) * dec, sm)
    return x * dbg(gs)[:, None], float(sm.min())


def bus_compressor(x, thresh_db=-15.0, ratio=2.0, knee_db=6.0, attack=0.025, release=0.3):
    """Gentle 2:1 master glue."""
    return compressor(x, thresh_db, ratio, knee_db, attack, release, detect_ms=50.0)


def true_peak_db(x):
    up = signal.resample_poly(x, 4, 1, axis=0)
    return 20 * np.log10(np.abs(up).max() + 1e-12)


def tp_limiter(x, ceiling_db, lookahead_ms=2.0, release_ms=60.0):
    up = signal.resample_poly(x, 4, 1, axis=0)
    pk = np.abs(up).max(1)[:4 * len(x)].reshape(-1, 4).max(1)
    need = np.minimum(1.0, dbg(ceiling_db) / np.maximum(pk, 1e-12))
    if need.min() >= 1.0:
        return x
    la = max(int(lookahead_ms * SR / 1000), 1)
    g = minimum_filter1d(need, 2 * la + 1, mode="nearest")
    g = uniform_filter1d(g, la + 1, mode="nearest")      # smooth attack, still <= need at every peak
    k = np.exp(-1 / (release_ms / 1000 * SR))
    out = np.empty_like(g)
    cur = 1.0
    for i, v in enumerate(g):
        cur = v if v < cur else k * cur + (1 - k) * v
        out[i] = cur
    return x * out[:, None]


def master(x, target_lufs, ceiling_db=-1.0):
    for _ in range(4):
        x = x * dbg(target_lufs - lufs(x))
        x = tp_limiter(x, ceiling_db - 0.15)
    tp = true_peak_db(x)
    if tp > ceiling_db - 0.05:
        x = x * dbg(ceiling_db - 0.05 - tp)
    return x, lufs(x), true_peak_db(x)


def render_mix(variant="full", glint=False):
    """Returns (mix NR-long pre-master, stems dict)."""
    if variant not in ("full", "sfx_only"):
        raise ValueError(f"unknown variant {variant!r}")
    stems = {}
    sfx = sfx_bus(variant, glint)
    breath_peak = sfx.pop("_breath_peak_frame")
    amb_c, amb_b = ambience()
    duck = duck_curve()[:, None]
    duck_rel = duck_curve(release_to=144)[:, None]

    if variant == "full":
        past = compose_past()
        pres = compose_present()
        p = past_bus(past) + gliss_bus(past["gliss"])
        p *= duck * keys([(DROP, 1.0), (150, 0.0)])[:, None]          # the memory is snuffed with the flame
        stems["past"] = p
        stems["present"] = present_bus(pres)

    sfx["crackle"] *= duck * keys([(DROP, 1.0), (146, 0.0)])[:, None]
    sfx["riser"] *= duck
    stems["amb_c"] = amb_c * duck
    stems["amb_b"] = amb_b * duck_rel
    stems.update(sfx)
    mix = sum(stems.values())
    stems["_breath_peak_frame"] = breath_peak
    return mix, stems


def build(out_path, variant="full", target_lufs=None, glint=False):
    """Render the soundtrack to a 48 kHz / 24-bit stereo WAV of exactly N samples (340 frames).

    target_lufs overrides the integrated target (default -14 full / -16 sfx_only); glint = end-card chime."""
    mix, stems = render_mix(variant, glint)
    x, gr = bus_compressor(dc_block(mix, 15.0))                       # safety net: no DC / infrasonics
    x = x * dbg(keys([(326, 0.0), (340, -14.0)]))[:, None]            # dip under the reverse swell into the loop
    x = x[:N]
    if target_lufs is None:
        target_lufs = -14.0 if variant == "full" else -16.0
    x, L, tp = master(x, target_lufs, -1.0)
    x = fade(x, 0, 96)                                                 # 2 ms: clean loop wrap under the tick
    L, tp = lufs(x), true_peak_db(x)
    assert len(x) == N, len(x)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a.write_wav(out_path, x.astype(np.float32))
    res = dict(lufs=round(float(L), 2), true_peak_db=round(float(tp), 2), path=str(out_path), samples=len(x),
               variant=variant, comp_max_gr_db=round(gr, 2), breath_peak_frame=round(stems["_breath_peak_frame"], 1))
    if variant == "full":
        res["past_lufs"] = round(lufs(x[:S(134)]), 2)
        res["present_lufs"] = round(lufs(x[S(DROP):S(330)]), 2)
    return res


def envelope(x, hop_s=0.25, anchor_frame=DROP):
    """RMS (dBFS) per hop as [(t0_s, f0, f1, dB)]; the grid is aligned so a window boundary falls on
    `anchor_frame` (the drop), so the pre-drop duck and the drop land in separate windows."""
    h = int(hop_s * SR)
    off = S(anchor_frame) % h
    starts = ([0] if off else []) + list(range(off, len(x), h))
    ends = starts[1:] + [len(x)]
    return [(s0 / SR, s0 / SPF, e0 / SPF, rms_db(x[s0:e0])) for s0, e0 in zip(starts, ends)]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    out = argv[0]
    variant = argv[1] if len(argv) > 1 else "full"
    res = build(out, variant)
    print(json.dumps(res, indent=1))
    import soundfile

    x, _ = soundfile.read(out)
    print("RMS per 0.25 s (grid aligned to the f140 drop):")
    for t, f0, f1, d in envelope(x):
        print(f"{t:6.2f}s f{f0:5.1f}-f{f1:5.1f} {d:7.1f} dBFS " + "#" * max(0, int((d + 60) / 1.5)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
