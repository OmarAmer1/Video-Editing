"""Original score + sound design, composed to picture.

"Happy Birthday to You" (Mildred & Patty Hill, 1893) is in the public domain; the arrangement,
voicings and timings here are original. Instruments are rendered from MIDI with FluidSynth and the
FluidR3 GM soundfont; everything else (risers, shimmer, sub, crackle) is synthesised in numpy.
"""
import numpy as np

from . import audio_dsp as a

# General MIDI programs used
MUSIC_BOX, CELESTA, PIANO, EPIANO, HARP, STRINGS, SLOW_STRINGS, WARM_PAD, CHOIR = 10, 8, 0, 4, 46, 48, 49, 89, 52

NOTE = {n: i for i, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
NOTE.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})


def midi(name):
    """'C4' -> 60, 'Bb3' -> 58."""
    pitch, octave = name[:-1], int(name[-1])
    return 12 * (octave + 1) + NOTE[pitch]


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


class Bus:
    """A named mix bus: notes per GM program are rendered, then a processing chain is applied."""

    def __init__(self, duration):
        self.duration = duration
        self.notes = {}

    def add(self, program, t, dur, note, vel=80):
        n = midi(note) if isinstance(note, str) else note
        self.notes.setdefault(program, []).append((t, dur, n, vel))

    def chord(self, program, t, dur, notes, vel=60, roll=0.0):
        for i, n in enumerate(notes):
            self.add(program, t + i * roll, dur - i * roll, n, vel)

    def render(self, gains=None, tail=4.0):
        out = a.silence(self.duration + tail)
        for prog, ns in self.notes.items():
            g = (gains or {}).get(prog, 1.0)
            x = a.render_notes(ns, program=prog, gain=0.5, tail=tail)
            a.place(out, x, 0, g)
        return out


def lowpass_sweep(x, t0, t1, f0, f1, steps=24):
    """Open (or close) a low-pass filter from f0 to f1 between t0 and t1 (block-wise crossfaded)."""
    out = np.zeros_like(x)
    n = len(x)
    i0, i1 = int(t0 * a.SR), int(t1 * a.SR)
    pre = a.butter(x, f0, "low", 2) if f0 < a.SR / 2 - 200 else x
    post = a.butter(x, f1, "low", 2) if f1 < a.SR / 2 - 200 else x
    out[:i0] = pre[:i0]
    out[i1:] = post[i1:]
    edges = np.linspace(i0, i1, steps + 1).astype(int)
    for k in range(steps):
        fc = f0 * (f1 / f0) ** ((k + 0.5) / steps)
        seg = a.butter(x, min(fc, a.SR / 2 - 200), "low", 2)
        s, e = edges[k], edges[k + 1]
        out[s:e] = seg[s:e]
    # smooth block seams
    return out.astype(np.float32)


def reverse(x):
    return x[::-1].copy()


def reverse_swell(notes_hz, dur=0.65, seed=0):
    """Reversed bell cluster with reverb: rises into a hit point (place so that its END lands on the hit)."""
    n = int(dur * a.SR)
    t = np.arange(int(2.5 * a.SR)) / a.SR
    x = np.zeros(len(t), np.float32)
    for f in notes_hz:
        x += (np.sin(2 * np.pi * f * t) * np.exp(-t * 2.5)).astype(np.float32)
    x = a.reverb(a.to_stereo(x / len(notes_hz)), decay=2.5, mix=0.6, seed=seed)
    r = reverse(x)[-n:]
    env = np.linspace(0, 1, n) ** 2
    return r * env[:, None]


def ember_ping(freq=5500.0, dur=0.12):
    n = int(dur * a.SR)
    t = np.arange(n) / a.SR
    x = np.sin(2 * np.pi * freq * t) * np.exp(-t * 40) * np.minimum(t / 0.001, 1)
    return a.to_stereo(x.astype(np.float32))


def glass_shimmer(partials=(2093.0, 2637.0, 3136.0, 4186.0), dur=1.8, seed=0):
    n = int(dur * a.SR)
    t = np.arange(n) / a.SR
    r = np.random.default_rng(seed)
    out = np.zeros((n, 2), np.float32)
    for i, f in enumerate(partials):
        ph = r.uniform(0, 2 * np.pi)
        x = np.sin(2 * np.pi * f * t + ph) * np.exp(-t * (2.0 + i * 0.6)) * np.minimum(t / 0.003, 1)
        x *= 1 + 0.15 * np.sin(2 * np.pi * r.uniform(3, 7) * t)
        p = (i + 0.5) / len(partials)
        out[:, 0] += x * np.sqrt(1 - p)
        out[:, 1] += x * np.sqrt(p)
    return out / (np.abs(out).max() + 1e-9)


def harp_gliss(t0, notes, step, vel=70):
    return [(t0 + i * step, 1.2, n, vel) for i, n in enumerate(notes)]


def pan(x, p):
    """Constant-power pan, p in [-1, 1]."""
    th = (p + 1) * np.pi / 4
    y = x.copy()
    y[:, 0] *= np.cos(th) * 1.414
    y[:, 1] *= np.sin(th) * 1.414
    return y


def narrow(x, width=0.3):
    m = (x[:, 0] + x[:, 1]) / 2
    s = (x[:, 0] - x[:, 1]) / 2 * width
    return np.stack([m + s, m - s], -1).astype(np.float32)
