"""Audio building blocks: MIDI rendering through FluidSynth, synthesised SFX, reverb, mastering.

All signals are float32 stereo arrays shaped (n, 2) at SR.
"""
import os
import subprocess
import tempfile

import numpy as np
from scipy import signal

SR = 48000
# Debian/Ubuntu install it system-wide (fluid-soundfont-gm); on a Mac scripts/setup.sh puts it in models/
_SF2_SYSTEM = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
SOUNDFONT = os.environ.get("REEL_SF2", _SF2_SYSTEM if os.path.exists(_SF2_SYSTEM) else
                           os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models",
                                        "FluidR3_GM.sf2"))


def silence(dur):
    return np.zeros((int(round(dur * SR)), 2), np.float32)


def to_stereo(x):
    x = np.asarray(x, np.float32)
    return np.stack([x, x], -1) if x.ndim == 1 else x


def place(bus, clip, t, gain=1.0):
    """Mix `clip` into `bus` starting at time t (seconds); grows nothing, clips what overhangs."""
    clip = to_stereo(clip)
    i = int(round(t * SR))
    if i < 0:
        clip, i = clip[-i:], 0
    n = min(len(clip), len(bus) - i)
    if n > 0:
        bus[i:i + n] += clip[:n] * gain
    return bus


def db(x):
    return 10 ** (x / 20)


def env_adsr(n, a=0.005, d=0.1, s=0.7, r=0.2):
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    ns = max(n - na - nd - nr, 0)
    e = np.concatenate([np.linspace(0, 1, na, endpoint=False), np.linspace(1, s, nd, endpoint=False),
                        np.full(ns, s), np.linspace(s, 0, nr)])
    return np.pad(e, (0, max(0, n - len(e))))[:n].astype(np.float32)


def butter(x, cutoff, kind="low", order=4):
    sos = signal.butter(order, cutoff, btype=kind, fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0).astype(np.float32)


def sweep_filter(noise, f0, f1, q=2.0, steps=64):
    """Band-pass a noise signal with a centre frequency gliding exponentially f0 -> f1 (block-wise)."""
    n = len(noise)
    out = np.zeros_like(noise)
    edges = np.linspace(0, n, steps + 1).astype(int)
    win_len = int(SR * 0.04)
    for k in range(steps):
        fc = f0 * (f1 / f0) ** (k / max(steps - 1, 1))
        bw = fc / q
        lo, hi = max(fc - bw / 2, 20), min(fc + bw / 2, SR / 2 - 100)
        sos = signal.butter(2, [lo, hi], btype="band", fs=SR, output="sos")
        a, b = max(edges[k] - win_len, 0), min(edges[k + 1] + win_len, n)
        seg = signal.sosfilt(sos, noise[a:b], axis=0)
        w = np.zeros(b - a)
        w[edges[k] - a:edges[k + 1] - a] = 1
        w = np.convolve(w, np.hanning(win_len) / np.hanning(win_len).sum(), "same")
        out[a:b] += seg * w[:, None]
    return out.astype(np.float32)


def rng(seed):
    return np.random.default_rng(seed)


# ----------------------------------------------------------------------------------------- SFX


def whoosh(dur=0.8, f0=300, f1=4000, seed=1, peak_at=0.6):
    n = int(dur * SR)
    noise = rng(seed).standard_normal((n, 2)).astype(np.float32)
    x = sweep_filter(noise, f0, f1, q=1.5)
    t = np.linspace(0, 1, n)
    e = np.where(t < peak_at, (t / peak_at) ** 2.2, np.exp(-(t - peak_at) / (1 - peak_at) * 5))
    x *= e[:, None]
    # gentle stereo movement
    pan = 0.5 + 0.35 * np.sin(np.pi * t)
    x[:, 0] *= np.sqrt(1 - pan) * 1.41
    x[:, 1] *= np.sqrt(pan) * 1.41
    return x / (np.abs(x).max() + 1e-9)


def riser(dur=1.5, f0=200, f1=3000, seed=2):
    n = int(dur * SR)
    t = np.linspace(0, 1, n)
    noise = rng(seed).standard_normal((n, 2)).astype(np.float32)
    x = sweep_filter(noise, f0, f1, q=3) * 0.7
    ph = 2 * np.pi * np.cumsum(f0 * 0.5 * (f1 / f0) ** t) / SR
    tone = (np.sin(ph) + 0.3 * np.sin(2 * ph)).astype(np.float32)
    x += to_stereo(tone) * 0.15
    x *= (t ** 2.5)[:, None]
    return x / (np.abs(x).max() + 1e-9)


def sub_drop(dur=1.2, f0=90, f1=32):
    n = int(dur * SR)
    t = np.linspace(0, dur, n)
    f = f1 + (f0 - f1) * np.exp(-t * 4)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-t * 2.2) * np.minimum(t / 0.004, 1)
    return to_stereo(x.astype(np.float32))


def shimmer(dur=2.0, base=1568.0, count=28, seed=3, spread=1.0):
    """Glittering high bell partials with random onsets (fairy-dust / sparkle bed)."""
    n = int(dur * SR)
    out = np.zeros((n, 2), np.float32)
    r = rng(seed)
    ratios = np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6]) * (base / 2)
    for _ in range(count):
        f = r.choice(ratios) * r.uniform(0.995, 1.005)
        t0 = r.uniform(0, dur * 0.8 * spread)
        ln = min(r.uniform(0.25, 0.9), dur - t0)
        m = int(ln * SR)
        tt = np.arange(m) / SR
        tone = np.sin(2 * np.pi * f * tt) * np.exp(-tt * r.uniform(5, 10)) * np.minimum(tt / 0.002, 1)
        tone += 0.25 * np.sin(2 * np.pi * f * 2.76 * tt) * np.exp(-tt * 18)
        p = r.uniform(0.15, 0.85)
        i = int(t0 * SR)
        out[i:i + m, 0] += tone * np.sqrt(1 - p) * r.uniform(0.4, 1)
        out[i:i + m, 1] += tone * np.sqrt(p) * r.uniform(0.4, 1)
    return out / (np.abs(out).max() + 1e-9)


def chime(freq=2093.0, dur=2.5):
    """Single clean bell (FM) – a 'ding' for a sparkle accent."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    mod = np.sin(2 * np.pi * freq * 3.5 * t) * 2.0 * np.exp(-t * 6)
    x = np.sin(2 * np.pi * freq * t + mod) * np.exp(-t * 2.5) * np.minimum(t / 0.001, 1)
    x += 0.3 * np.sin(2 * np.pi * freq * 2.01 * t) * np.exp(-t * 5)
    return to_stereo(x.astype(np.float32) / 1.3)


def breath(dur=1.0, seed=4):
    """Soft blowing / breath noise (for the candle blow)."""
    n = int(dur * SR)
    x = rng(seed).standard_normal((n, 2)).astype(np.float32)
    x = butter(butter(x, 350, "high", 2), 2500, "low", 2)
    t = np.linspace(0, 1, n)
    e = np.minimum(t / 0.15, 1) * np.exp(-np.maximum(t - 0.5, 0) * 5)
    e *= 1 + 0.25 * np.sin(2 * np.pi * 9 * t)
    return x * e[:, None] / (np.abs(x * e[:, None]).max() + 1e-9)


def flame_out(seed=5):
    """Tiny 'fffp' puff when a flame dies: short filtered noise + low thump."""
    a = breath(0.35, seed) * 0.8
    b = sub_drop(0.3, 140, 60) * 0.25
    return a + b


def vinyl_crackle(dur, density=18, seed=6, hiss=0.04):
    n = int(dur * SR)
    r = rng(seed)
    x = r.standard_normal((n, 2)).astype(np.float32) * hiss
    x = butter(x, 1200, "high", 2) + butter(x, 6000, "low", 2) * 0.3
    for _ in range(int(density * dur)):
        i = r.integers(0, n - 200)
        L = r.integers(20, 160)
        click = r.standard_normal(L) * np.exp(-np.arange(L) / (L / 4)) * r.uniform(0.2, 1)
        x[i:i + L, r.integers(0, 2)] += click
    return x


def projector(dur, fps=24.0, seed=7):
    """Low 8 mm projector whirr/flutter (for the B&W 'memory' section)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    clack = (np.sin(2 * np.pi * fps * t) > 0.95).astype(np.float32)
    clack = signal.lfilter([1], [1, -0.995], clack) * 0.02
    noise = butter(rng(seed).standard_normal(n).astype(np.float32), 900, "low", 2) * 0.05
    motor = 0.03 * np.sin(2 * np.pi * 110 * t) * (1 + 0.3 * np.sin(2 * np.pi * fps * t))
    return to_stereo((clack + noise + motor).astype(np.float32))


def impact(dur=1.5, seed=8):
    """Cinematic soft hit: sub thump + noise transient + short tail."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    thump = sub_drop(dur, 110, 38)[:, 0]
    nz = rng(seed).standard_normal(n).astype(np.float32)
    nz = butter(nz, 3000, "low", 2) * np.exp(-t * 18)
    return to_stereo((thump + 0.35 * nz).astype(np.float32))


# ----------------------------------------------------------------------------------------- music


def render_notes(notes, program=0, gain=0.5, tail=3.0, channel=0, bank=0, reverb=False, sample_format=None):
    """Render [(start_s, dur_s, midi_note, velocity), ...] with one GM program through FluidSynth.

    sample_format: FluidSynth file format ('s16' is its default; 'float' avoids 16-bit truncation of quiet
    renders that are gained up later)."""
    import mido

    tpb = 960
    tempo = 500000  # 120 bpm -> 1 s == 2 beats == 1920 ticks
    mid = mido.MidiFile(ticks_per_beat=tpb)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    if bank:
        tr.append(mido.Message("control_change", channel=channel, control=0, value=bank, time=0))
    tr.append(mido.Message("program_change", channel=channel, program=program, time=0))
    tr.append(mido.Message("control_change", channel=channel, control=91, value=0, time=0))  # fluid reverb off
    tr.append(mido.Message("control_change", channel=channel, control=93, value=0, time=0))
    ev = []
    for s, d, note, vel in notes:
        ev.append((s, 1, note, vel))
        ev.append((s + d, 0, note, 0))
    ev.sort(key=lambda e: (e[0], e[1]))
    last = 0
    for t, on, note, vel in ev:
        tick = int(round(t * tpb * 2))
        msg = "note_on" if on else "note_off"
        tr.append(mido.Message(msg, channel=channel, note=int(note), velocity=int(vel), time=max(tick - last, 0)))
        last = tick
    end = max(e[0] for e in ev) + tail
    tr.append(mido.MetaMessage("end_of_track", time=int(round((end - last / (tpb * 2)) * tpb * 2))))
    with tempfile.TemporaryDirectory() as d:
        mp, wp = os.path.join(d, "m.mid"), os.path.join(d, "o.wav")
        mid.save(mp)
        fmt = ["-O", sample_format] if sample_format else []
        subprocess.run(["fluidsynth", "-ni", "-q", "-g", str(gain), "-r", str(SR), "-R", "1" if reverb else "0",
                        "-C", "0", *fmt, "-F", wp, SOUNDFONT, mp], check=True, capture_output=True)
        import soundfile

        x, sr = soundfile.read(wp, dtype="float32")
    assert sr == SR
    return to_stereo(x)


def synth_pad(freqs, dur, attack=0.8, release=1.2, detune=0.12, cutoff=2200, seed=9):
    """Warm detuned-saw pad chord, low-passed."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2), np.float32)
    r = rng(seed)
    for f in freqs:
        for k, dt in enumerate((-detune, 0, detune)):
            ff = f * 2 ** (dt / 12)
            ph = r.uniform(0, 1)
            saw = 2 * ((ff * t + ph) % 1) - 1
            side = k % 2
            out[:, side] += saw * 0.6
            out[:, 1 - side] += saw * 0.4
    out = butter(out, cutoff, "low", 2)
    e = env_adsr(n, attack, 0.2, 0.9, release)
    return (out * e[:, None] / (len(freqs) * 3)).astype(np.float32)


# ----------------------------------------------------------------------------------------- fx & master


def reverb(x, decay=2.2, mix=0.25, predelay=0.02, seed=10, damp=6000):
    """Stereo convolution reverb with a synthetic exponentially-decaying noise IR."""
    n = int(decay * SR)
    r = rng(seed)
    t = np.arange(n) / SR
    ir = r.standard_normal((n, 2)).astype(np.float32) * np.exp(-t * 6.9 / decay)[:, None]
    ir = butter(ir, damp, "low", 1)
    ir = np.concatenate([np.zeros((int(predelay * SR), 2), np.float32), ir])
    ir /= np.sqrt((ir ** 2).sum(0, keepdims=True))
    wet = np.stack([signal.fftconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], -1)
    return (x * (1 - mix) + wet * mix).astype(np.float32)


def tape_wobble(x, depth=0.0025, rate=0.6, seed=11):
    """Subtle wow/flutter pitch wobble (nostalgic music-box / vinyl feel)."""
    n = len(x)
    t = np.arange(n) / SR
    r = rng(seed)
    mod = depth * (np.sin(2 * np.pi * rate * t) + 0.4 * np.sin(2 * np.pi * rate * 3.1 * t + r.uniform(0, 6)))
    idx = np.clip(np.arange(n) + np.cumsum(mod) * 0 + mod * SR * 0.01, 0, n - 1)
    return np.stack([np.interp(idx, np.arange(n), x[:, c]) for c in range(2)], -1).astype(np.float32)


def sidechain(x, hits, depth=0.5, release=0.25):
    """Duck x at the given hit times (seconds)."""
    g = np.ones(len(x), np.float32)
    rel = int(release * SR)
    curve = 1 - depth * np.exp(-np.arange(rel) / (rel / 4))
    for h in hits:
        i = int(h * SR)
        if i < len(g):
            m = min(rel, len(g) - i)
            g[i:i + m] = np.minimum(g[i:i + m], curve[:m])
    return x * g[:, None]


def fade(x, fin=0.0, fout=0.0):
    x = x.copy()
    if fin > 0:
        m = int(fin * SR)
        x[:m] *= np.linspace(0, 1, m)[:, None] ** 2
    if fout > 0:
        m = int(fout * SR)
        x[-m:] *= np.linspace(1, 0, m)[:, None] ** 2
    return x


def true_peak(x):
    up = signal.resample_poly(x, 4, 1, axis=0)
    return float(np.abs(up).max())


def limiter(x, ceiling_db=-1.0, lookahead=0.005, release=0.08):
    ceiling = db(ceiling_db)
    la = int(lookahead * SR)
    peak = np.abs(signal.resample_poly(x, 4, 1, axis=0)).max(1).reshape(-1, 4).max(1)[:len(x)]
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    # look-ahead min filter then smooth release
    from scipy.ndimage import minimum_filter1d

    g = minimum_filter1d(need, size=2 * la + 1, origin=0)
    g = np.concatenate([g[la:], np.full(la, g[-1])])
    a = np.exp(-1 / (release * SR))
    out = np.empty_like(g)
    cur = 1.0
    for i, v in enumerate(g):
        cur = v if v < cur else a * cur + (1 - a) * v
        out[i] = cur
    return (x * out[:, None]).astype(np.float32)


def master(x, lufs=-14.0, ceiling_db=-1.0):
    import pyloudnorm as pyln

    meter = pyln.Meter(SR)
    for _ in range(3):
        loud = meter.integrated_loudness(x.astype(np.float64))
        x = x * db(lufs - loud)
        x = limiter(x, ceiling_db - 0.2)
    return x.astype(np.float32), meter.integrated_loudness(x.astype(np.float64)), 20 * np.log10(true_peak(x))


def write_wav(path, x):
    import soundfile

    soundfile.write(str(path), x, SR, subtype="PCM_24")
