"""Shared synth for the reels' soundtracks: deterministic instruments, a stereo bus with a reverb send and a
kick sidechain, and the mixdown. Call init(duration), place instruments, then mixdown(path)."""
import wave

import numpy as np

SR = 48000
BPM = 120
BEAT = 60 / BPM
DUR = 30.0
N = 0
L = R = send = side = None
rng = np.random.default_rng(7)


def init(dur, bpm=120, seed=7):
    """Fresh buffers for a soundtrack of `dur` seconds (plus one second of tail room)."""
    global DUR, N, L, R, send, side, rng, BPM, BEAT
    DUR, BPM, BEAT = dur, bpm, 60 / bpm
    N = int(dur * SR) + SR
    L, R = np.zeros(N), np.zeros(N)
    send, side = np.zeros(N), np.zeros(N)  # mono reverb send, kick sidechain envelope
    rng = np.random.default_rng(seed)


def t_axis(dur):
    return np.arange(int(dur * SR)) / SR


def place(buf, sig, at, gain=1.0, pan=0.0, rev=0.0):
    i = int(at * SR)
    if i >= N or i + len(sig) <= 0:
        return
    s = sig[: N - i] * gain
    lg, rg = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    L[i:i + len(s)] += s * lg * 1.41
    R[i:i + len(s)] += s * rg * 1.41
    if rev:
        send[i:i + len(s)] += s * rev


def fft_filter(x, lo=0.0, hi=None, slope=1.0):
    """Gentle band-pass by spectral shaping (fine for one-shots and stems)."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    h = np.ones_like(f)
    if lo > 0:
        h *= 1 / np.sqrt(1 + (lo / np.maximum(f, 1)) ** (4 * slope))
    if hi:
        h *= 1 / np.sqrt(1 + (f / hi) ** (4 * slope))
    return np.fft.irfft(X * h, n)


def env(dur, a=0.002, d=0.2, curve=1.0):
    t = t_axis(dur)
    e = np.minimum(1, t / max(a, 1e-4)) * np.exp(-np.maximum(0, t - a) / d)
    return e ** curve


def noise(dur):
    return rng.uniform(-1, 1, int(dur * SR))


def saw(freq, dur, detune=0.0):
    t = t_axis(dur)
    ph = (t * freq * (1 + detune)) % 1
    return 2 * ph - 1


def midi(n):
    return 440 * 2 ** ((n - 69) / 12)


# ================================================================ instruments
def kick(at, gain=1.0):
    t = t_axis(0.5)
    f = 42 + 110 * np.exp(-t / 0.035)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(0.5, 0.001, 0.16)
    click = fft_filter(noise(0.01), 2000, 8000) * env(0.01, 0.0005, 0.003)
    s[: len(click)] += click * 0.4
    place(None, np.tanh(s * 1.6) * 0.95, at, gain)
    i = int(at * SR)
    e = env(0.35, 0.001, 0.09)
    side[i:i + len(e)] = np.maximum(side[i:i + len(e)], e[: max(0, min(len(e), N - i))] * gain)


def snare(at, gain=0.5):
    n = fft_filter(noise(0.3), 900, 9000) * env(0.3, 0.001, 0.07)
    t = t_axis(0.3)
    body = np.sin(2 * np.pi * 190 * t) * env(0.3, 0.001, 0.04)
    place(None, n + body * 0.6, at, gain, 0.0, rev=0.35)


def clap(at, gain=0.4):
    s = np.zeros(int(0.35 * SR))
    for k, o in enumerate([0, 0.011, 0.022]):
        b = fft_filter(noise(0.3), 1100, 6000) * env(0.3, 0.001, 0.02 if k < 2 else 0.12)
        i = int(o * SR)
        s[i:i + len(b)] += b[: len(s) - i]
    place(None, s, at, gain, 0.1, rev=0.4)


def hat(at, gain=0.12, open_=False, pan=0.25):
    d = 0.25 if open_ else 0.045
    s = fft_filter(noise(d), 7000, None) * env(d, 0.0005, d / 3)
    place(None, s, at, gain, pan)


def bass(at, note, dur, gain=0.35, cutoff=500):
    f = midi(note)
    s = saw(f, dur) + 0.6 * np.sin(2 * np.pi * f / 2 * t_axis(dur))
    s = fft_filter(s, 30, cutoff) * env(dur, 0.005, dur * 0.8)
    s[-200:] *= np.linspace(1, 0, 200)
    place(None, np.tanh(s * 1.5), at, gain)


def pad(at, notes, dur, gain=0.08, cutoff=1800):
    s = np.zeros(int(dur * SR))
    for n_ in notes:
        for dt in (-0.006, 0.0, 0.007):
            s += saw(midi(n_), dur, dt)
    s = fft_filter(s, 120, cutoff)
    t = t_axis(dur)
    s *= np.minimum(1, t / 0.4) * np.minimum(1, (dur - t) / 0.4)
    place(None, s, at, gain, rev=0.25)


def pluck(at, note, gain=0.12, pan=0.0):
    f = midi(note)
    t = t_axis(0.35)
    s = (saw(f, 0.35) * 0.5 + np.sin(2 * np.pi * f * t)) * env(0.35, 0.001, 0.08)
    place(None, fft_filter(s, 200, 5000), at, gain, pan, rev=0.3)


def bell(at, base=880, gain=0.25, pan=0.0, dec=0.6):
    t = t_axis(dec * 3)
    s = sum(a * np.sin(2 * np.pi * base * r * t) * np.exp(-t / (dec / r ** 0.5))
            for r, a in [(1, 1), (2.76, 0.5), (5.4, 0.25), (8.93, 0.12)])
    place(None, s * np.minimum(1, t / 0.002), at, gain, pan, rev=0.5)


def impact(at, gain=1.0):
    t = t_axis(2.5)
    boom = np.sin(2 * np.pi * np.cumsum(30 + 70 * np.exp(-t / 0.08)) / SR) * np.exp(-t / 0.7)
    crack = fft_filter(noise(2.5), 300, 9000) * np.exp(-t / 0.25)
    place(None, np.tanh(boom * 2.2) * 0.9 + crack * 0.45, at, gain, rev=0.6)
    i = int(at * SR)
    e = env(0.6, 0.001, 0.25)
    side[i:i + len(e)] = np.maximum(side[i:i + len(e)], e[: max(0, min(len(e), N - i))])


def riser(at, dur, gain=0.3, f0=300, f1=6000):
    t = t_axis(dur)
    k = t / dur
    n = noise(dur)
    out = np.zeros_like(n)
    # sweep the band in slices
    seg = int(0.05 * SR)
    for i in range(0, len(n), seg):
        c = f0 * (f1 / f0) ** k[i]
        out[i:i + seg] = fft_filter(n[i:i + seg + 0], c * 0.6, c * 1.6)[: len(out[i:i + seg])]
    tone = np.sin(2 * np.pi * np.cumsum(110 * 2 ** (3 * k)) / SR) * 0.25
    place(None, (out + tone) * k ** 2.2, at, gain, rev=0.3)


def whoosh(at, dur=0.5, gain=0.35, reverse=False):
    t = t_axis(dur)
    k = t / dur
    shape = np.sin(np.pi * k) ** 2
    s = fft_filter(noise(dur), 400, 5000) * shape
    if reverse:
        s = s * (k ** 2)
    place(None, s, at, gain, rev=0.3)


def key_click(at, i):
    d = 0.03
    s = fft_filter(noise(d), 2500, 7000) * env(d, 0.0003, 0.006)
    t = t_axis(d)
    s += np.sin(2 * np.pi * (1800 + 300 * (i % 3)) * t) * env(d, 0.0003, 0.004) * 0.4
    place(None, s, at, 0.32, pan=-0.2 + 0.4 * ((i * 7) % 5) / 4)


def pop(at, f=700, gain=0.3, pan=0.0):
    t = t_axis(0.12)
    s = np.sin(2 * np.pi * np.cumsum(f * (1 + 0.6 * np.exp(-t / 0.015))) / SR) * env(0.12, 0.001, 0.03)
    place(None, s, at, gain, pan, rev=0.15)


def tap(at, gain=0.5):
    t = t_axis(0.15)
    s = np.sin(2 * np.pi * np.cumsum(260 * (1 + 1.5 * np.exp(-t / 0.01))) / SR) * env(0.15, 0.0005, 0.035)
    place(None, s, at, gain)
    hat(at, 0.08)




# ================================================================ mix
def mixdown(path):
    # sidechain the whole bed (not the kick) a little: pump
    pump = 1 - 0.45 * np.clip(side, 0, 1)
    # reverb: exponentially decaying noise IR, decorrelated per side
    ir_len = int(1.8 * SR)
    tt = np.arange(ir_len) / SR
    irL = rng.normal(0, 1, ir_len) * np.exp(-tt / 0.45)
    irR = rng.normal(0, 1, ir_len) * np.exp(-tt / 0.45)
    irL[: int(0.012 * SR)] = 0
    irR[: int(0.017 * SR)] = 0
    nfft = 1 << int(np.ceil(np.log2(N + ir_len)))
    S = np.fft.rfft(send, nfft)
    revL = np.fft.irfft(S * np.fft.rfft(irL, nfft), nfft)[:N]
    revR = np.fft.irfft(S * np.fft.rfft(irR, nfft), nfft)[:N]
    revL = fft_filter(revL, 200, 7000)
    revR = fft_filter(revR, 200, 7000)
    rs = 0.9 / max(np.max(np.abs(revL)), np.max(np.abs(revR)), 1e-9) * max(np.max(np.abs(send)), 1e-9)
    mixL = L * pump ** 0.5 + revL * rs * 0.35
    mixR = R * pump ** 0.5 + revR * rs * 0.35
    # fade out the last half second, trim to the video
    n = int(DUR * SR)
    mixL, mixR = mixL[:n], mixR[:n]
    fade = np.ones(n)
    fade[-int(0.5 * SR):] = np.linspace(1, 0, int(0.5 * SR)) ** 2
    mixL *= fade
    mixR *= fade
    # glue: soft clip, normalize to -1 dBFS
    peak = max(np.max(np.abs(mixL)), np.max(np.abs(mixR)))
    mixL, mixR = np.tanh(mixL / peak * 1.4), np.tanh(mixR / peak * 1.4)
    g = 10 ** (-1 / 20) / max(np.max(np.abs(mixL)), np.max(np.abs(mixR)))
    out = np.stack([mixL * g, mixR * g], 1)
    pcm = (out * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print('wrote', path, f'{n / SR:.2f}s')

