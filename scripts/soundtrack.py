"""Soundtrack for the NVIDIA reel: synthesized from scratch, locked to the reel's 120 BPM grid and cue times.

    python3 scripts/soundtrack.py out.wav

Cue times (milestone crossings, typing, taps) mirror the constants in index.html.
"""
import sys
import wave

import numpy as np

SR = 48000
BPM = 120
BEAT = 60 / BPM
DUR = 30.0
N = int(DUR * SR) + SR  # one second of tail room, trimmed at the end
rng = np.random.default_rng(7)

# ---- cue times from index.html
MILESTONES = [7.780, 10.073, 11.750, 11.968, 13.674]  # first $1, $10, $50, $100, $200
EVENTS = [3.422, 5.906, 6.307, 9.976, 11.063, 11.475]  # context bands stamping in
T_TYPE, T_KEY, N_KEYS = 21.0, 0.125, 13
T_RES, T_TAP1, T_PROF, T_TAP2, T_END = 22.75, 23.5, 24.0, 26.5, 27.5

L = np.zeros(N)
R = np.zeros(N)
send = np.zeros(N)  # mono reverb send
side = np.zeros(N)  # sidechain envelope (kick)


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


# ================================================================ arrangement
CHORDS = [(57, [57, 60, 64]), (53, [53, 57, 60]), (48, [55, 60, 64]), (55, [55, 59, 62])]  # Am F C G
bar = 4 * BEAT  # one chord per bar (2 s)

# act 1 — drone, slams, riser into the chart
pad(0.0, [45, 52, 57], 3.2, gain=0.07, cutoff=900)
riser(0.0, 0.5, gain=0.25, f0=200, f1=3000)
impact(0.5, 1.0)
for at in (1.0, 1.5):
    kick(at, 1.0)
    bell(at, 220, 0.18)
    whoosh(at - 0.18, 0.2, 0.15, reverse=True)
for i in range(4):
    pluck(2.0 + i * 0.125, [69, 72, 76, 81][i], 0.14, pan=-0.3 + 0.2 * i)
riser(2.0, 1.0, gain=0.3)

# act 2 — the climb (3.0 → 16.5), building
t = 3.0
while t < 16.5 - 1e-6:
    b = round((t - 3.0) / BEAT)  # beat index since 3.0
    ci = int(((t - 3.0) // bar) % 4)
    root, chord = CHORDS[ci]
    level = 0 if t < 7 else 1 if t < 11 else 2
    kick(t, 0.9)
    if level >= 1 and b % 2 == 1:
        snare(t, 0.45)
    if level >= 2 and b % 2 == 1:
        clap(t, 0.3)
    # hats: offbeat 8ths, then 16ths
    hat(t + BEAT / 2, 0.11, open_=(level >= 1 and b % 4 == 3))
    if level >= 1:
        hat(t + BEAT / 4, 0.05, pan=-0.3)
        hat(t + 3 * BEAT / 4, 0.05, pan=-0.3)
    # bass on 8ths, filter opening with the climb
    cut = 250 + 1400 * (t - 3.0) / 13.5
    for k in range(2):
        bass(t + k * BEAT / 2, root - 24 + (12 if k == 1 and level >= 1 else 0), BEAT / 2 * 0.9, 0.32, cut)
    # arp in the last section
    if level >= 2:
        for k in range(4):
            pluck(t + k * BEAT / 4, chord[(b * 4 + k) % 3] + 12 + (12 if k == 3 else 0), 0.09, pan=0.4 * ((k % 2) * 2 - 1))
    t += BEAT
for i in range(7):  # one pad per chord, brightening
    at = 3.0 + i * bar
    root, chord = CHORDS[i % 4]
    pad(at, chord, min(bar, 16.5 - at) + 0.3, 0.06, 800 + 350 * i)
for at in EVENTS:
    bell(at, 1320, 0.07, pan=0.5, dec=0.25)
for k, at in enumerate(MILESTONES):  # every new milestone rings higher
    bell(at, 660 * 2 ** (k / 6), 0.3, pan=0.0, dec=0.8)
    kick(at, 0.6)
riser(14.5, 2.5, gain=0.45, f0=250, f1=9000)

# act 3 — the payoff
impact(17.0, 1.15)
hat(17.0, 0.25, open_=True)
pad(17.0, [45, 57, 60, 64, 69], 3.0, 0.09, 2500)
bass(17.0, 33, 2.0, 0.4, 300)
for k, at in enumerate((17.5, 18.0, 18.5)):
    kick(at, 0.9)
    bell(at, [440, 523.25, 659.25][k], 0.22)
    snare(at, 0.25)
for k in range(8):  # the ×2,389 counter ticking up
    pop(17.5 + k * 0.1, 900 + k * 120, 0.08, pan=0.3)
whoosh(19.4, 0.6, 0.45, reverse=True)

# act 4 — the search, the profile, the end card
PAD4 = [(57, [57, 60, 64]), (53, [53, 57, 60]), (48, [55, 60, 64]), (55, [55, 59, 62])]
for i in range(5):
    root, chord = PAD4[i % 4]
    pad(20.0 + i * 2.0, chord + [chord[0] + 12], 2.0 + (0.5 if i == 4 else 0.0), 0.055, 1400 + 200 * i)
t = 20.0
while t < T_END - 1e-6:
    b = round((t - 20.0) / BEAT)
    if b % 4 in (0, 2):
        kick(t, 0.55)
    if b % 4 == 3:
        kick(t + BEAT / 2, 0.45)
    if b % 2 == 1:
        snare(t, 0.18)
    hat(t + BEAT / 2, 0.06)
    if t >= T_PROF:  # the groove lifts once the profile is on screen
        hat(t + BEAT / 4, 0.035, pan=-0.3)
        hat(t + 3 * BEAT / 4, 0.035, pan=-0.3)
    root = PAD4[int((t - 20.0) // 2.0) % 4][0]
    bass(t, root - 24, BEAT * 0.8, 0.22, 400 + (300 if t >= T_PROF else 0))
    t += BEAT
pop(20.0, 500, 0.25)  # the bar grows out of the spark
whoosh(20.0, 0.5, 0.2)
for i in range(5):  # headline words
    pluck(20.25 + i * 0.125, [76, 79, 81, 84, 88][i], 0.07, pan=-0.4 + 0.2 * i)
for i in range(N_KEYS):
    key_click(T_TYPE + i * T_KEY, i)
for i in range(4):
    pop(T_RES + i * 0.0625, 600 + 150 * i, 0.18, pan=-0.2 + 0.15 * i)
tap(T_TAP1)
whoosh(T_PROF - 0.05, 0.55, 0.4)
# the profile assembles: top bar, ring, stats, bio, buttons, posts
pop(T_PROF + 0.2, 480, 0.15)
riser(T_PROF + 0.1, 0.6, gain=0.12, f0=800, f1=5000)
for i in range(3):
    pop(T_PROF + 0.4 + i * 0.0625, 700 + 120 * i, 0.14, pan=-0.2 + 0.2 * i)
for i in range(6):
    key_click(T_PROF + 0.6 + i * 0.05, i)
pop(T_PROF + 0.75, 560, 0.16)
for i in range(3):
    pop(T_PROF + 0.9 + i * 0.125, 820 + 140 * i, 0.16, pan=-0.4 + 0.4 * i)
    pluck(T_PROF + 0.9 + i * 0.125, [72, 76, 79][i], 0.06)
for k in range(10):  # view counters ticking
    pop(T_PROF + 1.2 + k * 0.08, 1400 + k * 60, 0.04, pan=0.3)
# follow: tap, bright chime, followers 2 → 3
tap(T_TAP2, 0.6)
for k, n_ in enumerate([72, 76, 79, 84]):
    bell(T_TAP2 + k * 0.06, midi(n_), 0.22, pan=-0.3 + 0.2 * k, dec=0.9)
impact(T_TAP2, 0.3)
riser(T_TAP2 + 0.2, 0.8, gain=0.25, f0=400, f1=7000)
# end card: the mark slams, the handle, the line
impact(T_END, 0.9)
kick(T_END, 0.9)
pad(T_END, [45, 57, 60, 64, 69, 72], 2.5, 0.08, 2600)
bass(T_END, 33, 2.0, 0.3, 300)
bell(T_END, 440, 0.2, dec=1.4)
kick(T_END + 0.5, 0.7)
bell(T_END + 0.5, 659.25, 0.18, dec=1.0)
for k, n_ in enumerate([69, 72, 76, 81]):
    pluck(T_END + 1.0 + k * 0.125, n_ + 12, 0.08, pan=-0.3 + 0.2 * k)
bell(T_END + 1.5, 880, 0.12, dec=1.5)

# ================================================================ mix
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
with wave.open(sys.argv[1] if len(sys.argv) > 1 else 'soundtrack.wav', 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', sys.argv[1] if len(sys.argv) > 1 else 'soundtrack.wav', f'{n / SR:.2f}s')
