"""Soundtrack for the Bitcoin Pizza reel: synthesized, 120 BPM in D minor, every hit placed on the reel's own cues.

    node scripts/cues.cjs http://localhost:8765/index.html audio/pizza-cues.json
    python3 scripts/soundtrack_pizza.py audio/pizza-cues.json out.wav

Every 10x in the pizzas' value rings the next note of a rising pentatonic; each gate ($1M … $1B) is an impact;
crashes get a falling sweep.
"""
import json
import sys

import numpy as np

import synth as s
from synth import BEAT, bass, bell, clap, hat, impact, key_click, kick, midi, pad, pluck, pop, riser, snare, tap, whoosh

cue = json.load(open(sys.argv[1]))
s.init(cue['DUR'])
DUR = cue['DUR']
T2A, T2B, T2END, T4 = cue['T2A'], cue['T2B'], cue['T2END'], cue['T4']
DECADES = [d['t'] for d in cue['decades']]
GATES = cue['gates']
EVENTS = cue['events']


def fall(at, gain=0.3):
    """A falling sweep for a crash."""
    t = s.t_axis(0.7)
    f = 420 * np.exp(-t / 0.18) + 60
    sig = np.sin(2 * np.pi * np.cumsum(f) / s.SR) * s.env(0.7, 0.005, 0.3)
    s.place(None, np.tanh(sig * 1.5), at, gain, rev=0.3)


def coin(at, gain=0.2, pan=0.0):
    """A bright metallic clink."""
    bell(at, 2093, gain, pan, dec=0.25)
    bell(at + 0.03, 2637, gain * 0.7, -pan, dec=0.3)


# D minor: Dm – Bb – F – C, one chord per bar
CH = [(38, [62, 65, 69]), (34, [58, 62, 65]), (41, [60, 65, 69]), (36, [60, 64, 67])]
PENT = [74, 77, 79, 81, 84, 86, 89, 91, 93, 96]  # D minor pentatonic, rising
BAR = 4 * BEAT


def chord_at(t):
    return CH[int(t // BAR) % 4]


# ---------------------------------------------------------------- act 1 — the hook (0–4)
pad(0.0, [50, 57, 62, 65], 4.2, 0.07, 900)
riser(0.0, 0.6, 0.18, 200, 2500)
for i in range(12):  # the date types itself
    key_click(0.5 + i * 0.04, i)
for at, note in ((1.0, 62), (1.5, 65), (2.0, 69)):
    kick(at, 1.0)
    impact(at, 0.45)
    bell(at, midi(note), 0.16)
coin(1.5, 0.25, 0.3)
coin(1.62, 0.18, -0.3)
for k in range(2):  # the pizzas land
    pop(2.0 + k * 0.125 + 0.18, 170, 0.5)
    whoosh(2.0 + k * 0.125 - 0.1, 0.25, 0.15)
# ka-ching on the $41 tag
coin(2.5, 0.35)
bell(2.5, midi(86), 0.2, dec=0.5)
impact(2.5, 0.3)
for i in range(5):
    pluck(3.0 + i * 0.0625, [74, 77, 81, 84, 86][i], 0.08, pan=-0.4 + 0.2 * i)
riser(3.0, 1.0, 0.4, 250, 9000)
whoosh(3.45, 0.55, 0.4, reverse=True)

# ---------------------------------------------------------------- act 2 — the flight (4–22)
impact(T2A, 1.1)
hat(T2A, 0.25, open_=True)
t = T2A
while t < T2B - 1e-6:
    b = round((t - T2A) / BEAT)
    root, chord = chord_at(t)
    half = 15.0 <= t < 17.0  # a half-time bar pair for contrast before the last climb
    if not half or b % 2 == 0:
        kick(t, 0.95)
    if b % 2 == 1:
        clap(t, 0.32 if not half else 0.4)
        snare(t, 0.25)
    hat(t + BEAT / 2, 0.11, open_=(b % 4 == 3))
    if not half:
        hat(t + BEAT / 4, 0.045, pan=-0.3)
        hat(t + 3 * BEAT / 4, 0.045, pan=-0.3)
    cut = 500 + 1600 * (t - T2A) / (T2B - T2A)
    for k in range(2):
        bass(t + k * BEAT / 2, root - 12 + (12 if k == 1 else 0), BEAT / 2 * 0.9, 0.3, cut)
    # rolling 16th arp, brighter as the value grows
    for k in range(4):
        n_ = chord[(b * 4 + k) % 3] + 12 + (12 if (k == 3 and t > 12) else 0)
        pluck(t + k * BEAT / 4, n_, 0.06 + 0.03 * (t - T2A) / (T2B - T2A), pan=0.45 * ((k % 2) * 2 - 1))
    t += BEAT
i = 0
while T2A + i * BAR < T2B:  # pads, one per bar
    at = T2A + i * BAR
    root, chord = chord_at(at)
    pad(at, chord + [chord[0] - 12], min(BAR, T2B - at) + 0.3, 0.05, 900 + 120 * i)
    i += 1
# every 10x: the next note up the scale, plus coins
for k, at in enumerate(DECADES):
    bell(at, midi(PENT[k]), 0.26, pan=-0.2 + 0.05 * k, dec=0.9)
    pluck(at, PENT[k] - 12, 0.12)
    coin(at + 0.06, 0.12, 0.4)
# gates: a reversed swell into an impact and a crash
for at in GATES:
    whoosh(at - 0.45, 0.45, 0.35, reverse=True)
    impact(at, 0.85)
    hat(at, 0.3, open_=True)
    kick(at, 0.8)
for at in EVENTS:  # crashes fall
    fall(at, 0.28)
bell(cue['peak'], midi(86), 0.2, dec=1.2)
bell(cue['peak'] + 0.05, midi(93), 0.15, dec=1.2)
riser(17.0, 2.0, 0.4, 250, 9000)  # into the $1B gate
# the reveal: drums out, a swell, a breath, the drop
pad(T2B, [50, 57, 62, 65, 69], T2END - T2B + 0.4, 0.09, 2600)
riser(T2B + 0.5, T2END - T2B - 0.6, 0.45, 300, 10000)
for k in range(8):
    pluck(T2B + k * 0.125, [74, 77, 81, 86, 89, 93, 98, 101][k], 0.07, pan=-0.4 + 0.1 * k)

# ---------------------------------------------------------------- act 3 — the payoff (22–26)
impact(T2END, 1.2)
kick(T2END, 1.0)
hat(T2END, 0.3, open_=True)
bass(T2END, 26, 2.0, 0.4, 300)
pad(T2END, [50, 62, 65, 69, 74], 4.0, 0.08, 2600)
t = T2END + BEAT
while t < T4 - 0.5:
    b = round((t - T2END) / BEAT)
    if b % 2 == 0:
        kick(t, 0.7)
    else:
        clap(t, 0.25)
    hat(t + BEAT / 2, 0.07)
    t += BEAT
whoosh(22.9, 0.25, 0.25)
pop(23.0, 300, 0.4)
coin(23.0, 0.25)
for k in range(12):  # the multiplier counting up
    pop(23.5 + k * 0.075, 900 + k * 110, 0.06 + 0.01 * k, pan=0.3)
coin(24.4, 0.3)
bell(24.5, midi(81), 0.2, dec=0.8)
whoosh(T4 - 0.55, 0.6, 0.45, reverse=True)

# ---------------------------------------------------------------- act 4 — the search and the profile (26–36)
T_TYPE, T_KEY, NK = cue['T_TYPE'], cue['T_KEY'], cue['n']
T_RES, T_TAP1, T_PROF, T_TAP2, T_END = cue['T_RES'], cue['T_TAP1'], cue['T_PROF'], cue['T_TAP2'], cue['T_END']
for i in range(5):
    root, chord = CH[i % 4]
    pad(T4 + i * 2.0, chord + [chord[0] + 12], 2.0 + (0.5 if i == 4 else 0.0), 0.05, 1400 + 200 * i)
t = T4
while t < T_END - 1e-6:
    b = round((t - T4) / BEAT)
    if b % 4 in (0, 2):
        kick(t, 0.55)
    if b % 4 == 3:
        kick(t + BEAT / 2, 0.45)
    if b % 2 == 1:
        snare(t, 0.18)
    hat(t + BEAT / 2, 0.06)
    if t >= T_PROF:
        hat(t + BEAT / 4, 0.035, pan=-0.3)
        hat(t + 3 * BEAT / 4, 0.035, pan=-0.3)
    root = CH[int((t - T4) // 2.0) % 4][0]
    bass(t, root - 12, BEAT * 0.8, 0.22, 400 + (300 if t >= T_PROF else 0))
    t += BEAT
pop(T4, 500, 0.25)
for i in range(5):
    pluck(T4 + 0.25 + i * 0.125, [74, 77, 79, 81, 86][i], 0.07, pan=-0.4 + 0.2 * i)
for i in range(NK):
    key_click(T_TYPE + i * T_KEY, i)
for i in range(4):
    pop(T_RES + i * 0.0625, 600 + 150 * i, 0.18, pan=-0.2 + 0.15 * i)
tap(T_TAP1)
whoosh(T_PROF - 0.05, 0.55, 0.4)
pop(T_PROF + 0.2, 480, 0.15)
riser(T_PROF + 0.1, 0.6, gain=0.12, f0=800, f1=5000)
for i in range(3):
    pop(T_PROF + 0.4 + i * 0.0625, 700 + 120 * i, 0.14, pan=-0.2 + 0.2 * i)
for i in range(6):
    key_click(T_PROF + 0.6 + i * 0.05, i)
pop(T_PROF + 0.75, 560, 0.16)
for i in range(3):
    pop(T_PROF + 0.9 + i * 0.125, 820 + 140 * i, 0.16, pan=-0.4 + 0.4 * i)
    pluck(T_PROF + 0.9 + i * 0.125, [74, 77, 81][i], 0.06)
for k in range(10):
    pop(T_PROF + 1.2 + k * 0.08, 1400 + k * 60, 0.04, pan=0.3)
tap(T_TAP2, 0.6)
for k, n_ in enumerate([74, 77, 81, 86]):
    bell(T_TAP2 + k * 0.06, midi(n_), 0.22, pan=-0.3 + 0.2 * k, dec=0.9)
coin(T_TAP2 + 0.25, 0.2)
impact(T_TAP2, 0.3)
riser(T_TAP2 + 0.2, 0.8, gain=0.25, f0=400, f1=7000)
impact(T_END, 0.9)
kick(T_END, 0.9)
pad(T_END, [50, 62, 65, 69, 74, 77], DUR - T_END, 0.08, 2600)
bass(T_END, 26, 2.0, 0.3, 300)
bell(T_END, midi(74), 0.2, dec=1.4)
kick(T_END + 0.5, 0.7)
bell(T_END + 0.5, midi(81), 0.18, dec=1.0)
for k, n_ in enumerate([74, 77, 81, 86]):
    pluck(T_END + 1.0 + k * 0.125, n_ + 12, 0.08, pan=-0.3 + 0.2 * k)
bell(T_END + 1.5, midi(86), 0.12, dec=1.5)

s.mixdown(sys.argv[2] if len(sys.argv) > 2 else 'soundtrack.wav')
