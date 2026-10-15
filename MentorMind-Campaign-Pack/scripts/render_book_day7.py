import sys, json, math, random, subprocess, time
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, "/home/claude/social")
import render_feeling as rf
from social_lib import ss, glow_layer, sprite, flame_img, paste_flame_np, GOLD, PARCH, PARCH_DIM
import book_lib as BL
from book_lib import *

W, H, FPS = 1080, 1920, 24
G = np.array(GOLD, F32) / 255
P = np.array(PARCH, F32) / 255
PD = np.array(PARCH_DIM, F32) / 255
V = json.load(open("/home/claude/voice/day07_voice.json"))
st, du = V["starts"], V["durs"]
OUT_SILENT = "/home/claude/social/_silent_7.mp4"
OUT_FINAL = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/social/day07_final.mp4"
STILLS = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else [])]
T0F = time.time()

# ---------------------------------------------------------------- timeline
E = [st[i] + du[i] for i in range(5)]
TC0, TC1 = st[0] + 0.30, st[0] + 2.10              # the cover swings open
TS1_0, TS1_1 = E[1] + 0.10, E[1] + 1.25            # first page turn
TS2_0, TS2_1 = E[3] + 0.10, E[3] + 1.25            # second page turn
t_close = E[4] + 1.35
T = t_close + 4.2

keys = [  # (time, camera scale, centre x, centre y, extra dip when it travels between pages)
    (0.0, 1.10, 261, 520, 0), (TC0, 1.10, 261, 520, 0), (TC1 + 0.2, 1.00, 0, 590, 0),
    (st[1] - 0.15, 1.90, 250, 590, 0), (E[1] + 0.05, 1.96, 250, 590, 0),
    (TS1_0 + 0.25, 1.15, 0, 590, 0), (TS1_1 - 0.05, 1.15, 0, 590, 0),
    (st[2] + 0.20, 1.90, -250, 590, 0), (E[2] + 0.05, 1.96, -250, 590, 0),
    (st[3] - 0.10, 1.90, 250, 590, 0.55), (E[3] + 0.05, 1.96, 250, 590, 0),
    (TS2_0 + 0.25, 1.15, 0, 590, 0), (TS2_1 - 0.05, 1.15, 0, 590, 0),
    (st[4] + 0.25, 1.90, -250, 590, 0), (E[4] + 0.15, 1.96, -250, 590, 0),
    (t_close - 0.25, 0.90, 0, 590, 0), (T + 1, 0.90, 0, 590, 0)]


def camera(t):
    for k in range(len(keys) - 1):
        t0, s0, x0, y0, _ = keys[k]
        t1, s1, x1, y1, dip = keys[k + 1]
        if t0 <= t <= t1:
            u = ss((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
            return s0 + (s1 - s0) * u - dip * math.sin(math.pi * u), x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
    return keys[-1][1], keys[-1][2], keys[-1][3]


def intensity(t):
    I = 0.28 + 0.34 * ss((t - 0.4) / 1.2) + 0.14 * ss((t - TC1) / 1.0) + 0.12 * ss((t - TS1_0) / 1.0) + 0.12 * ss((t - st[3]) / 1.0) + 0.12 * ss((t - TS2_0) / 1.0)
    return min(I, 1.0)


# ---------------------------------------------------------------- the book, built once
A = build_assets()
closed = compose(None, A["P1"], False, True, A)
BASE0 = closed.copy()
stateB = compose(None, A["P1"], True, True, A)
BASE1 = compose(None, A["P3"], True, True, A)
stateC = compose(A["P2b"], A["P3"], True, True, A)
BASE2 = compose(A["P2b"], None, True, True, A)
stateD = compose(A["P4b"], None, True, True, A)
apply_turn(closed, A["cover"], A["end"], 0.0, BOARD_Y0, 0.35)       # the closed book, cover on top
print("book built", round(time.time() - T0F, 1), "s", flush=True)


def scene(t):
    if t < TC0:
        return closed
    if t < TC1:
        c = BASE0.copy(); apply_turn(c, A["cover"], A["end"], (t - TC0) / (TC1 - TC0), BOARD_Y0, 0.35); return c
    if t < TS1_0:
        return stateB
    if t < TS1_1:
        c = BASE1.copy(); apply_turn(c, A["P1"], A["P2b"], (t - TS1_0) / (TS1_1 - TS1_0), PAGE_Y0, 1.0); return c
    if t < TS2_0:
        return stateC
    if t < TS2_1:
        c = BASE2.copy(); apply_turn(c, A["P3"], A["P4b"], (t - TS2_0) / (TS2_1 - TS2_0), PAGE_Y0, 1.0); return c
    return stateD


# ---------------------------------------------------------------- the frame around it
bg = glow_layer(W, H, [(540, 1000, 700, 0.10, GOLD), (800, 260, 640, 0.10, (63, 90, 140)), (120, 1500, 560, 0.12, (40, 52, 80))])
close_bg = glow_layer(W, H, [(540, 1250, 520, 0.28, GOLD), (800, 260, 640, 0.17, (63, 90, 140)), (120, 1600, 520, 0.22, (40, 52, 80))])
vig = (1 - 0.30 * (((np.arange(W)[None, :] - W / 2) ** 2 + (np.arange(H)[:, None] - H / 2) ** 2) /
                   ((W / 2) ** 2 + (H / 2) ** 2)) ** 1.1)[..., None].astype(F32)
GR = 520
gy_, gx_ = np.mgrid[-GR:GR + 1, -GR:GR + 1].astype(F32)
gspr = np.exp(-((gx_ ** 2 + gy_ ** 2) / (GR * 0.42) ** 2)).astype(F32)
_GC = {}


def add_glow(frame, cx, cy, R, amt):
    if R not in _GC:
        _GC[R] = cv2.resize(gspr, (2 * R, 2 * R), interpolation=cv2.INTER_LINEAR)
    s_ = _GC[R]
    x0, y0 = int(cx - R), int(cy - R)
    x1, y1, x2, y2 = max(x0, 0), max(y0, 0), min(x0 + 2 * R, W), min(y0 + 2 * R, H)
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] += s_[y1 - y0:y2 - y0, x1 - x0:x2 - x0, None] * np.array([1.0, 0.86, 0.6], F32) * amt


FL = [flame_img(4.6, i * 0.17) for i in range(12)]                 # a cycle of flame shapes, reused
rnd = random.Random(77)
dust = []
for _ in range(110):
    spr, sz = sprite(rnd.uniform(1.3, 3.0))
    dust.append(dict(x=rnd.uniform(0, W), y=rnd.uniform(0, H), sp=rnd.uniform(14, 50), ph=rnd.uniform(0, 6.28),
                     tw=rnd.uniform(0, 6.28), spr=spr, s=sz))
LINES = ["Three ways to find your mentor.", "First, say what you are carrying.", "Second, name someone you wish you could learn from.",
         "Third, browse by the part of your life that needs counsel.", "Then just say something."]
blocks = [rf.block_mask(t_, "CormorantGaramond", W, 215)[:2] for t_ in LINES]
WIN = [(st[i] - 0.10, (st[i + 1] - 0.10) if i < 4 else t_close - 0.05) for i in range(5)]

bug_w = 240
bug_arr, bug_ox, bug_oy = rf.lock_scaled(bug_w)
bug_sh = cv2.GaussianBlur(bug_arr.max(axis=2), (0, 0), 7)
bug_cx, bug_cy = 62 + bug_w / 2, 1405
hm, hb = rf.tmask("@mentormindhq", rf.font("Inter", 23, 500), 3)
hm_sh = cv2.GaussianBlur(hm, (0, 0), 5)
hx, hy = 62 - 40, int(bug_cy + 62 - hb)
LS, lox, loy = rf.lock_scaled(700)
tag_m, tag_b = rf.tmask("Opening soon.", rf.font("CormorantGaramond-Italic", 64, 500), 0)
sub_m, sub_b = rf.tmask("AI mentors, built from what they wrote.", rf.font("Inter", 31, 500), 0)
cta_m, cta_b = rf.tmask("FOLLOW FOR LAUNCH DAY", rf.font("Inter", 34, 500), 8)
XF2 = 0.9
rng = np.random.RandomState(8)
NOISE = [rng.normal(0, 0.006, (H, W, 1)).astype(F32) for _ in range(6)]
YS = np.arange(0, H, 8, dtype=F32)[:, None]
XS = np.arange(0, W, 8, dtype=F32)[None, :]

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
       "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", OUT_SILENT]
pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
still_idx = {int(round(x * FPS)): x for x in STILLS}

for i in range(int(T * FPS)):
    t = i / FPS
    Sc, cx, cy = camera(t)
    I = intensity(t) * (1 + 0.025 * math.sin(t * 9.0) + 0.015 * math.sin(t * 14.3))
    sc = scene(t)
    Mx = np.array([[Sc / S0, 0, 540 - Sc * SPX / S0 - Sc * cx], [0, Sc / S0, 1060 - Sc * (cy + YOFF / S0)]], F32)
    frame = cv2.warpAffine(sc, Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0.03, 0.036, 0.056))
    xx = (XS - 540) / Sc + cx
    yy = (YS - 1060) / Sc + cy
    Mhi = np.exp(-((xx / 620) ** 2 + ((yy - 40) / 1000) ** 2))
    f = cv2.resize(0.30 + I * (0.22 + 0.56 * Mhi), (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    frame = frame * f + bg * 0.55

    # the flame above the spine
    fx = 540 + Sc * (0 - cx)
    fy = 1060 + Sc * (29 - cy)
    grow = ss((t - 0.35) / 1.0)
    if grow > 0 and -200 < fy < H + 100:
        img, anc = FL[int(t * 6) % 12]
        add_glow(frame, fx, fy - 100, 420, 0.40 * grow * min(I + 0.2, 1.0))
        paste_flame_np(frame, img, anc, int(fx), int(fy), op=grow)

    fade = ss(t / 1.2)
    for p_ in dust:
        x = (p_["x"] - p_["sp"] * 0.8 * t) % (W + 60) - 30
        y = p_["y"] - 9 * t + 12 * math.sin(p_["ph"] + t * 0.6)
        a_ = (0.20 + 0.20 * math.sin(p_["tw"] + t * 1.7)) * fade
        sz = p_["s"]
        xi, yi = int(x) - sz // 2, int(y) - sz // 2
        if 0 <= xi < W - sz and 0 <= yi < H - sz:
            frame[yi:yi + sz, xi:xi + sz] += p_["spr"][..., None] * G * (a_ * 1.3)

    # a soft dark band at the top when we are close, so the words read over the pale margin
    scr = 0.72 * ss((Sc - 1.25) / 0.45)
    if scr > 0.01:
        gy = np.clip(1 - np.arange(520, dtype=F32) / 520, 0, 1) ** 1.2
        frame[:520] *= (1 - scr * gy)[:, None, None]

    bo = ss(t / 0.8) * 0.92
    h_, w_ = bug_arr.shape[:2]
    bx0, by0 = int(bug_cx - bug_ox - w_ / 2), int(bug_cy - bug_oy - h_ / 2)
    bx1, by1, bx2, by2 = max(bx0, 0), max(by0, 0), min(bx0 + w_, W), min(by0 + h_, H)
    frame[by1:by2, bx1:bx2] *= (1 - 0.55 * bo * bug_sh[by1 - by0:by2 - by0, bx1 - bx0:bx2 - bx0, None])
    rf.add_light(frame, bug_arr, bug_ox, bug_oy, bug_cx, bug_cy, bo)
    frame[hy:hy + hm_sh.shape[0], hx:hx + hm_sh.shape[1]] *= (1 - 0.5 * bo * hm_sh[..., None])
    rf.blend(frame, hm, P, hx, hy, bo * 0.9)

    ca = ss((t - (t_close - XF2 / 2)) / XF2)
    if ca > 0:
        cb = close_bg.copy()
        rf.add_light(cb, LS, lox, loy, 540, 640, 1.0)
        for p_ in dust:
            x = (p_["x"] - p_["sp"] * 0.8 * t) % (W + 60) - 30
            y = (p_["y"] - 12 * t) % H
            sz = p_["s"]
            xi, yi = int(x) - sz // 2, int(y) - sz // 2
            if 0 <= xi < W - sz and 0 <= yi < H - sz:
                cb[yi:yi + sz, xi:xi + sz] += p_["spr"][..., None] * G * 0.35
        for (m_, b_, col, base_y, t_on) in ((tag_m, tag_b, P, 830, t_close + 0.7), (sub_m, sub_b, PD, 900, t_close + 1.4), (cta_m, cta_b, G, 1010, t_close + 2.3)):
            pc = ss((t - t_on) / 0.7)
            if pc > 0:
                rf.blend(cb, m_, col, (W - m_.shape[1]) // 2, int(base_y - b_ + (1 - pc) * 12), pc)
        frame = frame * (1 - ca) + cb * ca

    for (m, hh), (w0, w1) in zip(blocks, WIN):
        pin = ss((t - w0) / 0.5)
        pout = 1 - ss((t - (w1 - 0.35)) / 0.35)
        op = pin * pout
        if op > 0:
            mm = m
            sg = (1 - pin) * 8
            if sg > 0.4:
                mm = cv2.GaussianBlur(m, (0, 0), sg)
            ytxt = 200 + 165 * ss((Sc - 1.35) / 0.4)                       # close up, the words sit on the plate's empty top, clear of the numeral
            yv = int(ytxt - hh / 2 + (1 - pin) * 18)
            reg = frame[yv:yv + hh]
            reg[:] = reg * (1 - (mm * op)[..., None]) + P * (mm * op)[..., None]

    frame = frame * vig + NOISE[i % 6]
    o8 = (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
    if i in still_idx:
        Image.fromarray(o8).save(f"/home/claude/social/_still_7_{still_idx[i]:.2f}.png")
    pipe.stdin.write(o8.tobytes())
    if i % 60 == 0:
        print("frame", i, "of", int(T * FPS), round(time.time() - T0F), "s", flush=True)

pipe.stdin.close()
pipe.wait()

# a little paper sound at the cover and the two page turns (quiet; the voice stays in front)
sr = 24000
sfx = np.zeros(int(T * sr) + sr, np.float32)
rs = np.random.RandomState(3)
def swish(t0, dur, gain):
    n = int(dur * sr); x = rs.randn(n).astype(np.float32)
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 2
    i0 = int(t0 * sr); sfx[i0:i0 + n] += x * env * gain
swish(TC0 + 0.2, 1.4, 0.05); swish(TS1_0, 0.9, 0.05); swish(TS2_0, 0.9, 0.05)
import soundfile as sf
sf.write("/home/claude/social/_sfx7.wav", sfx, sr)
af = "highpass=f=70,equalizer=f=170:t=q:w=1:g=3,equalizer=f=3800:t=q:w=1:g=-1.5,acompressor=threshold=-20dB:ratio=2.5:attack=15:release=180:makeup=2"
wind = f"lowpass=f=800,highpass=f=70,volume=0.05,afade=t=in:d=1.2,afade=t=out:st={max(T - 1.6, 0):.2f}:d=1.6"
paper = "highpass=f=1500,lowpass=f=6000,volume=1.0"
fc = (f"[1:a]{af}[v];[2:a]{wind}[w];[3:a]{paper}[pp];"
      f"[v][w][pp]amix=inputs=3:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7,alimiter=limit=0.95[a]")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT_SILENT, "-i", "/home/claude/voice/day07_voice_raw.wav",
                "-f", "lavfi", "-i", f"anoisesrc=d={T:.2f}:c=pink:r=24000:a=0.5", "-i", "/home/claude/social/_sfx7.wav",
                "-filter_complex", fc, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", f"{T:.2f}", OUT_FINAL], check=True)
print("done", OUT_FINAL, f"({T:.1f}s)", round(time.time() - T0F), "s")
