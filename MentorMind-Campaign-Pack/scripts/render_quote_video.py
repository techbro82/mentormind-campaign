import math, random, subprocess, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw
from social_lib import *

W, H, FPS, DUR = 1080, 1920, 30, 8.0
OUT = sys.argv[1]
DAY = int(sys.argv[2])

QUOTES = {
    4: dict(name="EPICTETUS", cat="ADVERSITY", era="a Stoic school at Nicopolis, around 108 AD",
            text="Remember that it is not he who reviles you or strikes you, who insults you, but it is your opinion about these things as being insulting."),
    5: dict(name="MARCUS AURELIUS", cat="MORTALITY", era="the Roman frontier, around 175 AD, on campaign",
            text="Take pains therefore to know what it is that thy nature requireth, and let nothing else distract thee."),
    6: dict(name="FREDERICK DOUGLASS", cat="FREEDOM", era="the American North, 1845, after escaping slavery",
            text="Freedom now appeared, to disappear no more forever."),
    7: dict(name="BENJAMIN FRANKLIN", cat="CRAFT", era="Philadelphia and London, late 1700s",
            text="Human felicity is produced not so much by great pieces of good fortune that seldom happen, as by little advantages that occur every day."),
    8: dict(name="RALPH WALDO EMERSON", cat="THINKING", era="New England, mid-1800s",
            text="We are afraid of truth, afraid of fortune, afraid of death and afraid of each other."),
}
Q = QUOTES[DAY]
G = np.array(GOLD, np.float32) / 255
P = np.array(PARCH, np.float32) / 255
PD = np.array(PARCH_DIM, np.float32) / 255
GS = np.array(GOLD_SOFT, np.float32) / 255


def mask(txt, f, tracking=0, pad=50):
    asc, desc = f.getmetrics()
    if tracking:
        ws = [f.getlength(c) for c in txt]
        w = int(sum(ws) + tracking * (len(txt) - 1))
    else:
        w = int(f.getlength(txt))
    im = Image.new("L", (w + 2 * pad, asc + desc + 2 * pad), 0)
    d = ImageDraw.Draw(im)
    if tracking:
        x = pad
        for c, wc in zip(txt, ws):
            d.text((x, pad + asc), c, font=f, fill=255, anchor="ls")
            x += wc + tracking
    else:
        d.text((pad, pad + asc), txt, font=f, fill=255, anchor="ls")
    return np.asarray(im, np.float32) / 255.0, pad + asc, pad


def blend(frame, m, color, x, y, op):
    h, w = m.shape
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, W), min(y + h, H)
    if x2 <= x1 or y2 <= y1 or op <= 0:
        return
    mm = (m[y1 - y:y2 - y, x1 - x:x2 - x] * op)[..., None]
    reg = frame[y1:y2, x1:x2]
    reg[:] = reg * (1 - mm) + color * mm


# ---- quote layout (safe zone: keep everything above y=1480)
text = "\u201c" + Q["text"] + "\u201d"
size, fq, lines = choose_size(text, 860, 470)
lh = size * 1.18
q_top, q_bot = 870, 1340
base0 = q_top + (q_bot - q_top - lh * len(lines)) / 2 + size * 0.95
words = []
for li, (ln, cw) in enumerate(lines):
    x = (W - cw) / 2
    sp = fq.getlength(" ")
    for w, wl in ln:
        m, b, pad = mask(w, fq)
        words.append(dict(m=m, b=b, pad=pad, x=int(x) - pad, y=int(base0 + li * lh)))
        x += wl + sp
n = len(words)
step = min(0.17, 3.2 / n)
w_start = 2.3
t_end = w_start + (n - 1) * step + 0.6

f_cat = font("Inter", 29, 500)
f_name = font("Inter", 32, 500)
fe = font("CormorantGaramond-Italic", 42, 500)
if fe.getlength(Q["era"]) > 860:
    fe = font("CormorantGaramond-Italic", 35, 500)
m_cat, b_cat, _ = mask(Q["cat"], f_cat, 9)
m_name, b_name, _ = mask(Q["name"], f_name, 9)
m_era, b_era, _ = mask(Q["era"], fe)

FLAME_H = 250
FK = FLAME_H / 44
FBX, FBY = 540, 650

# embers
rnd = random.Random(DAY)
E = [dict(ph=rnd.random(), dx=rnd.uniform(-26, 26), sp=rnd.uniform(0.32, 0.55), r=rnd.uniform(1.6, 3.0)) for _ in range(16)]
# dust
P_ = []
for _ in range(150):
    spr, s = sprite(rnd.uniform(1.3, 3.2))
    P_.append(dict(x=rnd.uniform(0, W), y=rnd.uniform(0, H), sp=rnd.uniform(24, 95), ph=rnd.uniform(0, 6.28),
                   tw=rnd.uniform(0, 6.28), spr=spr, s=s))

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
       "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
       "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT]
pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)

for i in range(int(DUR * FPS)):
    t = i / FPS
    rise = ss(t / 2.4)
    specs = [(540, 660, 330 + 110 * rise, 0.40 * ss(t / 2.0) * (1 + 0.04 * math.sin(t * 2.4)), GOLD),
             (540, 690, 120 + 60 * rise, 0.16 * ss(t / 2.0), (255, 226, 168)),
             (770 + 30 * math.sin(t * 0.5), 300, 640, 0.18, (63, 90, 140)),
             (120 + 40 * math.cos(t * 0.4), 1700, 560, 0.24, (40, 52, 80)),
             (540, 1850, 460, 0.16, GOLD)]
    frame = glow_layer(W, H, specs)

    fade = ss(t / 1.6)
    for p in P_:
        y = (p["y"] - p["sp"] * t) % (H + 60) - 30
        x = p["x"] + 16 * math.sin(p["ph"] + t * 0.6)
        a = (0.28 + 0.28 * math.sin(p["tw"] + t * 1.7)) * fade
        s = p["s"]
        xi, yi = int(x) - s // 2, int(y) - s // 2
        if xi < 0 or yi < 0 or xi + s > W or yi + s > H:
            continue
        frame[yi:yi + s, xi:xi + s] += p["spr"][..., None] * G * (a * 1.6)

    # flame + embers
    fo = ss((t - 0.4) / 1.2)
    if fo > 0:
        fimg, anc = flame_img(FK, t)
        paste_flame_np(frame, fimg, anc, FBX, FBY, fo)
        tip = FBY - FLAME_H * 0.9
        for e in E:
            tl = (t * e["sp"] + e["ph"]) % 1.0
            ex = FBX + e["dx"] * tl * 1.6 + 10 * math.sin(tl * 9 + e["ph"] * 6)
            ey = tip - 230 * tl
            a = math.sin(math.pi * tl) * 0.85 * fo
            spr, s = sprite(e["r"])
            xi, yi = int(ex) - s // 2, int(ey) - s // 2
            if 0 <= xi < W - s and 0 <= yi < H - s:
                frame[yi:yi + s, xi:xi + s] += spr[..., None] * np.array((1.0, 0.8, 0.45), np.float32) * a

    def anim(m, b, pad, color, cx_or_x, base, start, dur, centered=True, lift=22, blur=8):
        p = ss((t - start) / dur)
        if p <= 0:
            return
        mm = m
        sig = (1 - p) * blur
        if sig > 0.4:
            mm = cv2.GaussianBlur(m, (0, 0), sig)
        x = (W - m.shape[1]) // 2 if centered else cx_or_x
        blend(frame, mm, color, x, int(base - b + (1 - p) * lift), p)

    anim(m_cat, b_cat, 0, GS, 0, 775, 1.5, 0.9)
    r = ss((t - 1.9) / 0.8)
    if r > 0:
        ln_ = int(90 * r)
        frame[800:802, W // 2 - ln_ // 2:W // 2 + ln_ // 2] = (
            frame[800:802, W // 2 - ln_ // 2:W // 2 + ln_ // 2] * (1 - 0.8 * r) + np.array(GOLD_DIM, np.float32) / 255 * 0.8 * r)

    for k, w in enumerate(words):
        st = w_start + k * step
        p = ss((t - st) / 0.55)
        if p <= 0:
            continue
        mm = w["m"]
        sig = (1 - p) * 7
        if sig > 0.4:
            mm = cv2.GaussianBlur(mm, (0, 0), sig)
        blend(frame, mm, P, w["x"], int(w["y"] - w["b"] + (1 - p) * 14), p)

    anim(m_name, b_name, 0, G, 0, 1400, t_end + 0.2, 0.8, lift=14, blur=5)
    anim(m_era, b_era, 0, PD, 0, 1462, t_end + 0.7, 0.8, lift=14, blur=5)

    out8 = (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
    if i in (int(3.6 * FPS), int(7.6 * FPS)):
        Image.fromarray(out8).save(f"/home/claude/social/qstill_{i}.png")
    pipe.stdin.write(out8.tobytes())

pipe.stdin.close()
pipe.wait()
print("done", OUT, pipe.returncode)
