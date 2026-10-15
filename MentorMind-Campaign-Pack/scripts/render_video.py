import math, random, subprocess, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
FPS, DUR = 30, 8.0
GOLD = np.array((216, 166, 87), np.float32) / 255
GOLD_SOFT = np.array((202, 162, 94), np.float32) / 255
PARCH = np.array((236, 227, 210), np.float32) / 255
PARCH_DIM = np.array((169, 162, 144), np.float32) / 255
INK = np.array((14, 17, 26), np.float32) / 255
FF = "/tmp/fonts/"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/social/day01.mp4"
STILLS = {int(1.6 * FPS): "s1", int(3.0 * FPS): "s2", int(4.8 * FPS): "s3", int(7.3 * FPS): "s4"}


def font(name, size, wght=500):
    f = ImageFont.truetype(FF + name + ".ttf", size)
    vals = []
    for a in f.get_variation_axes():
        n = a["name"].decode() if isinstance(a["name"], bytes) else a["name"]
        vals.append(wght if "eight" in n else a["default"])
    f.set_variation_by_axes(vals)
    return f


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
    return np.asarray(im, np.float32) / 255.0, pad + asc


def ss(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def blend(frame, m, color, x, y, op):
    h, w = m.shape
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, W), min(y + h, H)
    if x2 <= x1 or y2 <= y1 or op <= 0:
        return
    mm = (m[y1 - y:y2 - y, x1 - x:x2 - x] * op)[..., None]
    reg = frame[y1:y2, x1:x2]
    reg[:] = reg * (1 - mm) + color * mm


# ---- text elements: (mask, baseline offset, color, x, baseline y, in_start, in_dur, out_start)
f_head = font("CormorantGaramond", 158, 500)
f_head_i = font("CormorantGaramond-Italic", 158, 500)
f_sub = font("CormorantGaramond-Italic", 66, 500)
f_word = font("CormorantGaramond", 132, 500)
f_cta = font("Inter", 37, 500)
f_dom = font("Inter", 34, 500)

els = []


def add_center(txt, f, color, base, s, d, out, tracking=0):
    m, b = mask(txt, f, tracking)
    els.append(dict(m=m, b=b, c=color, x=(W - m.shape[1]) // 2, y=base, s=s, d=d, o=out))


add_center("Reach toward", f_head, PARCH, 770, 0.9, 1.0, 5.4)
add_center("the wise,", f_head_i, GOLD, 940, 1.4, 1.0, 5.4)
add_center("Something is coming.", f_sub, PARCH_DIM, 1075, 3.7, 1.0, 5.4)

# wordmark: "Mentor" parch + "Mind" gold, aligned on one baseline
mA, bA = mask("Mentor", f_word)
mB, bB = mask("Mind", f_word)
trimA = 100  # remove double padding between the two pieces
total = mA.shape[1] + mB.shape[1] - trimA
x0 = (W - total) // 2
els.append(dict(m=mA, b=bA, c=PARCH, x=x0, y=900, s=5.9, d=1.0, o=99))
els.append(dict(m=mB, b=bB, c=GOLD, x=x0 + mA.shape[1] - trimA, y=900, s=5.9, d=1.0, o=99))
add_center("FOLLOW FOR LAUNCH DAY", f_cta, GOLD, 1030, 6.4, 0.9, 99, tracking=8)
add_center("mentormindhq.com", f_dom, PARCH, 1105, 6.7, 0.9, 99, tracking=3)

# ---- background layers computed at half resolution
hw, hh = W // 2, H // 2
Y, X = np.mgrid[0:hh, 0:hw].astype(np.float32) * 2


def glow(cx, cy, r, inten, color):
    g = inten * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (r * r)))
    return g[..., None] * color


vig = 1 - 0.38 * (((X - W / 2) ** 2 + (Y - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)) ** 1.1
vig = vig[..., None]

# ---- dust
rnd = random.Random(7)
P = []
for _ in range(170):
    r = rnd.uniform(1.3, 3.4)
    s = int(r * 4) | 1
    ax = np.arange(s) - s // 2
    spr = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (r * r * 0.9)).astype(np.float32)
    P.append(dict(x=rnd.uniform(0, W), y=rnd.uniform(0, H), sp=rnd.uniform(26, 105),
                  ph=rnd.uniform(0, 6.28), tw=rnd.uniform(0, 6.28), spr=spr, s=s))

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
       "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
       "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT]
pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)

N = int(DUR * FPS)
for i in range(N):
    t = i / FPS
    rise = ss(t / 3.4)
    lay = INK * np.ones((hh, hw, 3), np.float32)
    lay = lay + glow(540, 1470 - 70 * rise, 190 + 420 * rise, 0.52 * ss(t / 2.6) * (1 + 0.05 * math.sin(t * 2.2)), GOLD)
    lay = lay + glow(540, 1470 - 70 * rise, 80 + 160 * rise, 0.22 * ss(t / 2.6), np.array((1.0, 0.89, 0.66), np.float32))
    lay = lay + glow(770 + 30 * math.sin(t * 0.5), 300, 640, 0.20, np.array((63, 90, 140), np.float32) / 255)
    lay = lay + glow(120 + 40 * math.cos(t * 0.4), 1700, 560, 0.28, np.array((40, 52, 80), np.float32) / 255)
    lay = lay * vig
    frame = cv2.resize(lay, (W, H), interpolation=cv2.INTER_LINEAR)

    fade = ss(t / 1.6)
    for p in P:
        y = (p["y"] - p["sp"] * t) % (H + 60) - 30
        x = p["x"] + 16 * math.sin(p["ph"] + t * 0.6)
        a = (0.30 + 0.30 * math.sin(p["tw"] + t * 1.7)) * fade
        s = p["s"]
        xi, yi = int(x) - s // 2, int(y) - s // 2
        if xi < 0 or yi < 0 or xi + s > W or yi + s > H:
            continue
        frame[yi:yi + s, xi:xi + s] += p["spr"][..., None] * GOLD * (a * 1.7)

    for e in els:
        p = ss((t - e["s"]) / e["d"])
        q = ss((t - e["o"]) / 0.6)
        op = p * (1 - q)
        if op <= 0:
            continue
        m = e["m"]
        sig = (1 - p) * 9
        if sig > 0.4:
            m = cv2.GaussianBlur(m, (0, 0), sig)
        yy = int(e["y"] - e["b"] + (1 - p) * 30)
        blend(frame, m, e["c"], e["x"], yy, op)

    # thin gold rule that draws itself between headline and subline, then with the wordmark
    r1 = ss((t - 3.3) / 0.8) * (1 - ss((t - 5.4) / 0.6))
    r2 = ss((t - 6.2) / 0.8)
    for rr, yline in ((r1, 1004), (r2, 945)):
        if rr > 0:
            ln = int(130 * rr)
            if ln > 1:
                frame[yline:yline + 2, W // 2 - ln // 2:W // 2 + ln // 2] = (
                    frame[yline:yline + 2, W // 2 - ln // 2:W // 2 + ln // 2] * (1 - 0.8 * rr) + GOLD * 0.8 * rr)

    out = np.clip(frame, 0, 1)
    out8 = (out * 255 + 0.5).astype(np.uint8)
    if i in STILLS:
        Image.fromarray(out8).save(f"/home/claude/social/still_{STILLS[i]}.png")
    pipe.stdin.write(out8.tobytes())

pipe.stdin.close()
pipe.wait()
print("done", OUT, pipe.returncode)
