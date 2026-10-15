import sys, math, random, subprocess, json
import numpy as np
import cv2
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/claude/social")
from social_lib import font, ss, glow_layer, sprite, layout_words, GOLD, PARCH, PARCH_DIM

STYLE = sys.argv[1]                      # "sil" or "cartoon"
W, H, FPS = 1080, 1920, 30
V = json.load(open("/home/claude/voice/story_voice.json"))
t1, t2, T = V["t1"], V["t2"], V["T"]
SILENT = f"/home/claude/social/story_{STYLE}_silent.mp4"
FINAL = f"/home/claude/social/story_{STYLE}.mp4"
G = np.array(GOLD, np.float32) / 255
P = np.array(PARCH, np.float32) / 255

# ---------------- background ----------------
sky = glow_layer(W, H, [(540, 1290, 560, 0.30, GOLD), (800, 260, 640, 0.17, (63, 90, 140)),
                        (120, 1500, 520, 0.20, (40, 52, 80))])
yy = np.arange(H, dtype=np.float32)[:, None, None]
sky = sky * (0.50 + 0.50 * np.clip(yy / 1300.0, 0, 1) ** 1.2)

LW, LTOP = 3240, 1100                    # layers tile every 2160 px
LH = H - LTOP


def make_layer(ridge, amps, freqs, phases, c_top, c_bot):
    S = 2
    x = np.arange(LW * S, dtype=np.float32) / S
    y = np.full_like(x, ridge - LTOP)
    for a, f, p in zip(amps, freqs, phases):
        y += a * np.sin(2 * np.pi * f * x / 2160.0 + p)
    im = Image.new("L", (LW * S, LH * S), 0)
    d = ImageDraw.Draw(im)
    pts = [(float(i), float(y[i]) * S) for i in range(0, LW * S, 3)]
    pts += [(LW * S, LH * S), (0, LH * S)]
    d.polygon(pts, fill=255)
    a = np.asarray(im.resize((LW, LH), Image.LANCZOS), np.float32)[..., None] / 255.0
    t = np.linspace(0, 1, LH, dtype=np.float32)[:, None, None]
    rgb = (np.array(c_top, np.float32) * (1 - t) + np.array(c_bot, np.float32) * t) / 255.0
    rgb = np.broadcast_to(rgb, (LH, LW, 3)).copy()
    return rgb, a


LAYERS = [
    (make_layer(1240, [34, 18, 9], [2, 3, 7], [0.4, 1.9, 0.7], (58, 58, 76), (20, 26, 46)), 18),
    (make_layer(1385, [30, 14, 8], [3, 5, 9], [2.2, 0.3, 1.1], (26, 30, 50), (12, 16, 30)), 48),
    (make_layer(1525, [9, 5, 3], [4, 6, 11], [1.0, 2.6, 0.2], (14, 17, 30), (8, 10, 19)), 130),
]

vig = 1 - 0.30 * (((np.arange(W)[None, :] - W / 2) ** 2 + (np.arange(H)[:, None] - H / 2) ** 2) /
                  ((W / 2) ** 2 + (H / 2) ** 2)) ** 1.1
vig = vig[..., None].astype(np.float32)

# ---------------- figure ----------------
SK = (7, 10, 19, 255)
PAL = dict(skin=(150, 102, 72, 255), hair=(22, 18, 20, 255), coat=(196, 146, 70, 255), coat_d=(150, 108, 50, 255),
           pants=(40, 48, 78, 255), pants_d=(28, 34, 58, 255), shoe=(20, 22, 34, 255), strap=(236, 196, 120, 255),
           eye=(24, 20, 22, 255))


def figure_img(phase, sc, style):
    S, CW, CH = 3, 600, 560
    im = Image.new("RGBA", (CW * S, CH * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    gx, gy = 330, 480
    sil = style == "sil"

    def Pt(x, y):
        return ((gx + x * sc) * S, (gy + y * sc) * S)

    def col(name, dark=False):
        if sil:
            return SK
        return PAL[name + "_d"] if dark and (name + "_d") in PAL else PAL[name]

    def limb(p0, p1, w, c):
        d.line([Pt(*p0), Pt(*p1)], fill=c, width=max(int(w * sc * S), 1))
        r = w * sc * S / 2
        for p in (p0, p1):
            cx, cy = Pt(*p)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)

    bob = 4 * abs(math.sin(phase))
    hip = (0.0, -170.0 + bob)
    lean = math.radians(13)
    ax, ay = math.sin(lean), -math.cos(lean)
    nx, ny = math.cos(lean), math.sin(lean)
    neck = (hip[0] + 100 * ax, hip[1] + 100 * ay)
    sho = (hip[0] + 88 * ax, hip[1] + 88 * ay)
    head = (neck[0] + 24 * ax + 11, neck[1] + 24 * ay + 7)

    # legs (back leg first)
    legs = []
    for i, ph in enumerate((phase + math.pi, phase)):
        th = math.radians(30 * math.sin(ph))
        bend = math.radians(38 * max(0.0, math.sin(ph + 1.3)))
        knee = (hip[0] + 85 * math.sin(th), hip[1] + 85 * math.cos(th))
        sh = th - bend
        foot = (knee[0] + 85 * math.sin(sh), knee[1] + 85 * math.cos(sh))
        legs.append((knee, foot))
    for i, (knee, foot) in enumerate(legs):
        dark = i == 0
        limb(hip, knee, 31, col("pants", dark))
        limb(knee, foot, 21, col("pants", dark))
        limb(foot, (foot[0] + 22, foot[1] + 2), 16, col("shoe"))

    # back arm holds the strap
    limb(sho, (sho[0] - 26, sho[1] + 30), 19, col("coat", True))
    limb((sho[0] - 26, sho[1] + 30), (sho[0] - 50, sho[1] + 14), 15, col("skin") if not sil else SK)

    # torso: a rounded capsule
    limb((hip[0] + ax * 6, hip[1] + ay * 6), sho, 64, col("coat"))
    limb(sho, head, 19, col("skin") if not sil else SK)

    # front arm swings
    sw = math.radians(20 * math.sin(phase + math.pi))
    elbow = (sho[0] + 44 * math.sin(sw), sho[1] + 44 * math.cos(sw))
    hand = (elbow[0] + 40 * math.sin(sw * 0.6 - 0.15), elbow[1] + 40 * math.cos(sw * 0.6 - 0.15))
    limb(sho, elbow, 19, col("coat"))
    limb(elbow, hand, 15, col("skin") if not sil else SK)

    # head
    r = 29 * sc * S
    cx, cy = Pt(*head)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col("skin"))
    if not sil:
        d.pieslice([cx - r * 1.02, cy - r * 1.12, cx + r * 1.02, cy + r * 0.96], 168, 352, fill=PAL["hair"])
        for ex in (4, 18):
            ex_, ey_ = Pt(head[0] + ex, head[1] - 3)
            rr = 3.4 * sc * S
            d.ellipse([ex_ - rr, ey_ - rr, ex_ + rr, ey_ + rr], fill=PAL["eye"])
        d.line([Pt(head[0] + 1, head[1] - 11), Pt(head[0] + 9, head[1] - 14)], fill=PAL["eye"], width=max(int(2.6 * sc * S), 1))
        d.line([Pt(head[0] + 15, head[1] - 14), Pt(head[0] + 23, head[1] - 11)], fill=PAL["eye"], width=max(int(2.6 * sc * S), 1))
        d.line([Pt(head[0] + 8, head[1] + 13), Pt(head[0] + 20, head[1] + 12)], fill=PAL["eye"], width=max(int(3.0 * sc * S), 1))

    img = im.resize((CW, CH), Image.LANCZOS)
    arr = np.asarray(img, np.float32) / 255.0
    # warm rim light from the orb on the side facing it
    m = arr[..., 3] > 0.5
    rim = (m & ~np.roll(m, 6, axis=1)).astype(np.float32)
    rim = cv2.GaussianBlur(rim, (0, 0), 1.3)[..., None]
    k = 0.85 if sil else 0.40
    arr[..., :3] = arr[..., :3] * (1 - rim * k) + G * rim * k
    orb = (gx + (neck[0] - 95) * sc, gy + (hip[1] - 100) * sc)
    return arr, (gx, gy), orb


# orb sprite
R0 = 82
yy_, xx_ = np.mgrid[-R0 - 6:R0 + 7, -R0 - 6:R0 + 7].astype(np.float32)
rr = np.sqrt(xx_ ** 2 + yy_ ** 2) / R0
orb_a = np.clip(1.15 - rr ** 3.0, 0, 1)
orb_rgb = (np.clip(1 - rr, 0, 1)[..., None] ** 0.8) * np.array([1.0, 0.95, 0.80]) + \
          (1 - np.clip(1 - rr, 0, 1)[..., None] ** 0.8) * np.array([0.84, 0.62, 0.28])
G_R = 430
gy_, gx_ = np.mgrid[-G_R:G_R + 1, -G_R:G_R + 1].astype(np.float32)
glow_spr = np.exp(-((gx_ ** 2 + gy_ ** 2) / (G_R * 0.42) ** 2)).astype(np.float32)


def add_spr(frame, spr, cx, cy, tint, amt):
    h, w = spr.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    x1, y1, x2, y2 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if x2 <= x1 or y2 <= y1:
        return
    s = spr[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
    frame[y1:y2, x1:x2] += s[..., None] * tint * amt


# ---------------- text ----------------
def block_mask(text, fname, size):
    f = font(fname, size, 500)
    lines = layout_words(text, f, 860)
    lh = size * 1.18
    pad = 50
    h = int(lh * len(lines) + 2 * pad)
    im = Image.new("L", (W, h), 0)
    d = ImageDraw.Draw(im)
    base = pad + size * 0.95
    sp = f.getlength(" ")
    for ln, cw in lines:
        x = (W - cw) / 2
        for w_, wl in ln:
            d.text((x, base), w_, font=f, fill=255, anchor="ls")
            x += wl + sp
        base += lh
    return np.asarray(im, np.float32) / 255.0, h


T1 = block_mask("You're carrying a lot.", "CormorantGaramond", 104)
T2 = block_mask("Mostly quietly.", "CormorantGaramond-Italic", 104)

# dust
rnd = random.Random(5)
dust = []
for _ in range(120):
    spr, s = sprite(rnd.uniform(1.3, 3.0))
    dust.append(dict(x=rnd.uniform(0, W), y=rnd.uniform(0, H), sp=rnd.uniform(14, 50), ph=rnd.uniform(0, 6.28),
                     tw=rnd.uniform(0, 6.28), spr=spr, s=s))

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
       "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", SILENT]
pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
FX0, FY0 = 490, 1585

for i in range(int(T * FPS)):
    t = i / FPS
    frame = sky.copy()
    for (rgb, a), spd in LAYERS:
        ox = int((spd * t) % 2160)
        reg = frame[LTOP:H]
        sl_a = a[:, ox:ox + W]
        reg[:] = reg * (1 - sl_a) + rgb[:, ox:ox + W] * sl_a

    pull = ss((t - t2) / (T - t2))
    sc = 1.2 * (1.0 - 0.22 * pull)
    fx, fy = FX0, FY0 - (1.2 - sc) * 70
    phase = t * 2.4
    pulse = 0.93 + 0.07 * math.sin(t * 2.6)
    fig, (gx, gy), (ox_, oy_) = figure_img(phase, sc, STYLE)
    orb_x, orb_y = fx + (ox_ - gx), fy + (oy_ - gy)

    # light from the weight: on the world, the ground and the figure
    gl = cv2.resize(glow_spr, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
    add_spr(frame, gl, orb_x, orb_y, G, 0.34 * pulse)
    add_spr(frame, gl[:int(gl.shape[0] * 0.5)] * 0, orb_x, orb_y, G, 0)  # keep shapes simple
    floor = cv2.resize(glow_spr, None, fx=sc * 0.9, fy=sc * 0.16, interpolation=cv2.INTER_LINEAR)
    add_spr(frame, floor, orb_x + 40, fy - 6, G, 0.42 * pulse)

    # the weight itself, behind the figure
    s_ = sc * (0.96 + 0.04 * pulse)
    o_a = cv2.resize(orb_a, None, fx=s_, fy=s_, interpolation=cv2.INTER_LINEAR)
    o_c = cv2.resize(orb_rgb.astype(np.float32), None, fx=s_, fy=s_, interpolation=cv2.INTER_LINEAR)
    h_, w_ = o_a.shape
    x0, y0 = int(orb_x - w_ / 2), int(orb_y - h_ / 2)
    frame[y0:y0 + h_, x0:x0 + w_] = frame[y0:y0 + h_, x0:x0 + w_] * (1 - o_a[..., None]) + o_c * o_a[..., None]

    # figure
    fh, fw = fig.shape[:2]
    fx0, fy0 = int(fx - gx), int(fy - gy)
    xa, ya, xb, yb = max(fx0, 0), max(fy0, 0), min(fx0 + fw, W), min(fy0 + fh, H)
    sub = fig[ya - fy0:yb - fy0, xa - fx0:xb - fx0]
    al = sub[..., 3:4]
    frame[ya:yb, xa:xb] = frame[ya:yb, xa:xb] * (1 - al) + sub[..., :3] * al

    # dust
    fade = ss(t / 1.5)
    for p in dust:
        x = (p["x"] - p["sp"] * 0.9 * t) % (W + 60) - 30
        y = p["y"] - 8 * t + 12 * math.sin(p["ph"] + t * 0.6)
        a_ = (0.28 + 0.28 * math.sin(p["tw"] + t * 1.7)) * fade
        s = p["s"]
        xi, yi = int(x) - s // 2, int(y) - s // 2
        if 0 <= xi < W - s and 0 <= yi < H - s:
            frame[yi:yi + s, xi:xi + s] += p["spr"][..., None] * G * (a_ * 1.5)

    # words, timed to the voice
    for (m, h), c, s0, e in ((T1, P, t1 - 0.12, t2 - 0.1), (T2, G, t2 - 0.12, 99)):
        pin = ss((t - s0) / 0.6)
        pout = 1 - ss((t - (e - 0.4)) / 0.4) if e < 90 else 1
        op = pin * pout
        if op > 0:
            mm = m
            sg = (1 - pin) * 8
            if sg > 0.4:
                mm = cv2.GaussianBlur(m, (0, 0), sg)
            y = int(470 - h / 2 + (1 - pin) * 20)
            reg = frame[y:y + h]
            reg[:] = reg * (1 - (mm * op)[..., None]) + c * (mm * op)[..., None]

    out = np.clip(frame * vig, 0, 1)
    o8 = (out * 255 + 0.5).astype(np.uint8)
    if i in (int(2.0 * FPS), int((T - 0.4) * FPS)):
        Image.fromarray(o8).save(f"/home/claude/social/storystill_{STYLE}_{i}.png")
    pipe.stdin.write(o8.tobytes())

pipe.stdin.close()
pipe.wait()
af = ("asetrate=24000*0.985,atempo=1.0152,aresample=24000,equalizer=f=170:t=q:w=1:g=3.5,equalizer=f=3800:t=q:w=1:g=-2,"
      "acompressor=threshold=-20dB:ratio=2.5:attack=15:release=180:makeup=2,aecho=0.8:0.7:45:0.10")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", SILENT, "-i", "/home/claude/voice/story_voice_raw.wav",
                "-af", af, "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", FINAL], check=True)
print("done", FINAL)
