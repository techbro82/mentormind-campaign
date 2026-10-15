import sys, math, random, subprocess, json
import numpy as np
import cv2
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/claude/social")
from social_lib import font, ss, sprite, layout_words, GOLD, PARCH

SRC = "/mnt/project/Person_carrying_glowing_orb_2K_20261008115742.jpg"
W, H, FPS = 1080, 1920, 30
V = json.load(open("/home/claude/voice/story_voice.json"))
t1, t2, T = V["t1"], V["t2"], V["T"]
SILENT = "/home/claude/social/story_photo_silent.mp4"
FINAL = "/home/claude/social/story_photo.mp4"
G = np.array(GOLD, np.float32) / 255
P = np.array(PARCH, np.float32) / 255

# ---- base image: upscale once, sharpen a little
img = cv2.cvtColor(cv2.imread(SRC), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
UP = 1.6
base = cv2.resize(img, None, fx=UP, fy=UP, interpolation=cv2.INTER_LANCZOS4)
blur = cv2.GaussianBlur(base, (0, 0), 1.4)
base = np.clip(base + 0.30 * (base - blur), 0, 1)
Bh, Bw = base.shape[:2]
f0 = H / Bh                                     # scale that fits the image height to the frame

# orb mask (warm, bright region around the moon) for the breathing glow
yy, xx = np.mgrid[0:Bh, 0:Bw].astype(np.float32)
lum = base @ np.array([0.299, 0.587, 0.114], np.float32)
circ = np.exp(-(((xx / Bw - 0.455) / 0.19) ** 2 + ((yy / Bh - 0.45) / 0.105) ** 2) ** 1.6)
orb_mask = np.clip((lum - 0.50) / 0.30, 0, 1) * circ
bloom = cv2.GaussianBlur(orb_mask, (0, 0), 55)
WIN = (int(0.10 * Bh), int(0.80 * Bh), int(0.05 * Bw), int(0.95 * Bw))   # y0,y1,x0,x1 window for per-frame light changes
tint = np.array([1.0, 0.80, 0.46], np.float32)

U, Vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

# ---- mist texture
rs = np.random.RandomState(4)
fog = np.zeros((220, 900), np.float32)
for sc, amp in ((6, 1.0), (14, 0.5), (30, 0.25)):
    n = rs.rand(220 // sc + 2, 900 // sc + 2).astype(np.float32)
    fog += amp * cv2.resize(n, (900, 220), interpolation=cv2.INTER_CUBIC)
fog = (fog - fog.min()) / (fog.max() - fog.min())
fog = np.clip((fog - 0.35) / 0.5, 0, 1)
fog = np.tile(fog, (1, 3))                      # 2700 wide, scrolls horizontally

# ---- words
def block_mask(text, fname, size):
    f = font(fname, size, 500)
    lines = layout_words(text, f, 860)
    lh = size * 1.18
    pad = 50
    h = int(lh * len(lines) + 2 * pad)
    im = Image.new("L", (W, h), 0)
    d = ImageDraw.Draw(im)
    base_y = pad + size * 0.95
    sp = f.getlength(" ")
    for ln, cw in lines:
        x = (W - cw) / 2
        for w_, wl in ln:
            d.text((x, base_y), w_, font=f, fill=255, anchor="ls")
            x += wl + sp
        base_y += lh
    return np.asarray(im, np.float32) / 255.0, h


T1 = block_mask("You're carrying a lot.", "CormorantGaramond", 104)
T2 = block_mask("Mostly quietly.", "CormorantGaramond-Italic", 104)

# ---- dust
rnd = random.Random(8)
dust = []
for _ in range(90):
    spr, s = sprite(rnd.uniform(1.3, 3.0))
    dust.append(dict(x=rnd.uniform(0, W), y=rnd.uniform(0, H), sp=rnd.uniform(10, 38), ph=rnd.uniform(0, 6.28),
                     tw=rnd.uniform(0, 6.28), spr=spr, s=s))

top_dark = 1 - 0.38 * (1 - np.clip(np.arange(H, dtype=np.float32) / 760.0, 0, 1)) ** 1.5
top_dark = top_dark[:, None, None]
vig = 1 - 0.30 * (((np.arange(W)[None, :] - W / 2) ** 2 + (np.arange(H)[:, None] - H / 2) ** 2) /
                  ((W / 2) ** 2 + (H / 2) ** 2)) ** 1.1
vig = vig[..., None].astype(np.float32)
fog_mask = np.exp(-(((np.arange(H, dtype=np.float32) - 1010) / 190) ** 2))[:, None]

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
       "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p", SILENT]
pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
rng = np.random.RandomState(1)

for i in range(int(T * FPS)):
    t = i / FPS
    # camera: slow push toward the person on the first line, then ease back to reveal how big it is
    push = ss(t / t2)
    pull = ss((t - t2) / (T - t2))
    z = 1.04 + 0.13 * push - 0.10 * pull
    cx = Bw * (0.500 - 0.020 * (push - pull))
    cy = Bh * (0.500 - 0.012 * (push - pull))

    zmap = z * (1 + 0.045 * (Vv / H - 0.5))          # a little depth: the top moves less than the bottom
    x = cx + (U - W / 2) / (f0 * zmap)
    y = cy + (Vv - H / 2) / (f0 * zmap)

    # wind in the grass, kept away from the person
    wy = np.clip((y / Bh - 0.60) / 0.22, 0, 1)
    pm = np.exp(-(((x / Bw - 0.52) / 0.19) ** 2 + ((y / Bh - 0.60) / 0.19) ** 2) ** 1.4)
    x = x + 3.2 * wy * (1 - pm) * np.sin(2 * np.pi * (y / 70.0) + 1.7 * t + x / 140.0)

    # the weight breathes
    pulse = 0.5 + 0.5 * math.sin(t * 2.4)
    y0, y1, x0, x1 = WIN
    bt = base.copy()
    sub = bt[y0:y1, x0:x1]
    add = (0.07 + 0.05 * pulse) * orb_mask[y0:y1, x0:x1, None] + (0.10 + 0.10 * pulse) * bloom[y0:y1, x0:x1, None]
    sub[:] = np.clip(sub + add * tint, 0, 1)

    frame = cv2.remap(bt, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # drifting mist in the valley
    off = int((t * 14) % 900)
    fl = cv2.resize(fog[:, off:off + 900], (W, 420), interpolation=cv2.INTER_LINEAR)
    fb = np.zeros((H, W), np.float32)
    fb[800:1220] = fl
    a = (fb * fog_mask * 0.20)[..., None]
    frame = frame * (1 - a) + np.array([0.55, 0.62, 0.74], np.float32) * a

    # dust
    fade = ss(t / 1.4)
    for p in dust:
        dx = (p["x"] - p["sp"] * 0.8 * t) % (W + 60) - 30
        dy = p["y"] - 9 * t + 12 * math.sin(p["ph"] + t * 0.6)
        al = (0.22 + 0.22 * math.sin(p["tw"] + t * 1.7)) * fade
        s = p["s"]
        xi, yi = int(dx) - s // 2, int(dy) - s // 2
        if 0 <= xi < W - s and 0 <= yi < H - s:
            frame[yi:yi + s, xi:xi + s] += p["spr"][..., None] * G * (al * 1.4)

    frame = frame * top_dark * vig

    # the words, timed to the voice
    for (m, h), c, s0, e in ((T1, P, t1 - 0.12, t2 - 0.1), (T2, G, t2 - 0.12, 99)):
        pin = ss((t - s0) / 0.6)
        pout = 1 - ss((t - (e - 0.4)) / 0.4) if e < 90 else 1
        op = pin * pout
        if op > 0:
            mm = m
            sg = (1 - pin) * 8
            if sg > 0.4:
                mm = cv2.GaussianBlur(m, (0, 0), sg)
            yv = int(430 - h / 2 + (1 - pin) * 20)
            reg = frame[yv:yv + h]
            reg[:] = reg * (1 - (mm * op)[..., None]) + c * (mm * op)[..., None]

    frame = frame + rng.normal(0, 0.007, (H, W, 1)).astype(np.float32)      # film grain
    o8 = (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
    if i in (int(2.0 * FPS), int((T - 0.4) * FPS)):
        Image.fromarray(o8).save(f"/home/claude/social/storystill_photo_{i}.png")
    pipe.stdin.write(o8.tobytes())

pipe.stdin.close()
pipe.wait()
af = ("asetrate=24000*0.985,atempo=1.0152,aresample=24000,equalizer=f=170:t=q:w=1:g=3.5,equalizer=f=3800:t=q:w=1:g=-2,"
      "acompressor=threshold=-20dB:ratio=2.5:attack=15:release=180:makeup=2,aecho=0.8:0.7:45:0.10")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", SILENT, "-i", "/home/claude/voice/story_voice_raw.wav",
                "-af", af, "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", FINAL], check=True)
print("done", FINAL)
