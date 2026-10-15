import sys, json, math, random, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/claude/social")
import render_feeling as rf
from social_lib import font, ss, glow_layer, sprite, GOLD, PARCH, PARCH_DIM

W, H, FPS = 1080, 1920, 24
F32 = np.float32
G = np.array(GOLD, F32) / 255
P = np.array(PARCH, F32) / 255
PD = np.array(PARCH_DIM, F32) / 255
V = json.load(open("/home/claude/voice/day05_voice.json"))
s, d = V["starts"], V["durs"]
t_close = s[-1] + d[-1] + 1.0
T = t_close + 4.2
OUT_SILENT = "/home/claude/social/_silent_5.mp4"
OUT_FINAL = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/social/day05_final.mp4"
STILLS = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else [])]

# ---------------------------------------------------------------- the real screens
IW, IH = 524, 1128                      # inner screen size inside the phone
PW, PH = 560, 1190
SX = IW / 728.0


def load(p):
    return Image.open(p).convert("RGB")


S04 = load("/mnt/project/screen-04-onboarding-struggle.png")
S06 = load("/mnt/project/screen-06-mentor-recommendation.png")
S11 = load("/mnt/project/screen-11-empty-chat.png")
arr = lambda im: np.asarray(im.resize((IW, IH), Image.LANCZOS), F32) / 255.0
a04, a06 = arr(S04), arr(S06)

# the answer she taps, in the screenshot's own coordinates
ROW = (67, 860, 657, 1033)              # "Dealing with people" (it lists loneliness)
TAP_T = s[1] + d[1] - 0.25
T_46 = TAP_T + 0.55                     # the screen changes after the tap lands
T_611 = s[4] - 0.3

FIELD_COL = S11.getpixel((300, 408))
F_TYPE = font("Inter", 29, 400)
TRUTH = "Honestly, not great"
TYPE_T0, TYPE_DUR = s[5], 1.5
PULSE_T = s[6] + d[6] - 0.55


def screen04(t):
    img = S04.copy()
    return img


def tap_overlay(t):
    """Gold highlight on the chosen row and a ripple, drawn on the inner-size screen."""
    ov = Image.new("RGBA", (IW, IH), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    k = ss((t - TAP_T) / 0.25)
    if k > 0:
        x0, y0, x1, y1 = [v * SX for v in ROW]
        dr.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=GOLD + (int(46 * k),), outline=GOLD + (int(255 * k),), width=3)
        rp = ss((t - TAP_T) / 0.55)
        if rp < 1:
            cx, cy = 362 * SX, 946 * SX
            r = 12 + 70 * rp
            dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=GOLD + (int(220 * (1 - rp)),), width=4)
            dr.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=(255, 236, 190, int(230 * (1 - rp))))
    return np.asarray(ov, F32) / 255.0


def screen11(t):
    img = S11.copy()
    dr = ImageDraw.Draw(img)
    typing = t >= TYPE_T0
    n = int(np.clip((t - TYPE_T0) / TYPE_DUR, 0, 1) * len(TRUTH)) if typing else 0
    txt = TRUTH[:n]
    if typing:
        dr.rectangle([82, 410, 418, 476], fill=FIELD_COL)
        dr.text((97, 443), txt, font=F_TYPE, fill=(236, 227, 210), anchor="lm")
        xcur = 97 + F_TYPE.getlength(txt) + 4
    else:
        xcur = 90
    if (t * 1.5) % 1.0 < 0.62 or (typing and t < TYPE_T0 + TYPE_DUR):
        dr.rectangle([xcur, 420, xcur + 3, 466], fill=GOLD)
    if t >= PULSE_T:
        kk = math.sin(math.pi * np.clip((t - PULSE_T) / 0.6, 0, 1))
        box = (495, 401, 658, 487)
        reg = np.asarray(img.crop(box), F32)
        m = reg[..., 0] > 120
        reg[m] = reg[m] * (1 - 0.45 * kk) + np.array([255, 224, 160], F32) * 0.45 * kk
        img.paste(Image.fromarray(np.clip(reg, 0, 255).astype(np.uint8)), box[:2])
    return np.asarray(img.resize((IW, IH), Image.LANCZOS), F32) / 255.0


def screen_at(t):
    a46 = ss((t - T_46) / 0.6)
    a611 = ss((t - T_611) / 0.6)
    base = a04
    if t < T_46 + 0.7:
        ov = tap_overlay(t)
        base = a04 * (1 - ov[..., 3:4]) + ov[..., :3] * ov[..., 3:4]
    img = base * (1 - a46) + a06 * a46
    if a611 > 0:
        img = img * (1 - a611) + screen11(t) * a611
    return img


# ---------------------------------------------------------------- the phone
def make_bezel():
    S = 2
    im = Image.new("RGBA", (PW * S, PH * S), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    dr.rounded_rectangle([0, 0, PW * S - 1, PH * S - 1], radius=66 * S, fill=(20, 22, 30, 255), outline=(168, 132, 78, 255), width=3 * S)
    dr.rounded_rectangle([7 * S, 7 * S, (PW - 7) * S, (PH - 7) * S], radius=60 * S, outline=(46, 50, 64, 255), width=2 * S)
    dr.rounded_rectangle([PW * S / 2 - 36 * S, 13 * S, PW * S / 2 + 36 * S, 27 * S], radius=7 * S, fill=(9, 10, 15, 255))
    return np.asarray(im.convert("RGBa").resize((PW, PH), Image.LANCZOS).convert("RGBA"), F32) / 255.0


BEZEL = make_bezel()
mk = Image.new("L", (IW * 2, IH * 2), 0)
ImageDraw.Draw(mk).rounded_rectangle([0, 0, IW * 2 - 1, IH * 2 - 1], radius=40 * 2, fill=255)
SCR_MASK = (np.asarray(mk.resize((IW, IH), Image.LANCZOS), F32) / 255.0)[..., None]
OX, OY = 18, 34


def phone_rgba(screen):
    ph = BEZEL.copy()
    reg = ph[OY:OY + IH, OX:OX + IW, :3]
    ph[OY:OY + IH, OX:OX + IW, :3] = reg * (1 - SCR_MASK) + screen * SCR_MASK
    return ph


def warp_phone(ph, yaw, pitch, cx, cy, sc, alpha):
    pre = np.dstack([ph[..., :3] * ph[..., 3:4], ph[..., 3:4]]).astype(F32)
    Dz = 1900.0
    corners = [(-PW / 2, -PH / 2), (PW / 2, -PH / 2), (PW / 2, PH / 2), (-PW / 2, PH / 2)]
    dst = []
    for x, y in corners:
        x1, z1 = x * math.cos(yaw), -x * math.sin(yaw)
        y2 = y * math.cos(pitch) - z1 * math.sin(pitch)
        z2 = y * math.sin(pitch) + z1 * math.cos(pitch)
        f = Dz / (Dz - z2)
        dst.append((cx + x1 * f * sc, cy + y2 * f * sc))
    src = np.array([(0, 0), (PW, 0), (PW, PH), (0, PH)], F32)
    M = cv2.getPerspectiveTransform(src, np.array(dst, F32))
    out = cv2.warpPerspective(pre, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return out[..., :3] * alpha, out[..., 3:4] * alpha


# ---------------------------------------------------------------- the rest of the frame
base_bg = glow_layer(W, H, [(540, 1100, 640, 0.20, GOLD), (800, 260, 640, 0.17, (63, 90, 140)), (120, 1500, 560, 0.22, (40, 52, 80))])
close_bg = glow_layer(W, H, [(540, 1250, 520, 0.28, GOLD), (800, 260, 640, 0.17, (63, 90, 140)), (120, 1600, 520, 0.22, (40, 52, 80))])
vig = (1 - 0.30 * (((np.arange(W)[None, :] - W / 2) ** 2 + (np.arange(H)[:, None] - H / 2) ** 2) /
                   ((W / 2) ** 2 + (H / 2) ** 2)) ** 1.1)[..., None].astype(F32)
GR = 520
gy_, gx_ = np.mgrid[-GR:GR + 1, -GR:GR + 1].astype(F32)
gspr = np.exp(-((gx_ ** 2 + gy_ ** 2) / (GR * 0.42) ** 2)).astype(F32)


def add_glow(frame, cx, cy, R, amt):
    sp_ = cv2.resize(gspr, (2 * R, 2 * R), interpolation=cv2.INTER_LINEAR)
    x0, y0 = int(cx - R), int(cy - R)
    x1, y1, x2, y2 = max(x0, 0), max(y0, 0), min(x0 + 2 * R, W), min(y0 + 2 * R, H)
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] += sp_[y1 - y0:y2 - y0, x1 - x0:x2 - x0, None] * np.array([1.0, 0.86, 0.6], F32) * amt


rnd = random.Random(55)
dust = []
for _ in range(110):
    spr, sz = sprite(rnd.uniform(1.3, 3.0))
    dust.append(dict(x=rnd.uniform(0, W), y=rnd.uniform(0, H), sp=rnd.uniform(14, 50), ph=rnd.uniform(0, 6.28),
                     tw=rnd.uniform(0, 6.28), spr=spr, s=sz))

LINES = [("You say, \u201cI'm good,\u201d because who would understand?", False, 215),
         ("But people have carried this before, and they wrote it down.", False, 215),
         ("Now you can talk it through with AI mentors, built from what they wrote.", True, 300)]
WIN = [(s[0] - 0.12, s[2] - 0.10), (s[2] - 0.12, s[4] - 0.10), (s[4] - 0.12, t_close - 0.05)]
blocks = []
for txt, gold, mh in LINES:
    m, h, *_ = rf.block_mask(txt, "CormorantGaramond-Italic" if gold else "CormorantGaramond", W, mh)
    blocks.append((m, h, G if gold else P))

bug_w = 240
bug_arr, bug_ox, bug_oy = rf.lock_scaled(bug_w)
bug_sh = cv2.GaussianBlur(bug_arr.max(axis=2), (0, 0), 7)
bug_cx, bug_cy = 62 + bug_w / 2, 1405
hm, hb = rf.tmask("@mentormindhq", rf.font("Inter", 23, 500), 3)
hm_sh = cv2.GaussianBlur(hm, (0, 0), 5)
hx, hy = 62 - 40, int(bug_cy + 62 - hb)

LS, lox, loy = rf.lock_scaled(700)
tag_m, tag_b = rf.tmask("Opening soon.", rf.font("CormorantGaramond-Italic", 64, 500), 0)
cta_m, cta_b = rf.tmask("FOLLOW FOR LAUNCH DAY", rf.font("Inter", 34, 500), 8)
XF2 = 0.9
rng = np.random.RandomState(6)

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
       "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", OUT_SILENT]
pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
still_idx = {int(round(x * FPS)): x for x in STILLS}

for i in range(int(T * FPS)):
    t = i / FPS
    frame = base_bg.copy()
    flat = ss((t - (s[4] + 0.6)) / 1.6)                         # the phone settles to face us for the last part
    intro = ss(t / 1.4)
    yaw = math.radians((-34 * (1 - intro) + 11 * math.sin(0.55 * t + 0.4)) * (1 - flat))
    pitch = math.radians((5 * math.sin(0.7 * t) + 8 * (1 - intro)) * (1 - flat))
    cy = 1135 + 12 * math.sin(1.1 * t) * (1 - flat) + 300 * (1 - intro)
    sc = (0.92 + 0.08 * intro + 0.03 * flat) * 1.10
    add_glow(frame, 552, cy, 640, 0.10 + 0.20 * ss((t - T_611) / 1.5))

    rgb, al = warp_phone(phone_rgba(screen_at(t)), yaw, pitch, 552, cy, sc, ss(t / 0.8))
    sh = cv2.resize(al[..., 0], (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    sh = np.roll(cv2.resize(cv2.GaussianBlur(sh, (0, 0), 8), (W, H), interpolation=cv2.INTER_LINEAR), 40, axis=0)
    frame = frame * (1 - 0.5 * sh[..., None])
    frame = frame * (1 - al) + rgb

    fade = ss(t / 1.2)
    for p in dust:
        x = (p["x"] - p["sp"] * 0.8 * t) % (W + 60) - 30
        y = p["y"] - 9 * t + 12 * math.sin(p["ph"] + t * 0.6)
        a_ = (0.20 + 0.20 * math.sin(p["tw"] + t * 1.7)) * fade
        sz = p["s"]
        xi, yi = int(x) - sz // 2, int(y) - sz // 2
        if 0 <= xi < W - sz and 0 <= yi < H - sz:
            frame[yi:yi + sz, xi:xi + sz] += p["spr"][..., None] * G * (a_ * 1.3)

    bo = ss(t / 0.8) * 0.92
    h_, w_ = bug_arr.shape[:2]
    x0, y0 = int(bug_cx - bug_ox - w_ / 2), int(bug_cy - bug_oy - h_ / 2)
    x1, y1, x2, y2 = max(x0, 0), max(y0, 0), min(x0 + w_, W), min(y0 + h_, H)
    frame[y1:y2, x1:x2] *= (1 - 0.55 * bo * bug_sh[y1 - y0:y2 - y0, x1 - x0:x2 - x0, None])
    rf.add_light(frame, bug_arr, bug_ox, bug_oy, bug_cx, bug_cy, bo)
    frame[hy:hy + hm_sh.shape[0], hx:hx + hm_sh.shape[1]] *= (1 - 0.5 * bo * hm_sh[..., None])
    rf.blend(frame, hm, P, hx, hy, bo * 0.9)

    ca = ss((t - (t_close - XF2 / 2)) / XF2)
    if ca > 0:
        cb = close_bg.copy()
        rf.add_light(cb, LS, lox, loy, 540, 640, 1.0)
        for p in dust:
            x = (p["x"] - p["sp"] * 0.8 * t) % (W + 60) - 30
            y = (p["y"] - 12 * t) % H
            sz = p["s"]
            xi, yi = int(x) - sz // 2, int(y) - sz // 2
            if 0 <= xi < W - sz and 0 <= yi < H - sz:
                cb[yi:yi + sz, xi:xi + sz] += p["spr"][..., None] * G * 0.35
        for (m_, b_, col, base_y, t_on) in ((tag_m, tag_b, P, 830, t_close + 0.8), (cta_m, cta_b, G, 960, t_close + 1.7)):
            pc = ss((t - t_on) / 0.7)
            if pc > 0:
                rf.blend(cb, m_, col, (W - m_.shape[1]) // 2, int(base_y - b_ + (1 - pc) * 12), pc)
        frame = frame * (1 - ca) + cb * ca

    for (m, h, c), (w0, w1) in zip(blocks, WIN):
        pin = ss((t - w0) / 0.55)
        pout = 1 - ss((t - (w1 - 0.35)) / 0.35)
        op = pin * pout
        if op > 0:
            mm = m
            sg = (1 - pin) * 8
            if sg > 0.4:
                mm = cv2.GaussianBlur(m, (0, 0), sg)
            yv = int(255 - h / 2 + (1 - pin) * 20)
            reg = frame[yv:yv + h]
            reg[:] = reg * (1 - (mm * op)[..., None]) + c * (mm * op)[..., None]

    frame = frame * vig + rng.normal(0, 0.006, (H, W, 1)).astype(F32)
    o8 = (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
    if i in still_idx:
        Image.fromarray(o8).save(f"/home/claude/social/_still_5_{still_idx[i]:.2f}.png")
    pipe.stdin.write(o8.tobytes())

pipe.stdin.close()
pipe.wait()
af = "highpass=f=70,equalizer=f=170:t=q:w=1:g=3,equalizer=f=3800:t=q:w=1:g=-1.5,acompressor=threshold=-20dB:ratio=2.5:attack=15:release=180:makeup=2"
wind = f"lowpass=f=800,highpass=f=70,volume=0.05,afade=t=in:d=1.2,afade=t=out:st={max(T - 1.6, 0):.2f}:d=1.6"
fc = f"[1:a]{af}[v];[2:a]{wind}[w];[v][w]amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7,alimiter=limit=0.95[a]"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT_SILENT, "-i", "/home/claude/voice/day05_voice_raw.wav",
                "-f", "lavfi", "-i", f"anoisesrc=d={T:.2f}:c=pink:r=24000:a=0.5", "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", f"{T:.2f}", OUT_FINAL], check=True)
print("done", OUT_FINAL, f"({T:.1f}s)")
