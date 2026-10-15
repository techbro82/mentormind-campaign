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
V = json.load(open("/home/claude/voice/day04_slow.json"))
starts, durs = V["starts"], V["durs"]
t_close = starts[-1] + durs[-1] + 1.4          # a beat of stillness after the last line
XF2 = 0.9
T = t_close + 5.8

OUT_SILENT = "/home/claude/social/_silent_4.mp4"
OUT_FINAL = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/social/day04_final.mp4"
STILLS = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else [])]

# ------------------------------------------------ the chat card
S2 = 2
CW, CH = 880, 870
BG = (19, 24, 37)
INB = (38, 45, 64)
OUTB = GOLD
TXT = PARCH
AREA_T, AREA_B = 116, 726
BH, GAP = 100, 22
F_HEAD = font("Inter", 38 * S2, 600)
F_MSG = font("Inter", 34 * S2, 400)
F_MSG1 = font("Inter", 34, 400)
F_IN = font("Inter", 36 * S2, 400)
F_IN1 = font("Inter", 36, 400)
F_AV = font("Inter", 28 * S2, 600)
F_SEND = font("Inter", 34 * S2, 600)

MSGS = [
    dict(kind="in", who="A", col=(124, 96, 166), text="How are you?", t=0.65),
    dict(kind="in", who="K", col=(66, 128, 150), text="hey! how are you doing?", t=1.20),
    dict(kind="in", who="D", col=(164, 112, 80), text="How've you been?", t=1.75),
    dict(kind="in", who="M", col=(88, 142, 100), text="how are you?", t=2.30),
    dict(kind="in", who="T", col=(152, 90, 112), text="Hope you're well! How are you?", t=2.85),
    dict(kind="out", text="I'm good!", t=7.00),
    dict(kind="in", who="A", col=(124, 96, 166), text="Great!", t=7.55),
]
for m in MSGS:
    m["w"] = int(F_MSG1.getlength(m["text"])) + 60


def mix(c, a, bg=BG):
    return tuple(int(bg[i] + (c[i] - bg[i]) * a) for i in range(3))


def typed(t):
    """What is in the input box at time t, and whether she is typing."""
    TRUTH, LIE = "Honestly, not great", "I'm good!"
    if t < 3.4:
        return "", False
    if t < 4.8:
        return TRUTH[:int((t - 3.4) / 1.4 * len(TRUTH))], True
    if t < 5.5:
        return TRUTH, False                                            # she stops and looks at it
    if t < 6.1:
        return TRUTH[:int(len(TRUTH) * (1 - (t - 5.5) / 0.6))], True
    if t < 6.7:
        return LIE[:min(int((t - 6.1) / 0.6 * len(LIE)) + 1, len(LIE))], True
    if t < 7.0:
        return LIE, False
    return "", False


def draw_card(t):
    img = Image.new("RGBA", (CW * S2, CH * S2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    glow = ss((t - 8.0) / 1.4)                                   # the quiet box warms up at the end
    d.rounded_rectangle([0, 0, CW * S2 - 1, CH * S2 - 1], radius=46 * S2, fill=BG + (255,), outline=(46, 54, 74, 255), width=2 * S2)
    d.text((36 * S2, 28 * S2), "Chats", font=F_HEAD, fill=PARCH, anchor="la")
    d.line([(36 * S2, 102 * S2), ((CW - 36) * S2, 102 * S2)], fill=(36, 43, 60, 255), width=2 * S2)

    # messages, on their own layer so the top can fade out
    ah = AREA_B - AREA_T
    layer = Image.new("RGB", (CW * S2, ah * S2), BG)
    md = ImageDraw.Draw(layer)
    dim = 1 - 0.86 * ss((t - 7.6) / 0.9)
    prog = [ss((t - m["t"]) / 0.30) for m in MSGS]
    anchor = ah - 18
    for i, m in enumerate(MSGS):
        a = prog[i]
        if a <= 0:
            continue
        below = sum((BH + GAP) * prog[j] for j in range(i + 1, len(MSGS)))
        bottom = anchor - below + (1 - a) * (BH + GAP)          # a new bubble slides up from behind the input bar
        top = bottom - BH
        al = a * dim
        s = 0.92 + 0.08 * a
        w = m["w"] * s
        h = BH * s
        cy = top + BH / 2
        if m["kind"] == "in":
            x0 = 28 + 64 + 14
            col, tcol = mix(INB, al), mix(TXT, al)
            # avatar
            ax, ay, ar = 28 + 32, cy, 32 * s
            md.ellipse([(ax - ar) * S2, (ay - ar) * S2, (ax + ar) * S2, (ay + ar) * S2], fill=mix(m["col"], al))
            md.text((ax * S2, ay * S2), m["who"], font=F_AV, fill=mix((245, 240, 230), al), anchor="mm")
        else:
            x0 = CW - 28 - w
            col, tcol = mix(OUTB, al), mix((30, 22, 8), al)
        md.rounded_rectangle([x0 * S2, (cy - h / 2) * S2, (x0 + w) * S2, (cy + h / 2) * S2], radius=h / 2.6 * S2, fill=col)
        md.text(((x0 + w / 2) * S2, cy * S2), m["text"], font=F_MSG, fill=tcol, anchor="mm")
    mask = Image.new("L", (CW * S2, ah * S2), 255)
    fade = np.clip(np.arange(ah * S2) / (90.0 * S2), 0, 1)
    mask = Image.fromarray((np.tile(fade[:, None], (1, CW * S2)) * 255).astype(np.uint8))
    img.paste(layer, (0, AREA_T * S2), mask)

    # the input bar, drawn like the real app: a field with a microphone inside it, and a gold Send button
    FX0, FX1 = 28, CW - 28 - 168 - 20
    SX0, SX1 = CW - 28 - 168, CW - 28
    bc = tuple(int(c0 + (c1 - c0) * glow) for c0, c1 in zip((52, 62, 86), GOLD))
    d.rounded_rectangle([FX0 * S2, 742 * S2, FX1 * S2, 846 * S2], radius=26 * S2, fill=(27, 33, 49, 255),
                        outline=bc + (255,), width=int((2 + 2 * glow) * S2))
    s_, typing = typed(t)
    tx = FX0 + 36
    if s_:
        d.text((tx * S2, 794 * S2), s_, font=F_IN, fill=PARCH, anchor="lm")
        xcur = tx + F_IN1.getlength(s_) + 4
    else:
        d.text(((tx + 8) * S2, 794 * S2), "Say something...", font=F_IN, fill=(120, 124, 140, 255), anchor="lm")
        xcur = tx - 4
    if typing or ((t * (1.5 if glow < 0.5 else 0.9)) % 1.0 < 0.6):
        d.rectangle([xcur * S2, (794 - 22) * S2, (xcur + 3) * S2, (794 + 22) * S2], fill=GOLD + (255,))
    # the microphone, inside the field
    mx, my = FX1 - 48, 794
    mcol = (150, 144, 128, 255)
    d.rounded_rectangle([(mx - 9) * S2, (my - 24) * S2, (mx + 9) * S2, (my + 4) * S2], radius=9 * S2, outline=mcol, width=3 * S2)
    d.arc([(mx - 17) * S2, (my - 14) * S2, (mx + 17) * S2, (my + 14) * S2], 0, 180, fill=mcol, width=3 * S2)
    d.line([(mx * S2, (my + 14) * S2), (mx * S2, (my + 24) * S2)], fill=mcol, width=3 * S2)
    d.line([((mx - 10) * S2, (my + 24) * S2), ((mx + 10) * S2, (my + 24) * S2)], fill=mcol, width=3 * S2)
    # the gold Send button
    pulse = 1 + 0.08 * math.sin(math.pi * np.clip((t - 7.0) / 0.2, 0, 1)) if 7.0 <= t < 7.2 else 1
    bw, bh = 168 * pulse, 104 * pulse
    bcx = (SX0 + SX1) / 2
    bcol = GOLD if s_ else tuple(int(c * 0.86 + 20) for c in GOLD)
    d.rounded_rectangle([(bcx - bw / 2) * S2, (794 - bh / 2) * S2, (bcx + bw / 2) * S2, (794 + bh / 2) * S2], radius=26 * S2, fill=bcol + (255,))
    d.text((bcx * S2, 794 * S2), "Send", font=F_SEND, fill=(30, 22, 8, 255), anchor="mm")
    rgba = img.convert("RGBa").resize((CW, CH), Image.LANCZOS).convert("RGBA")
    return np.asarray(rgba, F32) / 255.0


# ------------------------------------------------ the rest of the frame
base_bg = glow_layer(W, H, [(540, 1150, 620, 0.20, GOLD), (800, 260, 640, 0.17, (63, 90, 140)), (120, 1500, 560, 0.22, (40, 52, 80))])
close_bg = glow_layer(W, H, [(540, 1250, 520, 0.28, GOLD), (800, 260, 640, 0.17, (63, 90, 140)), (120, 1600, 520, 0.22, (40, 52, 80))])
vig = (1 - 0.30 * (((np.arange(W)[None, :] - W / 2) ** 2 + (np.arange(H)[:, None] - H / 2) ** 2) /
                   ((W / 2) ** 2 + (H / 2) ** 2)) ** 1.1)[..., None].astype(F32)
GR = 520
gy_, gx_ = np.mgrid[-GR:GR + 1, -GR:GR + 1].astype(F32)
gspr = np.exp(-((gx_ ** 2 + gy_ ** 2) / (GR * 0.42) ** 2)).astype(F32)


def add_glow(frame, cx, cy, R, amt):
    s = cv2.resize(gspr, (2 * R, 2 * R), interpolation=cv2.INTER_LINEAR)
    x0, y0 = int(cx - R), int(cy - R)
    x1, y1, x2, y2 = max(x0, 0), max(y0, 0), min(x0 + 2 * R, W), min(y0 + 2 * R, H)
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] += s[y1 - y0:y2 - y0, x1 - x0:x2 - x0, None] * np.array([1.0, 0.86, 0.6], F32) * amt


rnd = random.Random(44)
dust = []
for _ in range(110):
    spr, s = sprite(rnd.uniform(1.3, 3.0))
    dust.append(dict(x=rnd.uniform(0, W), y=rnd.uniform(0, H), sp=rnd.uniform(14, 50), ph=rnd.uniform(0, 6.28),
                     tw=rnd.uniform(0, 6.28), spr=spr, s=s))

LINES = [("Everyone asks how you are.", False), ("Few wait for the answer.", False), ("You don't have to think alone.", True)]
blocks = []
for txt, gold in LINES:
    m, h, *_ = rf.block_mask(txt, "CormorantGaramond-Italic" if gold else "CormorantGaramond", W, 215)
    blocks.append((m, h, G if gold else P))

bug_w = 240
bug_arr, bug_ox, bug_oy = rf.lock_scaled(bug_w)
bug_sh = cv2.GaussianBlur(bug_arr.max(axis=2), (0, 0), 7)
bug_cx, bug_cy = 62 + bug_w / 2, 1405
hm, hb = rf.tmask("@mentormindhq", rf.font("Inter", 23, 500), 3)
hm_sh = cv2.GaussianBlur(hm, (0, 0), 5)
hx, hy = 62 - 40, int(bug_cy + 62 - hb)

LS, lox, loy = rf.lock_scaled(700)
tag_m, tag_b = rf.tmask("Something is coming.", rf.font("CormorantGaramond-Italic", 60, 500), 0)
sub_m, sub_b = rf.tmask("AI mentors, built from what the wisest people wrote.", rf.font("Inter", 31, 500), 0)
sm1, sb1 = rf.tmask("Not therapy. Not professional advice.", rf.font("Inter", 29, 500), 0)
sm2, sb2 = rf.tmask("In a crisis, contact local emergency services.", rf.font("Inter", 29, 500), 0)
cta_m, cta_b = rf.tmask("FOLLOW FOR LAUNCH DAY", rf.font("Inter", 34, 500), 8)
rng = np.random.RandomState(5)

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
       "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", OUT_SILENT]
pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
still_idx = {int(round(x * FPS)): x for x in STILLS}
IN_X, IN_Y = CW / 2, 794

for i in range(int(T * FPS)):
    t = i / FPS
    frame = base_bg.copy()

    push = ss((t - 7.6) / 2.8)
    s = 1.0 + 0.15 * push
    fx, fy = 540, 1264
    M = np.array([[s, 0, fx - s * IN_X], [0, s, fy - s * IN_Y]], F32)
    card = draw_card(t)
    warped = cv2.warpAffine(card, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    al = warped[..., 3:4]
    sh = cv2.resize(al[..., 0], (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    sh = cv2.resize(cv2.GaussianBlur(sh, (0, 0), 7), (W, H), interpolation=cv2.INTER_LINEAR)
    sh = np.roll(sh, 22, axis=0)
    frame = frame * (1 - 0.5 * sh[..., None])
    frame = frame * (1 - al) + warped[..., :3] * al
    g = ss((t - 8.0) / 1.4)
    if g > 0:
        add_glow(frame, fx, fy, int(380 + 120 * push), 0.34 * g)

    fade = ss(t / 1.2)
    for p in dust:
        x = (p["x"] - p["sp"] * 0.8 * t) % (W + 60) - 30
        y = p["y"] - 9 * t + 12 * math.sin(p["ph"] + t * 0.6)
        a_ = (0.20 + 0.20 * math.sin(p["tw"] + t * 1.7)) * fade
        sz = p["s"]
        xi, yi = int(x) - sz // 2, int(y) - sz // 2
        if 0 <= xi < W - sz and 0 <= yi < H - sz:
            frame[yi:yi + sz, xi:xi + sz] += p["spr"][..., None] * G * (a_ * 1.3)

    # the brand mark, all the way through
    bo = ss(t / 0.8) * 0.92
    h_, w_ = bug_arr.shape[:2]
    x0, y0 = int(bug_cx - bug_ox - w_ / 2), int(bug_cy - bug_oy - h_ / 2)
    x1, y1, x2, y2 = max(x0, 0), max(y0, 0), min(x0 + w_, W), min(y0 + h_, H)
    frame[y1:y2, x1:x2] *= (1 - 0.55 * bo * bug_sh[y1 - y0:y2 - y0, x1 - x0:x2 - x0, None])
    rf.add_light(frame, bug_arr, bug_ox, bug_oy, bug_cx, bug_cy, bo)
    frame[hy:hy + hm_sh.shape[0], hx:hx + hm_sh.shape[1]] *= (1 - 0.5 * bo * hm_sh[..., None])
    rf.blend(frame, hm, P, hx, hy, bo * 0.9)

    # closing card
    ca = ss((t - (t_close - XF2 / 2)) / XF2)
    if ca > 0:
        cb = close_bg.copy()
        rf.add_light(cb, LS, lox, loy, 540, 620, 1.0)
        for p in dust:
            x = (p["x"] - p["sp"] * 0.8 * t) % (W + 60) - 30
            y = (p["y"] - 12 * t) % H
            sz = p["s"]
            xi, yi = int(x) - sz // 2, int(y) - sz // 2
            if 0 <= xi < W - sz and 0 <= yi < H - sz:
                cb[yi:yi + sz, xi:xi + sz] += p["spr"][..., None] * G * 0.35
        for (m_, b_, col, base_y, t_on, op_) in ((tag_m, tag_b, P, 800, t_close + 0.7, 1.0), (sub_m, sub_b, PD, 870, t_close + 1.5, 1.0),
                                                   (sm1, sb1, P, 965, t_close + 2.6, 0.92), (sm2, sb2, P, 1010, t_close + 3.0, 0.92),
                                                   (cta_m, cta_b, G, 1110, t_close + 4.0, 1.0)):
            pc = ss((t - t_on) / 0.7)
            if pc > 0:
                rf.blend(cb, m_, col, (W - m_.shape[1]) // 2, int(base_y - b_ + (1 - pc) * 12), pc * op_)
        frame = frame * (1 - ca) + cb * ca

    # the words, timed to the voice
    for k, (m, h, c) in enumerate(blocks):
        s0 = starts[k] - 0.12
        e = (starts[k + 1] - 0.10) if k < len(blocks) - 1 else (t_close - 0.05)
        pin = ss((t - s0) / 0.55)
        pout = 1 - ss((t - (e - 0.35)) / 0.35)
        op = pin * pout
        if op > 0:
            mm = m
            sg = (1 - pin) * 8
            if sg > 0.4:
                mm = cv2.GaussianBlur(m, (0, 0), sg)
            yv = int(250 - h / 2 + (1 - pin) * 20)
            reg = frame[yv:yv + h]
            reg[:] = reg * (1 - (mm * op)[..., None]) + c * (mm * op)[..., None]

    frame = frame * vig + rng.normal(0, 0.006, (H, W, 1)).astype(F32)
    o8 = (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
    if i in still_idx:
        Image.fromarray(o8).save(f"/home/claude/social/_still_4_{still_idx[i]:.2f}.png")
    pipe.stdin.write(o8.tobytes())

pipe.stdin.close()
pipe.wait()
af = "highpass=f=70,equalizer=f=170:t=q:w=1:g=3,equalizer=f=3800:t=q:w=1:g=-1.5,acompressor=threshold=-20dB:ratio=2.5:attack=15:release=180:makeup=2"
wind = f"lowpass=f=800,highpass=f=70,volume=0.05,afade=t=in:d=1.2,afade=t=out:st={max(T - 1.6, 0):.2f}:d=1.6"
fc = f"[1:a]{af}[v];[2:a]{wind}[w];[v][w]amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7,alimiter=limit=0.95[a]"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT_SILENT, "-i", "/home/claude/voice/day04_slow_raw.wav",
                "-f", "lavfi", "-i", f"anoisesrc=d={T:.2f}:c=pink:r=24000:a=0.5", "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", f"{T:.2f}", OUT_FINAL], check=True)
print("done", OUT_FINAL, f"({T:.1f}s)")
