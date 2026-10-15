import sys, json, math, random, subprocess, time
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
V = json.load(open("/home/claude/voice/day09_voice.json"))
st, du = V["starts"], V["durs"]
E = [st[i] + du[i] for i in range(5)]
t_close = E[4] + 1.2
T = t_close + 4.2
OUT_SILENT = "/home/claude/social/_silent_9.mp4"
OUT_FINAL = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/social/day09_final.mp4"
STILLS = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else [])]
T0F = time.time()
INK = np.array([0.105, 0.075, 0.058], F32)
PAPER = np.array([0.935, 0.895, 0.805], F32)


def norm(a, s):
    return a / (a.std() + 1e-6) * s


# ---------------------------------------------------------------- the paper
def make_paper():
    rs = np.random.RandomState(21)
    fine = rs.randn(H, W).astype(F32) * 0.010
    mott = norm(cv2.GaussianBlur(rs.randn(H, W).astype(F32), (0, 0), 38), 0.016)
    fib = np.zeros((H, W), F32)
    for _ in range(1500):
        x, y, l, a = rs.randint(0, W), rs.randint(0, H), rs.randint(8, 28), rs.uniform(0, math.pi)
        cv2.line(fib, (x, y), (int(x + l * math.cos(a)), int(y + l * math.sin(a))), float(rs.uniform(-1, 1)), 1)
    fib = norm(cv2.GaussianBlur(fib, (0, 0), 0.9), 0.0055)
    chain = (0.010 * np.cos(np.arange(W, dtype=F32) * 2 * math.pi / 72))[None, :]              # the faint lines of laid paper
    tex = fine + mott + fib + chain
    yy, xx = np.mgrid[0:H:8, 0:W:8].astype(F32)
    light = cv2.resize(1.05 - 0.13 * (0.5 * xx / W + 0.5 * yy / H), (W, H), interpolation=cv2.INTER_LINEAR)
    return (PAPER[None, None, :] * (1 + tex[..., None]) * light[..., None]).astype(F32)


paper = make_paper()

# ---------------------------------------------------------------- text and plates, as ink masks
SS = 2


def new_canvas():
    return Image.new("L", (W * SS, H * SS), 0)


def draw_tracked(d, text, fnt, cx, cy, tracking):
    ws = [fnt.getlength(ch) for ch in text]
    total = sum(ws) + tracking * SS * (len(text) - 1)
    x = cx * SS - total / 2
    for ch, w_ in zip(text, ws):
        d.text((x, cy * SS), ch, font=fnt, fill=255, anchor="lm")
        x += w_ + tracking * SS


def to_mask(im):
    return np.asarray(im.resize((W, H), Image.LANCZOS), F32) / 255.0


def ornament(d, cy, half=240):
    d.line([(540 * SS - half * SS, (cy - 4) * SS), (540 * SS + half * SS, (cy - 4) * SS)], fill=255, width=3)
    d.line([(540 * SS - half * SS, (cy + 4) * SS), (540 * SS + half * SS, (cy + 4) * SS)], fill=255, width=3)
    r = 9 * SS
    d.polygon([(540 * SS, cy * SS - r), (540 * SS + r, cy * SS), (540 * SS, cy * SS + r), (540 * SS - r, cy * SS)], fill=0)
    r = 7 * SS
    d.polygon([(540 * SS, cy * SS - r), (540 * SS + r, cy * SS), (540 * SS, cy * SS + r), (540 * SS - r, cy * SS)], fill=255)


f_cap = font("Inter", 42 * SS, 500)
f_cap2 = font("Inter", 42 * SS, 500)
f_it = font("CormorantGaramond-Italic", 56 * SS, 500)
f_q = font("CormorantGaramond", 74 * SS, 500)
f_ft = font("CormorantGaramond-Italic", 50 * SS, 500)

raw = []                                           # (mask, bbox margin kind)


def text_item(fn):
    im = new_canvas()
    d = ImageDraw.Draw(im)
    fn(d)
    return to_mask(im)


m_head = text_item(lambda d: (draw_tracked(d, "FOR WHEN YOU DON'T KNOW", f_cap, 540, 238, 8), draw_tracked(d, "WHERE TO START.", f_cap, 540, 304, 8)))


def sub_fn(d):
    d.text((540 * SS, 424 * SS), "Benjamin Franklin, a printer, wrote this.", font=f_it, fill=255, anchor="mm")
    ornament(d, 500)


m_sub = text_item(sub_fn)
QUOTE = ["Human felicity is produced", "not so much by great pieces", "of good fortune that", "seldom happen,", "as by little advantages", "that occur every day."]
m_q = []
for i, ln in enumerate(QUOTE):
    m_q.append(text_item(lambda d, ln=ln, i=i: d.text((540 * SS, (612 + i * 100) * SS), ln, font=f_q, fill=255, anchor="mm")))


def attr_fn(d):
    draw_tracked(d, "BENJAMIN FRANKLIN", f_cap2, 540, 1228, 9)
    d.line([(540 * SS - 70 * SS, 1182 * SS), (540 * SS + 70 * SS, 1182 * SS)], fill=255, width=3)


m_attr = text_item(attr_fn)
m_foot = text_item(lambda d: (d.text((540 * SS, 1796 * SS), "He is one of your mentors.", font=f_ft, fill=255, anchor="mm"),
                              d.text((540 * SS, 1852 * SS), "AI, built from what he wrote.", font=f_ft, fill=255, anchor="mm")))

# the portrait, as a stippled print
PX0, PY0, PX1, PY1 = 340, 1292, 740, 1732


def portrait_mask():
    im = Image.open("/mnt/project/portrait-franklin-craft.jpeg").convert("L")
    im = im.crop((0, 0, im.width - 8, im.height - 8))
    im = im.crop((556, 0, 1196, im.height)).resize((PX1 - PX0, PY1 - PY0), Image.LANCZOS)
    g = np.asarray(im, F32) / 255.0
    g = np.clip((g - 0.04) / 0.72, 0, 1) ** 0.85
    rs = np.random.RandomState(33)
    n = cv2.GaussianBlur(rs.rand(*g.shape).astype(F32), (0, 0), 0.6)
    n = (n - n.mean()) / n.std() * 0.17
    ink = np.clip(((1 - g) * 1.35 + n - 0.48) * 6.0 + 0.5, 0, 1)
    full = np.zeros((H, W), F32)
    full[PY0:PY1, PX0:PX1] = ink
    fr = new_canvas()
    d = ImageDraw.Draw(fr)
    d.rectangle([(PX0 - 16) * SS, (PY0 - 16) * SS, (PX1 + 16) * SS, (PY1 + 16) * SS], outline=255, width=7)
    d.rectangle([(PX0 - 7) * SS, (PY0 - 7) * SS, (PX1 + 7) * SS, (PY1 + 7) * SS], outline=255, width=3)
    return np.maximum(full, to_mask(fr))


m_port = portrait_mask()

# when each press lands
chars1, chars2 = [26, 27, 20, 14], [23, 21]
tq = []
cum = 0
for c in chars1:
    tq.append(st[2] + 0.10 + cum / sum(chars1) * (du[2] - 0.40))
    cum += c
cum = 0
for c in chars2:
    tq.append(st[3] + 0.10 + cum / sum(chars2) * (du[3] - 0.30))
    cum += c
SPEC = [(m_head, st[0] + 0.30, 1.0), (m_sub, st[1] + 0.20, 1.0)]
SPEC += [(m_q[i], tq[i], 1.0) for i in range(6)]
SPEC += [(m_attr, E[3] + 0.45, 1.0), (m_port, E[3] + 1.15, 1.9), (m_foot, st[4] + 0.35, 1.0)]

rs2 = np.random.RandomState(8)
edge_noise = norm(cv2.GaussianBlur(rs2.randn(H, W).astype(F32), (0, 0), 0.8), 1.0)
cover_mott = 0.93 + norm(cv2.GaussianBlur(rs2.randn(H, W).astype(F32), (0, 0), 3.0), 0.035)

items = []
for (m, tc, heavy) in SPEC:
    ys, xs = np.where(m > 0.15)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if m is m_port:
        a = m.copy()
    else:
        mb = cv2.GaussianBlur(m, (0, 0), 0.7)
        a = np.clip((mb + 0.10 * edge_noise - 0.5) * 3.2 + 0.5, 0, 1)
    a = np.clip(a * cover_mott, 0, 1).astype(F32)
    h_ = cv2.GaussianBlur(a, (0, 0), 1.5)
    gx = cv2.Sobel(h_, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(h_, cv2.CV_32F, 0, 1, ksize=3)
    shade = np.clip(-(0.5 * gx + 0.7 * gy) * 0.9, -0.25, 0.25).astype(F32)
    mx, my = (70, 52) if heavy < 1.5 else (44, 44)
    bx0, by0, bx1, by1 = max(x0 - mx, 20), max(y0 - my, 20), min(x1 + mx, W - 20), min(y1 + my, H - 20)
    items.append(dict(a=a, shade=shade, tc=tc, heavy=heavy, bbox=(bx0, by0, bx1, by1), merged=False))
print("items", len(items), [round(i["tc"], 2) for i in items], round(time.time() - T0F, 1), "s", flush=True)

# the printed brand mark, in ink, on the paper itself
bug_w = 240
bug_arr, bug_ox, bug_oy = rf.lock_scaled(bug_w)
bm = np.clip(bug_arr.max(axis=2) * 1.5, 0, 1)
bug_cx, bug_cy = 62 + bug_w / 2, 1405
h_, w_ = bm.shape
x0b, y0b = int(bug_cx - bug_ox - w_ / 2), int(bug_cy - bug_oy - h_ / 2)
inkmask = np.zeros((H, W), F32)
x1c, y1c, x2c, y2c = max(x0b, 0), max(y0b, 0), min(x0b + w_, W), min(y0b + h_, H)
inkmask[y1c:y2c, x1c:x2c] = bm[y1c - y0b:y2c - y0b, x1c - x0b:x2c - x0b]
hm, hb = rf.tmask("@mentormindhq", rf.font("Inter", 23, 500), 3)
hx, hy = 62 - 40, int(bug_cy + 62 - hb)
inkmask[hy:hy + hm.shape[0], hx:hx + hm.shape[1]] = np.maximum(inkmask[hy:hy + hm.shape[0], hx:hx + hm.shape[1]], hm)
paper = paper * (1 - 0.80 * inkmask[..., None]) + INK * 0.80 * inkmask[..., None]

# ---------------------------------------------------------------- the platen
_PLATEN = {}


def platen_sprite(k):
    if k in _PLATEN:
        return _PLATEN[k]
    x0, y0, x1, y1 = items[k]["bbox"]
    w, h = x1 - x0, y1 - y0
    S = 2
    rs = np.random.RandomState(50 + k)
    yy = np.linspace(0, 1, h * S, dtype=F32)[:, None]
    base = (np.array([0.30, 0.31, 0.35], F32)[None, None, :] * (1 - yy[..., None]) + np.array([0.13, 0.14, 0.17], F32)[None, None, :] * yy[..., None])
    base = np.repeat(base, w * S, axis=1)
    streak = cv2.GaussianBlur(rs.randn(h * S, w * S).astype(F32), (0, 9), 3.0)
    base = base * (1 + norm(streak, 0.045)[..., None])
    img = Image.fromarray(np.clip(base * 255, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    b = 9 * S
    d.rectangle([0, 0, w * S - 1, b], fill=(120, 124, 134))
    d.rectangle([0, 0, b, h * S - 1], fill=(104, 108, 118))
    d.rectangle([0, h * S - b, w * S - 1, h * S - 1], fill=(20, 21, 25))
    d.rectangle([w * S - b, 0, w * S - 1, h * S - 1], fill=(24, 25, 30))
    for (cx, cy) in ((26, 26), (w - 26, 26), (26, h - 26), (w - 26, h - 26)):
        d.ellipse([(cx - 9) * S, (cy - 9) * S, (cx + 9) * S, (cy + 9) * S], fill=(22, 23, 27))
        d.ellipse([(cx - 6) * S, (cy - 6) * S, (cx + 6) * S, (cy + 6) * S], fill=(96, 100, 110))
        d.ellipse([(cx - 3) * S, (cy - 4) * S, (cx + 1) * S, (cy) * S], fill=(168, 172, 182))
    al = Image.new("L", (w * S, h * S), 0)
    ImageDraw.Draw(al).rounded_rectangle([0, 0, w * S - 1, h * S - 1], radius=16 * S, fill=255)
    rgb = np.asarray(img.resize((w, h), Image.LANCZOS), F32) / 255.0
    a = np.asarray(al.resize((w, h), Image.LANCZOS), F32) / 255.0
    _PLATEN[k] = (rgb, a)
    return _PLATEN[k]


def press_height(tc, t):
    """0 = pressing, 1 = hovering high. Returns (height, alpha) or None when not on screen."""
    t_in, t_hold, t_out = tc - 0.42, tc + 0.14, tc + 0.14 + 0.42
    if t < t_in or t > t_out:
        return None
    if t < tc:
        u = (t - t_in) / 0.42
        return (1 - u) ** 2, ss((t - t_in) / 0.20)
    if t <= t_hold:
        return 0.0, 1.0
    v = (t - t_hold) / 0.42
    return ss(v), 1 - ss((v - 0.45) / 0.55)


def draw_platen(frame, k, hgt, alpha):
    rgb, a = platen_sprite(k)
    x0, y0, x1, y1 = items[k]["bbox"]
    s = 1 + 0.16 * hgt
    nw, nh = int(round((x1 - x0) * s)), int(round((y1 - y0) * s))
    rgb2, a2 = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR), cv2.resize(a, (nw, nh), interpolation=cv2.INTER_LINEAR)
    sig = 7.0 * hgt
    if sig > 0.5:
        rgb2, a2 = cv2.GaussianBlur(rgb2, (0, 0), sig), cv2.GaussianBlur(a2, (0, 0), sig)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    px, py = int(round(cx - nw / 2)), int(round(cy - nh / 2))
    # the shadow it throws on the paper: far and soft while high, tight when pressing
    off_x, off_y, sh_sig = int(8 + 34 * hgt), int(12 + 52 * hgt), 5 + 15 * hgt
    pad = int(sh_sig * 3 + 70)
    sh = np.zeros((nh + 2 * pad, nw + 2 * pad), F32)
    sh[pad:pad + nh, pad:pad + nw] = a2
    sh = cv2.GaussianBlur(sh, (0, 0), sh_sig) * (0.42 * (1 - 0.35 * hgt)) * alpha
    sx0, sy0 = px - pad + off_x, py - pad + off_y
    gx1, gy1, gx2, gy2 = max(sx0, 0), max(sy0, 0), min(sx0 + sh.shape[1], W), min(sy0 + sh.shape[0], H)
    if gx2 > gx1 and gy2 > gy1:
        frame[gy1:gy2, gx1:gx2] *= (1 - sh[gy1 - sy0:gy2 - sy0, gx1 - sx0:gx2 - sx0, None])
    gx1, gy1, gx2, gy2 = max(px, 0), max(py, 0), min(px + nw, W), min(py + nh, H)
    if gx2 > gx1 and gy2 > gy1:
        aa = (a2 * alpha)[gy1 - py:gy2 - py, gx1 - px:gx2 - px, None]
        frame[gy1:gy2, gx1:gx2] = frame[gy1:gy2, gx1:gx2] * (1 - aa) + rgb2[gy1 - py:gy2 - py, gx1 - px:gx2 - px] * aa


# ---------------------------------------------------------------- the rest
vig = (1 - 0.20 * (((np.arange(W)[None, :] - W / 2) ** 2 + (np.arange(H)[:, None] - H / 2) ** 2) /
                   ((W / 2) ** 2 + (H / 2) ** 2)) ** 1.1)[..., None].astype(F32)
close_bg = glow_layer(W, H, [(540, 1250, 520, 0.28, GOLD), (800, 260, 640, 0.17, (63, 90, 140)), (120, 1600, 520, 0.22, (40, 52, 80))])
rnd = random.Random(99)
dust = []
for _ in range(110):
    spr, sz = sprite(rnd.uniform(1.3, 3.0))
    dust.append(dict(x=rnd.uniform(0, W), y=rnd.uniform(0, H), sp=rnd.uniform(14, 50), ph=rnd.uniform(0, 6.28),
                     tw=rnd.uniform(0, 6.28), spr=spr, s=sz))
LS, lox, loy = rf.lock_scaled(700)
tag_m, tag_b = rf.tmask("Opening soon.", rf.font("CormorantGaramond-Italic", 64, 500), 0)
cta_m, cta_b = rf.tmask("FOLLOW FOR LAUNCH DAY", rf.font("Inter", 34, 500), 8)
XF2 = 0.9
rng = np.random.RandomState(11)
NOISE = [rng.normal(0, 0.006, (H, W, 1)).astype(F32) for _ in range(6)]

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
       "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", OUT_SILENT]
pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
still_idx = {int(round(x * FPS)): x for x in STILLS}
S_alpha = np.zeros((H, W), F32)
S_shade = np.zeros((H, W), F32)

for i in range(int(T * FPS)):
    t = i / FPS
    # fold finished prints into the cached layer
    for it in items:
        if not it["merged"] and t > it["tc"] + 1.6:
            S_alpha = np.maximum(S_alpha, it["a"])
            S_shade += it["shade"]
            it["merged"] = True
    ink_a = S_alpha.copy()
    shade = S_shade.copy()
    gloss = np.zeros((H, W), F32)
    for it in items:
        if not it["merged"] and t >= it["tc"]:
            wet = math.exp(-(t - it["tc"]) / 0.7)
            squash = 1 + 0.22 * wet                                     # fresh ink spreads a little, then settles
            a = np.clip(it["a"] * squash, 0, 1)
            ink_a = np.maximum(ink_a, a)
            shade += it["shade"]
            gloss = np.maximum(gloss, it["a"] * wet)
    fr = paper * (1 + shade[..., None] * 0.9)
    fr = fr * (1 - 0.96 * ink_a[..., None]) + INK * 0.96 * ink_a[..., None]
    fr += gloss[..., None] * np.array([0.34, 0.30, 0.26], F32) * 0.55
    for k, it in enumerate(items):
        ph = press_height(it["tc"], t)
        if ph is not None:
            draw_platen(fr, k, ph[0], ph[1])
    # the whole sheet jolts a little when a press lands, and drifts slowly closer
    jolt = 0.0
    for it in items:
        dt_ = t - it["tc"]
        if dt_ >= 0:
            jolt += 6.0 * it["heavy"] * math.exp(-dt_ / 0.09) * math.cos(55 * dt_)
    z = 1.0 + 0.045 * ss(t / (t_close + 1.0))
    Mz = np.array([[z, 0, 540 - z * 540], [0, z, 900 - z * 900 + jolt]], F32)
    fr = cv2.warpAffine(fr, Mz, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fr = fr * vig + NOISE[i % 6]

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
        for (m_, b_, col, base_y, t_on) in ((tag_m, tag_b, P, 830, t_close + 0.7), (cta_m, cta_b, G, 960, t_close + 1.6)):
            pc = ss((t - t_on) / 0.7)
            if pc > 0:
                rf.blend(cb, m_, col, (W - m_.shape[1]) // 2, int(base_y - b_ + (1 - pc) * 12), pc)
        fr = fr * (1 - ca) + cb * ca

    o8 = (np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8)
    if i in still_idx:
        Image.fromarray(o8).save(f"/home/claude/social/_still_9_{still_idx[i]:.2f}.png")
    pipe.stdin.write(o8.tobytes())
    if i % 120 == 0:
        print("frame", i, "of", int(T * FPS), round(time.time() - T0F), "s", flush=True)

pipe.stdin.close()
pipe.wait()

# the sound of the press: a low thump and a short clack at each impression
import soundfile as sf
sr = 24000
sfx = np.zeros(int(T * sr) + sr, np.float32)
srs = np.random.RandomState(6)
for it in items:
    n = int(0.55 * sr)
    x = np.arange(n) / sr
    body = np.sin(2 * math.pi * (58 + 34 * np.exp(-x * 34)) * x) * np.exp(-x * 10.0)
    clack = srs.randn(n).astype(np.float32) * np.exp(-x * 85.0)
    snd = (0.60 * body + 0.20 * clack) * it["heavy"] ** 0.5 * 0.9
    i0 = int(it["tc"] * sr)
    sfx[i0:i0 + n] += snd.astype(np.float32)
sf.write("/home/claude/social/_sfx9.wav", sfx, sr)
af = "highpass=f=70,equalizer=f=170:t=q:w=1:g=3,equalizer=f=3800:t=q:w=1:g=-1.5,acompressor=threshold=-20dB:ratio=2.5:attack=15:release=180:makeup=2"
wind = f"lowpass=f=800,highpass=f=70,volume=0.05,afade=t=in:d=1.2,afade=t=out:st={max(T - 1.6, 0):.2f}:d=1.6"
press = "lowpass=f=1600,volume=0.55"
fc = (f"[1:a]{af}[v];[2:a]{wind}[w];[3:a]{press}[pp];"
      f"[v][w][pp]amix=inputs=3:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7,alimiter=limit=0.95[a]")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT_SILENT, "-i", "/home/claude/voice/day09_voice_raw.wav",
                "-f", "lavfi", "-i", f"anoisesrc=d={T:.2f}:c=pink:r=24000:a=0.5", "-i", "/home/claude/social/_sfx9.wav",
                "-filter_complex", fc, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", f"{T:.2f}", OUT_FINAL], check=True)
print("done", OUT_FINAL, f"({T:.1f}s)", round(time.time() - T0F), "s")
