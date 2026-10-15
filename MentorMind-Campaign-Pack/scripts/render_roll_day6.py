import sys, json, math, random, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, "/home/claude/social")
import render_feeling as rf
from social_lib import ss, glow_layer, sprite, GOLD, PARCH, PARCH_DIM

W, H, FPS = 1080, 1920, 24
F32 = np.float32
G = np.array(GOLD, F32) / 255
P = np.array(PARCH, F32) / 255
PD = np.array(PARCH_DIM, F32) / 255
V = json.load(open("/home/claude/voice/day06_voice.json"))
st, du = V["starts"], V["durs"]
OUT_SILENT = "/home/claude/social/_silent_6.mp4"
OUT_FINAL = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/social/day06_final.mp4"
STILLS = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else [])]

PR = "/mnt/project/"
GRID = [["portrait-addams-service.jpeg", "portrait-cicero-friendship.jpeg", "portrait-marcus_aurelius-mortality.jpeg"],
        ["portrait-epictetus-adversity.jpeg", "portrait-darwin-science.jpeg", "Marble_bust_of_Frederick_Douglass_2K_20261014041029.jpg"],
        ["portrait-carnegie-wealth.jpeg", "portrait-franklin-craft.jpeg", "portrait-grant-leadership.jpeg"],
        ["portrait-emerson-thinking.jpeg", "portrait-augustine-faith.jpeg", "portrait-thoreau-nature.jpeg"]]
MENTORS = [("MARCUS AURELIUS", "An emperor who wrote to himself at night.", (0, 2)),
           ("EPICTETUS", "Born a slave, he taught endurance.", (1, 0)),
           ("FREDERICK DOUGLASS", "He freed himself, then freed others with words.", (1, 2)),
           ("BENJAMIN FRANKLIN", "A printer who built a life patiently.", (2, 1)),
           ("RALPH WALDO EMERSON", "He taught us to trust our own mind.", (3, 0))]

# ------------------------------------------------ the wall (built at 2x so the zoom stays sharp)
TW, TH, GX, GY = 320, 350, 30, 24
X0 = (W - (3 * TW + 2 * GX)) // 2
Y0 = 1060 - (4 * TH + 3 * GY) // 2
CTR = (540, 1060)
bg1 = glow_layer(W, H, [(540, 1060, 700, 0.16, GOLD), (800, 260, 640, 0.12, (63, 90, 140)), (120, 1600, 560, 0.18, (40, 52, 80))])
bg2 = cv2.resize(bg1, (W * 2, H * 2), interpolation=cv2.INTER_LINEAR)


def frame_it(t):
    Wt, Ht = t.size
    B = 30
    arr = np.asarray(t.convert("RGB"), F32) / 255.0
    sh = Image.new("L", (Wt, Ht), 0)
    ImageDraw.Draw(sh).rectangle([B, B, Wt - B, Ht - B], fill=255)
    m = (1 - np.asarray(sh.filter(ImageFilter.GaussianBlur(16)), F32) / 255)[..., None]
    arr = arr * (1 - 0.65 * m)                                          # the picture sits back inside the frame
    yy, xx = np.mgrid[0:Ht, 0:Wt].astype(F32)
    k = ((xx / Wt + yy / Ht) / 2)[..., None]
    gold = np.array([244, 206, 128], F32) / 255 * (1 - k) + np.array([112, 78, 32], F32) / 255 * k
    ring = Image.new("L", (Wt, Ht), 0)
    rd = ImageDraw.Draw(ring)
    rd.rounded_rectangle([0, 0, Wt - 1, Ht - 1], radius=14, fill=255)
    rd.rectangle([B, B, Wt - 1 - B, Ht - 1 - B], fill=0)
    r = (np.asarray(ring, F32) / 255)[..., None]
    arr = arr * (1 - r) + gold * r
    img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([5, 5, Wt - 6, Ht - 6], radius=12, outline=(255, 232, 170), width=3)          # outer highlight
    d.rectangle([B - 8, B - 8, Wt - 1 - (B - 8), Ht - 1 - (B - 8)], outline=(255, 226, 160), width=2)  # moulding step
    d.rectangle([B - 1, B - 1, Wt - B, Ht - B], outline=(40, 26, 8), width=3)                          # the inner lip
    al = Image.new("L", (Wt, Ht), 0)
    ImageDraw.Draw(al).rounded_rectangle([0, 0, Wt - 1, Ht - 1], radius=14, fill=255)
    return np.asarray(img, F32) / 255.0, (np.asarray(al, F32) / 255.0)[..., None]


def tile_img(path):
    im = Image.open(PR + path).convert("RGB")
    g = np.asarray(im.convert("L"), F32)                          # measure and drop any white border on the source file
    top, bot, lef, rig = 0, g.shape[0], 0, g.shape[1]
    while bot > top + 10 and g[bot - 1, lef:rig].mean() > 235: bot -= 1
    while rig > lef + 10 and g[top:bot, rig - 1].mean() > 235: rig -= 1
    while top < bot - 10 and g[top, lef:rig].mean() > 235: top += 1
    while lef < rig - 10 and g[top:bot, lef].mean() > 235: lef += 1
    im = im.crop((lef, top, rig - 2, bot - 2))
    a = np.asarray(im, F32)
    h, w = a.shape[:2]
    gold = (a[..., 0] > 150) & (a[..., 1] > 90) & (a[..., 2] < 120) & (a[..., 0] > a[..., 2] + 55)
    cols = np.where(gold[:, int(w * 0.35):].sum(axis=0) > h * 0.05)[0] + int(w * 0.35)
    cx = (np.percentile(cols, 1) + np.percentile(cols, 99)) / 2 if len(cols) else w * 0.7
    cw = h * (TW / TH)
    x0 = int(np.clip(cx - cw / 2, 0, w - cw))
    crop = im.crop((x0, 0, int(x0 + cw), h)).resize((TW * 2, TH * 2), Image.LANCZOS)
    return frame_it(crop), (round(cx), x0)


shadow = np.zeros(bg2.shape[:2], F32)
tiles_ready, rects = [], {}
for r, row in enumerate(GRID):
    for c, f in enumerate(row):
        (t_, al_), info = tile_img(f)
        x, y = X0 + c * (TW + GX), Y0 + r * (TH + GY)
        rects[(r, c)] = (x, y, x + TW, y + TH)
        tiles_ready.append((t_, al_, x, y))
        pad = 70
        patch = np.zeros((TH * 2 + 2 * pad, TW * 2 + 2 * pad), F32)
        patch[pad:pad + TH * 2, pad:pad + TW * 2] = al_[..., 0]
        patch = cv2.GaussianBlur(patch, (0, 0), 24)
        ys, xs = y * 2 - pad + 20, x * 2 - pad
        sy0, sx0 = max(ys, 0), max(xs, 0)
        sy1, sx1 = min(ys + patch.shape[0], shadow.shape[0]), min(xs + patch.shape[1], shadow.shape[1])
        shadow[sy0:sy1, sx0:sx1] = np.maximum(shadow[sy0:sy1, sx0:sx1], patch[sy0 - ys:sy1 - ys, sx0 - xs:sx1 - xs])
        print("tile", f[:26], "arch centre / crop x0:", info)
base = bg2 * (1 - 0.62 * shadow[..., None])
dim2, lit2 = base.copy(), base.copy()
for t_, al_, x, y in tiles_ready:
    ys, xs = y * 2, x * 2
    reg = lit2[ys:ys + TH * 2, xs:xs + TW * 2]
    reg[:] = reg * (1 - al_) + t_ * al_
    reg = dim2[ys:ys + TH * 2, xs:xs + TW * 2]
    reg[:] = reg * (1 - al_) + (t_ * 0.30) * al_
lit8 = (np.clip(lit2, 0, 1) * 255).astype(np.uint8)
dim8 = (np.clip(dim2, 0, 1) * 255).astype(np.uint8)
del dim2, lit2

# ------------------------------------------------ timeline
N_I = [st[1], st[3], st[5], st[7], st[9]]                         # when each name is spoken
D_I = [st[2], st[4], st[6], st[8], st[10]]                        # when each line is spoken
E_I = [st[2] + du[2], st[4] + du[4], st[6] + du[6], st[8] + du[8], st[10] + du[10]]
t_final = st[11]
t_close = st[11] + du[11] + 1.3
T = t_close + 4.8
Z = 2.7

keys = [(0.0, 1.0, CTR[0], CTR[1]), (st[0] + du[0] + 0.2, 1.0, CTR[0], CTR[1])]
for i, (_, _, (r, c)) in enumerate(MENTORS):
    x0, y0, x1, y1 = rects[(r, c)]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    keys.append((N_I[i] - 0.10, Z, cx, cy))
    keys.append((E_I[i] + 0.10, Z * 1.05, cx, cy))
keys.append((t_final - 0.05, 1.0, CTR[0], CTR[1]))
keys.append((T + 1, 1.0, CTR[0], CTR[1]))


def camera(t):
    for k in range(len(keys) - 1):
        t0, s0, x0, y0 = keys[k]
        t1, s1, x1, y1 = keys[k + 1]
        if t0 <= t <= t1:
            u = ss((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
            dip = 0.9 * math.sin(math.pi * u) if (s0 > 1.5 and s1 > 1.5 and (x0, y0) != (x1, y1)) else 0.0
            return s0 + (s1 - s0) * u - dip, x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
    return keys[-1][1], keys[-1][2], keys[-1][3]


def lit_factor(i, t):
    a0 = N_I[i] - 0.20
    a1 = E_I[i] + 0.05
    return ss((t - a0) / 0.5) * (1 - ss((t - a1) / 0.5))


# ------------------------------------------------ text, mark and the rest
vig = (1 - 0.30 * (((np.arange(W)[None, :] - W / 2) ** 2 + (np.arange(H)[:, None] - H / 2) ** 2) /
                   ((W / 2) ** 2 + (H / 2) ** 2)) ** 1.1)[..., None].astype(F32)
GR = 520
gy_, gx_ = np.mgrid[-GR:GR + 1, -GR:GR + 1].astype(F32)
gspr = np.exp(-((gx_ ** 2 + gy_ ** 2) / (GR * 0.42) ** 2)).astype(F32)


_GCACHE = {}


def add_glow(frame, cx, cy, R, amt):
    if R not in _GCACHE:
        _GCACHE[R] = cv2.resize(gspr, (2 * R, 2 * R), interpolation=cv2.INTER_LINEAR)
    s_ = _GCACHE[R]
    x0, y0 = int(cx - R), int(cy - R)
    x1, y1, x2, y2 = max(x0, 0), max(y0, 0), min(x0 + 2 * R, W), min(y0 + 2 * R, H)
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] += s_[y1 - y0:y2 - y0, x1 - x0:x2 - x0, None] * np.array([1.0, 0.86, 0.6], F32) * amt


rnd = random.Random(66)
dust = []
for _ in range(110):
    spr, sz = sprite(rnd.uniform(1.3, 3.0))
    dust.append(dict(x=rnd.uniform(0, W), y=rnd.uniform(0, H), sp=rnd.uniform(14, 50), ph=rnd.uniform(0, 6.28),
                     tw=rnd.uniform(0, 6.28), spr=spr, s=sz))

intro_m, intro_h, *_ = rf.block_mask("Meet some of your mentors.", "CormorantGaramond", W, 215)
final_m, final_h, *_ = rf.block_mask("And more inside.", "CormorantGaramond-Italic", W, 215)
names = [rf.tmask(n, rf.font("Inter", 34, 500), 8) for n, _, _ in MENTORS]
descs = [rf.block_mask(d_, "CormorantGaramond", W, 175)[:2] for _, d_, _ in MENTORS]

bug_w = 240
bug_arr, bug_ox, bug_oy = rf.lock_scaled(bug_w)
bug_sh = cv2.GaussianBlur(bug_arr.max(axis=2), (0, 0), 7)
bug_cx, bug_cy = 62 + bug_w / 2, 1405
hm, hb = rf.tmask("@mentormindhq", rf.font("Inter", 23, 500), 3)
hm_sh = cv2.GaussianBlur(hm, (0, 0), 5)
hx, hy = 62 - 40, int(bug_cy + 62 - hb)

LS, lox, loy = rf.lock_scaled(700)
close_bg = glow_layer(W, H, [(540, 1250, 520, 0.28, GOLD), (800, 260, 640, 0.17, (63, 90, 140)), (120, 1600, 520, 0.22, (40, 52, 80))])
tag_m, tag_b = rf.tmask("Opening soon.", rf.font("CormorantGaramond-Italic", 64, 500), 0)
sub_m, sub_b = rf.tmask("AI mentors, built from what they wrote.", rf.font("Inter", 31, 500), 0)
sm_m, sm_b = rf.tmask("The portraits are AI-made illustrations.", rf.font("Inter", 27, 500), 0)
cta_m, cta_b = rf.tmask("FOLLOW FOR LAUNCH DAY", rf.font("Inter", 34, 500), 8)
XF2 = 0.9
rng = np.random.RandomState(7)
NOISE = [rng.normal(0, 0.006, (H, W, 1)).astype(F32) for _ in range(6)]


def put_block(frame, m, h, col, ymid, t, t_on, t_off):
    pin = ss((t - t_on) / 0.5)
    pout = 1 - ss((t - (t_off - 0.35)) / 0.35)
    op = pin * pout
    if op <= 0:
        return
    mm = m
    sg = (1 - pin) * 8
    if sg > 0.4:
        mm = cv2.GaussianBlur(m, (0, 0), sg)
    yv = int(ymid - h / 2 + (1 - pin) * 18)
    reg = frame[yv:yv + h]
    reg[:] = reg * (1 - (mm * op)[..., None]) + col * (mm * op)[..., None]


cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
       "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", OUT_SILENT]
pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
still_idx = {int(round(x * FPS)): x for x in STILLS}

for i in range(int(T * FPS)):
    t = i / FPS
    Sc, cx, cy = camera(t)
    M = np.array([[Sc / 2, 0, CTR[0] - Sc * cx], [0, Sc / 2, CTR[1] - Sc * cy]], F32)
    dimw = cv2.warpAffine(dim8, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(F32) / 255.0
    litw = cv2.warpAffine(lit8, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(F32) / 255.0

    L = np.zeros((H // 4, W // 4), F32)
    g_all = ss((t - (t_final - 0.3)) / 1.4)
    act, act_a = None, 0.0
    for k, (_, _, rc) in enumerate(MENTORS):
        a = lit_factor(k, t)
        if a > act_a:
            act, act_a = k, a
    if act is not None:
        x0, y0, x1, y1 = rects[MENTORS[act][2]]
        fx0, fy0 = (x0 - cx) * Sc + CTR[0], (y0 - cy) * Sc + CTR[1]
        fx1, fy1 = (x1 - cx) * Sc + CTR[0], (y1 - cy) * Sc + CTR[1]
        cv2.rectangle(L, (int(fx0 / 4), int(fy0 / 4)), (int(fx1 / 4), int(fy1 / 4)), float(act_a), -1)
        L = cv2.GaussianBlur(L, (0, 0), 3)
    L = np.maximum(cv2.resize(L, (W, H), interpolation=cv2.INTER_LINEAR), g_all)[..., None]
    frame = dimw * (1 - L) + litw * L
    if act is not None and act_a > 0:
        add_glow(frame, CTR[0], CTR[1], 640, 0.22 * act_a)
    if g_all > 0:
        add_glow(frame, CTR[0], CTR[1], 760, 0.16 * g_all)

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
    bx0, by0 = int(bug_cx - bug_ox - w_ / 2), int(bug_cy - bug_oy - h_ / 2)
    bx1, by1, bx2, by2 = max(bx0, 0), max(by0, 0), min(bx0 + w_, W), min(by0 + h_, H)
    frame[by1:by2, bx1:bx2] *= (1 - 0.55 * bo * bug_sh[by1 - by0:by2 - by0, bx1 - bx0:bx2 - bx0, None])
    rf.add_light(frame, bug_arr, bug_ox, bug_oy, bug_cx, bug_cy, bo)
    frame[hy:hy + hm_sh.shape[0], hx:hx + hm_sh.shape[1]] *= (1 - 0.5 * bo * hm_sh[..., None])
    rf.blend(frame, hm, P, hx, hy, bo * 0.9)

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
        for (m_, b_, col, base_y, t_on, op_) in ((tag_m, tag_b, P, 800, t_close + 0.7, 1.0), (sub_m, sub_b, PD, 870, t_close + 1.4, 1.0),
                                                   (sm_m, sm_b, PD, 1010, t_close + 2.3, 0.85), (cta_m, cta_b, G, 1100, t_close + 3.0, 1.0)):
            pc = ss((t - t_on) / 0.7)
            if pc > 0:
                rf.blend(cb, m_, col, (W - m_.shape[1]) // 2, int(base_y - b_ + (1 - pc) * 12), pc * op_)
        frame = frame * (1 - ca) + cb * ca

    put_block(frame, intro_m, intro_h, P, 190, t, st[0] - 0.1, N_I[0] - 0.30)
    for k in range(len(MENTORS)):
        nm, nb = names[k]
        pn = ss((t - (N_I[k] - 0.1)) / 0.5) * (1 - ss((t - (E_I[k] + 0.15)) / 0.35))
        if pn > 0:
            rf.blend(frame, nm, G, (W - nm.shape[1]) // 2, int(240 - nb + (1 - ss((t - (N_I[k] - 0.1)) / 0.5)) * 14), pn)
        dm, dh = descs[k]
        put_block(frame, dm, dh, P, 365, t, D_I[k] - 0.1, E_I[k] + 0.25)
    put_block(frame, final_m, final_h, G, 190, t, t_final - 0.1, t_close - 0.05)

    frame = frame * vig + NOISE[i % 6]
    o8 = (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
    if i in still_idx:
        Image.fromarray(o8).save(f"/home/claude/social/_still_6_{still_idx[i]:.2f}.png")
    pipe.stdin.write(o8.tobytes())

pipe.stdin.close()
pipe.wait()
af = "highpass=f=70,equalizer=f=170:t=q:w=1:g=3,equalizer=f=3800:t=q:w=1:g=-1.5,acompressor=threshold=-20dB:ratio=2.5:attack=15:release=180:makeup=2"
wind = f"lowpass=f=800,highpass=f=70,volume=0.05,afade=t=in:d=1.2,afade=t=out:st={max(T - 1.6, 0):.2f}:d=1.6"
fc = f"[1:a]{af}[v];[2:a]{wind}[w];[v][w]amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7,alimiter=limit=0.95[a]"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT_SILENT, "-i", "/home/claude/voice/day06_voice_raw.wav",
                "-f", "lavfi", "-i", f"anoisesrc=d={T:.2f}:c=pink:r=24000:a=0.5", "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", f"{T:.2f}", OUT_FINAL], check=True)
print("done", OUT_FINAL, f"({T:.1f}s)")
