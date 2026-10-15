import sys, json, math, random, subprocess, glob, os, time
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
V = json.load(open("/home/claude/voice/day08_voice.json"))
st, du = V["starts"], V["durs"]
OUT_SILENT = "/home/claude/social/_silent_8.mp4"
OUT_FINAL = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/social/day08_final.mp4"
STILLS = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else [])]
T0F = time.time()

# where the quote is carved (centre of the first line, spacing between lines); tuned to the real plaque once it arrives
QY0, QSP, QSIZE = int(os.environ.get("QY0", 1290)), 70, 56
QCX = 620                                  # a little right of centre, so no line touches the corner mark

# ---------------------------------------------------------------- timeline
T_AB, T_BC = 3.9, 8.6                       # the two crossfades (each takes one second)
t_close = st[4] + du[4] + 1.1
T = t_close + 4.0


# ---------------------------------------------------------------- the three stills (or stand-ins until they arrive)
YOURS = ["Marble_bust_under_gold_arch_2K_20261014112846.jpg",      # the smallest arch: "the arch in the dark"
         "Marble_bust_under_gold_arch_2K_20261014112509.jpg",      # the medium shot: "unmoved"
         "Marble_bust_under_golden_light_2K_20261014113232.jpg"]    # the close-up with the marble plinth: "the quote"


def trim_white(im):
    g = np.asarray(im.convert("L"), F32)
    top, bot, lef, rig = 0, g.shape[0], 0, g.shape[1]
    while bot > top + 10 and g[bot - 1, lef:rig].mean() > 235: bot -= 1
    while rig > lef + 10 and g[top:bot, rig - 1].mean() > 235: rig -= 1
    while top < bot - 10 and g[top, lef:rig].mean() > 235: top += 1
    while lef < rig - 10 and g[top:bot, lef].mean() > 235: lef += 1
    return im.crop((lef, top, rig - 2, bot - 2))


def find_still(n):
    f = "/mnt/project/" + YOURS[n - 1]
    if os.path.exists(f):
        return f
    for ext in ("png", "jpg", "jpeg", "webp", "PNG", "JPG", "JPEG"):
        g = glob.glob(f"/mnt/project/day08-still{n}*.{ext}")
        if g:
            return g[0]
    return None


def cover_fit(im, scale):
    """Resize a picture so that it covers the frame at the given extra zoom."""
    w, h = im.size
    k = max(W * scale / w, H * scale / h)
    im = im.resize((int(round(w * k)), int(round(h * k))), Image.LANCZOS)
    x0, y0 = (im.width - int(W * scale)) // 2, (im.height - int(H * scale)) // 2
    im = im.crop((x0, y0, x0 + int(W * scale), y0 + int(H * scale)))
    a = np.asarray(im, F32) / 255.0
    return np.clip(a + 0.5 * (a - cv2.GaussianBlur(a, (0, 0), 1.3)), 0, 1)       # a gentle sharpen, since the files are a little under full size


def standins():
    im = Image.open("/mnt/project/portrait-marcus_aurelius-mortality.jpeg").convert("RGB")
    im = im.crop((0, 0, im.width - 8, im.height - 8))
    arch = im.crop((560, 0, 1200, 700))
    bg = (glow_layer(W, H, [(540, 900, 700, 0.10, GOLD)]) + np.array([0.03, 0.035, 0.06], F32)).clip(0, 1)
    out = []
    for wdt in (330, 700):
        c = Image.fromarray((bg * 255).astype(np.uint8))
        a = arch.resize((wdt, int(wdt * 700 / 640)), Image.LANCZOS)
        c.paste(a, ((W - a.width) // 2, (H - a.height) // 2))
        out.append(c)
    top = arch.crop((0, 0, 640, 600)).resize((1080, int(1080 * 600 / 640)), Image.LANCZOS)
    c = Image.new("RGB", (W, H), (8, 9, 14))
    c.paste(top, (0, 0))
    plaque = np.ones((H - 1040, W, 3), F32) * np.array([0.050, 0.054, 0.070], F32)
    rs = np.random.RandomState(4)
    lo = cv2.GaussianBlur(rs.randn(H - 1040, W).astype(F32), (0, 0), 30) * 0.9
    plaque += lo[..., None] * 0.020
    for _ in range(5):
        v = np.zeros((H - 1040, W), F32)
        cv2.line(v, (rs.randint(0, W), 0), (rs.randint(0, W), H - 1040), 1.0, 2)
        plaque += cv2.GaussianBlur(v, (0, 0), 2.2)[..., None] * np.array([0.10, 0.075, 0.04], F32) * 0.6
    c.paste(Image.fromarray((np.clip(plaque, 0, 1) * 255).astype(np.uint8)), (0, 1040))
    out.append(c)
    return out


paths = [find_still(i) for i in (1, 2, 3)]
if all(paths):
    ims = [trim_white(Image.open(p).convert("RGB")) for p in paths]
    print("using your stills:", [os.path.basename(p) for p in paths], [i.size for i in ims])
else:
    ims = standins()
    print("stills not found, using stand-ins")
BASE_S = 1.20
SA, SB = cover_fit(ims[0], BASE_S), cover_fit(ims[1], BASE_S)
SC = cover_fit(ims[2], 1.0)


def kb(base, z, cx, cy):
    M = np.array([[z / BASE_S, 0, 540 - z * cx], [0, z / BASE_S, 960 - z * cy]], F32)
    return cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- the quote, carved
fq = font("CormorantGaramond", QSIZE, 500)
QL = [("Take pains therefore to know", 0), ("what it is that thy nature", 0), ("requireth,", 0), ("and let nothing else", 1), ("distract thee.", 1)]
windows = []
q1 = (st[2] + 0.05, st[2] + du[2] - 0.20)
q2 = (st[3] + 0.05, st[3] + du[3] - 0.10)
for grp, (a, b) in ((0, q1), (1, q2)):
    idx = [i for i, l in enumerate(QL) if l[1] == grp]
    tot = sum(len(QL[i][0]) for i in idx)
    cur = a
    for i in idx:
        span = (b - a) * len(QL[i][0]) / tot
        windows.append((cur, cur + span))
        cur += span
BAND = 62
lines = []
for i, (txt, _) in enumerate(QL):
    yc = QY0 + i * QSP
    ss2 = 2
    im2 = Image.new("L", (W * ss2, 2 * BAND * ss2), 0)
    f2 = font("CormorantGaramond", QSIZE * ss2, 500)
    ImageDraw.Draw(im2).text((QCX * ss2, BAND * ss2), txt, font=f2, fill=255, anchor="mm")
    M = np.asarray(im2.resize((W, 2 * BAND), Image.LANCZOS), F32) / 255.0
    Mb = cv2.GaussianBlur(M, (0, 0), 1.1)
    gx = cv2.Sobel(Mb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Mb, cv2.CV_32F, 0, 1, ksize=3)
    sh = np.clip(gx * 0.6 + gy * 0.8, 0, None) * 1.6                 # walls turned away from the light
    lit = np.clip(-(gx * 0.6 + gy * 0.8), 0, None) * 1.6             # walls turned toward it
    cols = np.where(M.max(axis=0) > 0.1)[0]
    lines.append(dict(y0=yc - BAND, y1=yc + BAND, M=M, Mb=Mb, sh=np.clip(sh, 0, 1), lit=np.clip(lit, 0, 1),
                      bloom=cv2.GaussianBlur(M, (0, 0), 7), xa=int(cols[0]), xb=int(cols[-1]), win=windows[i], yc=yc))

# chisel dust
rsp = np.random.RandomState(12)
parts = []                                   # each: [x, y, vx, vy, life, r]
spr_cache = {r: sprite(r)[0] for r in (1.2, 1.8, 2.6)}
spr_sz = {r: sprite(r)[1] for r in (1.2, 1.8, 2.6)}


def carve(frame, t, dt=1.0 / FPS):
    edge_pts = []
    for L in lines:
        a, b = L["win"]
        f = float(np.clip((t - a) / (b - a), 0, 1))
        if f <= 0:
            continue
        xe = L["xa"] - 12 + f * (L["xb"] - L["xa"] + 24)
        x = np.arange(W, dtype=F32)
        rev = np.clip((xe - x) / 16.0, 0, 1)
        boost = 1 + 1.7 * np.exp(-np.clip(xe - x, 0, None) / 60.0) * (1 if f < 1 else np.exp(-(t - b) * 1.4))
        reg = frame[L["y0"]:L["y1"]]
        R = rev[None, :]
        fill = (L["M"] * R)[..., None]
        reg *= (1 - 0.62 * fill)
        reg *= (1 - 0.40 * (L["sh"] * R)[..., None])
        shimmer = 0.92 + 0.08 * math.sin(t * 1.6 + L["yc"] * 0.01)
        reg += G * (0.62 * shimmer) * fill * boost[None, :, None]
        reg += np.array([1.0, 0.90, 0.66], F32) * (L["lit"] * R)[..., None] * 0.55
        reg += G * (L["bloom"] * R)[..., None] * 0.22 * boost[None, :, None]
        if f < 1:
            edge_pts.append((xe, L["yc"]))
    # chisel dust falls from the edge being carved
    for (xe, yc) in edge_pts:
        for _ in range(3):
            parts.append([xe + rsp.uniform(-6, 6), yc + rsp.uniform(-14, 18), rsp.uniform(-60, 60), rsp.uniform(-90, 10), rsp.uniform(0.5, 1.1), random.choice((1.2, 1.8, 2.6))])
    keep = []
    for p_ in parts:
        p_[0] += p_[2] * dt
        p_[1] += p_[3] * dt
        p_[3] += 520 * dt
        p_[4] -= dt
        if p_[4] > 0 and p_[1] < H:
            keep.append(p_)
            spr, sz, a = spr_cache[p_[5]], spr_sz[p_[5]], min(p_[4] / 0.6, 1.0)
            xi, yi = int(p_[0]) - sz // 2, int(p_[1]) - sz // 2
            if 0 <= xi < W - sz and 0 <= yi < H - sz:
                frame[yi:yi + sz, xi:xi + sz] += spr[..., None] * np.array([1.0, 0.84, 0.52], F32) * (a * 1.6)
    parts[:] = keep


# ---------------------------------------------------------------- light streaks: the voices rushing past
rs = np.random.RandomState(9)
NS = 70
s_y = rs.uniform(120, 1820, NS)
s_v = rs.uniform(420, 1500, NS)
s_l = rs.uniform(140, 520, NS)
s_w = rs.choice([2, 2, 3, 3, 4], NS)
s_a = rs.uniform(0.35, 1.0, NS)
s_x0 = rs.uniform(0, W + 800, NS)
s_d = rs.choice([-1, 1], NS, p=[0.35, 0.65])
CMASK = (1 - 0.92 * np.exp(-((np.arange(W, dtype=F32) - 540) / 250.0) ** 2))[None, :]     # keep the middle clear: he stays calm


def streak_amp(t):
    return float(ss((t - 0.8) / 2.0) * (1 - 0.65 * ss((t - 3.7) / 2.4)) * (1 - ss((t - 6.2) / 2.0)))


def streaks(t, amp):
    buf = np.zeros((H, W), np.uint8)
    for k in range(NS):
        span = W + s_l[k] * 2
        x = (s_x0[k] + s_d[k] * s_v[k] * t) % span - s_l[k]
        d = s_d[k]
        for seg in range(5):
            f0, f1 = seg / 5.0, (seg + 1) / 5.0
            xa, xb = x - d * s_l[k] * f0, x - d * s_l[k] * f1
            inten = int(255 * s_a[k] * (1 - f0) ** 1.6)
            cv2.line(buf, (int(xa), int(s_y[k])), (int(xb), int(s_y[k])), inten, int(s_w[k]), cv2.LINE_AA)
    b = cv2.GaussianBlur(buf, (0, 0), 1.6).astype(F32) / 255.0
    b = b + 0.75 * cv2.GaussianBlur(b, (0, 0), 8)           # a soft halo, so they read as light, not lines
    return np.clip(b, 0, 1.2) * CMASK * amp


# ---------------------------------------------------------------- the frame around it
vig = (1 - 0.30 * (((np.arange(W)[None, :] - W / 2) ** 2 + (np.arange(H)[:, None] - H / 2) ** 2) /
                   ((W / 2) ** 2 + (H / 2) ** 2)) ** 1.1)[..., None].astype(F32)
rnd = random.Random(88)
dust = []
for _ in range(110):
    spr, sz = sprite(rnd.uniform(1.3, 3.0))
    dust.append(dict(x=rnd.uniform(0, W), y=rnd.uniform(0, H), sp=rnd.uniform(14, 50), ph=rnd.uniform(0, 6.28),
                     tw=rnd.uniform(0, 6.28), spr=spr, s=sz))
close_bg = glow_layer(W, H, [(540, 1250, 520, 0.28, GOLD), (800, 260, 640, 0.17, (63, 90, 140)), (120, 1600, 520, 0.22, (40, 52, 80))])
w1 = rf.block_mask("For when everyone has an opinion about your life.", "CormorantGaramond", W, 215)[:2]
w2 = rf.block_mask("Marcus Aurelius, emperor of Rome, wrote this to himself.", "CormorantGaramond", W, 215)[:2]
w4 = rf.block_mask("He is one of your mentors. AI, built from what he wrote.", "CormorantGaramond", W, 150)[:2]
cred_m, cred_b = rf.tmask("MARCUS AURELIUS, MEDITATIONS", rf.font("Inter", 27, 500), 7)

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
rng = np.random.RandomState(10)
NOISE = [rng.normal(0, 0.006, (H, W, 1)).astype(F32) for _ in range(6)]
top_grad = (np.clip(1 - np.arange(430, dtype=F32) / 430, 0, 1) ** 1.2)[:, None, None]


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
    y1, y2 = max(yv, 0), min(yv + h, H)
    if y2 <= y1:
        return
    mc = mm[y1 - yv:y2 - yv]
    reg = frame[y1:y2]
    reg[:] = reg * (1 - (mc * op)[..., None]) + col * (mc * op)[..., None]


cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
       "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", OUT_SILENT]
pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
still_idx = {int(round(x * FPS)): x for x in STILLS}
random.seed(5)

for i in range(int(T * FPS)):
    t = i / FPS
    wA = 1 - ss((t - T_AB) / 1.0)
    wC = ss((t - T_BC) / 1.0)
    wB = max(0.0, 1 - wA - wC)
    frame = np.zeros((H, W, 3), F32)
    if wA > 0:
        frame += kb(SA, 1.00 + 0.09 * ss(t / (T_AB + 1.0)), 540, 960) * wA
    if wB > 0:
        frame += kb(SB, 1.00 + 0.08 * ss((t - (T_AB - 0.5)) / (T_BC - T_AB + 1.5)), 540, 960) * wB
    amp = streak_amp(t)
    if amp > 0.01:
        sk = streaks(t, amp) * (1 - wC)
        frame += sk[..., None] * np.array([1.0, 0.80, 0.46], F32) * 0.85
    scr = 0.5 * (1 - wC)
    frame[:430] *= (1 - scr * top_grad)
    if wC > 0:
        cf = SC.copy()
        # a soft light sweeps across the plaque as it is carved
        bx = -250 + (W + 500) * ss((t - 9.0) / 8.0)
        band = np.exp(-((np.arange(W, dtype=F32) - bx) / 190.0) ** 2)[None, :, None]
        cf[1226:] += band * np.array([1.0, 0.84, 0.55], F32) * 0.05
        carve(cf, t)
        z = 1.0 + 0.05 * ss((t - T_BC) / (t_close - T_BC))
        Mz = np.array([[z, 0, 540 - z * 540], [0, z, 700 - z * 700]], F32)
        cf = cv2.warpAffine(cf, Mz, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        frame += cf * wC

    fade = ss(t / 1.2)
    for p_ in dust:
        x = (p_["x"] - p_["sp"] * 0.8 * t) % (W + 60) - 30
        y = p_["y"] - 9 * t + 12 * math.sin(p_["ph"] + t * 0.6)
        a_ = (0.20 + 0.20 * math.sin(p_["tw"] + t * 1.7)) * fade
        sz = p_["s"]
        xi, yi = int(x) - sz // 2, int(y) - sz // 2
        if 0 <= xi < W - sz and 0 <= yi < H - sz:
            frame[yi:yi + sz, xi:xi + sz] += p_["spr"][..., None] * G * (a_ * 1.3)

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
        for (m_, b_, col, base_y, t_on) in ((tag_m, tag_b, P, 830, t_close + 0.7), (cta_m, cta_b, G, 960, t_close + 1.6)):
            pc = ss((t - t_on) / 0.7)
            if pc > 0:
                rf.blend(cb, m_, col, (W - m_.shape[1]) // 2, int(base_y - b_ + (1 - pc) * 12), pc)
        frame = frame * (1 - ca) + cb * ca

    put_block(frame, w1[0], w1[1], P, 210, t, st[0] - 0.10, st[1] - 0.15)
    put_block(frame, w2[0], w2[1], P, 210, t, st[1] - 0.10, T_BC + 0.35)
    put_block(frame, w4[0], w4[1], P, 1790, t, st[4] - 0.10, t_close - 0.05)
    pcr = ss((t - (st[3] + du[3] + 0.15)) / 0.8) * (1 - ss((t - (t_close - 0.5)) / 0.4))
    if pcr > 0:
        rf.blend(frame, cred_m, G, QCX - cred_m.shape[1] // 2, int(1672 - cred_b), pcr)

    frame = frame * vig + NOISE[i % 6]
    o8 = (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
    if i in still_idx:
        Image.fromarray(o8).save(f"/home/claude/social/_still_8_{still_idx[i]:.2f}.png")
    pipe.stdin.write(o8.tobytes())
    if i % 120 == 0:
        print("frame", i, "of", int(T * FPS), round(time.time() - T0F), "s", flush=True)

pipe.stdin.close()
pipe.wait()
af = "highpass=f=70,equalizer=f=170:t=q:w=1:g=3,equalizer=f=3800:t=q:w=1:g=-1.5,acompressor=threshold=-20dB:ratio=2.5:attack=15:release=180:makeup=2"
wind = f"lowpass=f=800,highpass=f=70,volume=0.05,afade=t=in:d=1.2,afade=t=out:st={max(T - 1.6, 0):.2f}:d=1.6"
fc = f"[1:a]{af}[v];[2:a]{wind}[w];[v][w]amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7,alimiter=limit=0.95[a]"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT_SILENT, "-i", "/home/claude/voice/day08_voice_raw.wav",
                "-f", "lavfi", "-i", f"anoisesrc=d={T:.2f}:c=pink:r=24000:a=0.5", "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", f"{T:.2f}", OUT_FINAL], check=True)
print("done", OUT_FINAL, f"({T:.1f}s)", round(time.time() - T0F), "s")
