import sys, json, math, random, subprocess, os
import numpy as np
import cv2
from PIL import Image
from scipy.signal import butter, sosfilt
sys.path.insert(0, "/home/claude/social")
import render_feeling as rf
from social_lib import ss, glow_layer, sprite, GOLD, PARCH, PARCH_DIM

CFG = json.load(open(sys.argv[1]))
W, H = 1080, 1920
FPS = int(CFG.get("fps", 30))
F32 = np.float32
G = np.array(GOLD, F32) / 255
P = np.array(PARCH, F32) / 255
PD = np.array(PARCH_DIM, F32) / 255
WARM = np.array([1.0, 0.84, 0.58], F32)
V = json.load(open(CFG["voice_json"]))
starts, durs = V["starts"], V["durs"]
n = len(starts)
XF, XF2 = 0.45, 0.90

# ---------------- timeline ----------------
bounds = [(starts[k] + durs[k] + starts[k + 1]) / 2 for k in range(n - 1)]
t_close = starts[-1] + durs[-1] + 0.60
T = t_close + 4.3
seg = []
for i in range(n):
    s = 0.0 if i == 0 else bounds[i - 1] - XF / 2
    e = (bounds[i] + XF / 2) if i < n - 1 else (t_close + XF2 / 2)
    seg.append((s, e))

# ---------------- normalise clips ----------------
def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,duration", "-of", "json", path], capture_output=True, text=True).stdout
    return json.loads(out)["streams"][0]

streams = []
for i, c in enumerate(CFG["clips"]):
    info = probe(c["path"])
    src_dur = float(info.get("duration") or 0)
    sp = float(c.get("speed", 1.0))
    need = seg[i][1] - seg[i][0] + 0.2
    avail = (src_dur - float(c.get("in", 0))) / sp
    print(f"clip {i + 1}: {info['width']}x{info['height']} @ {info['r_frame_rate']}, need {need:.2f}s, available {avail:.2f}s" +
          ("  <-- SHORT" if avail < need else ""))
    norm = f"/home/claude/social/_norm2_{CFG['day']}_{i}.mp4"
    vf = f"setpts=PTS/{sp},fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},format=yuv420p"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(c.get("in", 0)), "-i", c["path"], "-t", f"{need * sp:.3f}",
                    "-an", "-vf", vf, "-c:v", "libx264", "-crf", "12", "-preset", "fast", norm], check=True)
    streams.append(norm)


class ClipStream:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.idx, self.frame = -1, None

    def get(self, i):
        i = max(i, 0)
        while self.idx < i:
            ok, f = self.cap.read()
            if not ok:
                break
            self.frame, self.idx = f, self.idx + 1
        return cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB).astype(F32) / 255.0


# auto balance: nudge each clip's average colour toward the common average
means = []
for p_ in streams:
    cap = cv2.VideoCapture(p_)
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ms = []
    for q in (0.2, 0.5, 0.8):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(nf * q))
        ok, f = cap.read()
        if ok:
            ms.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(F32).reshape(-1, 3).mean(axis=0) / 255.0)
    means.append(np.mean(ms, axis=0))
target = np.mean(means, axis=0)
auto = [np.clip((target / m) ** 0.6, 0.90, 1.10).astype(F32) for m in means]
print("auto colour balance gains:", [np.round(a, 3).tolist() for a in auto])

cs = [ClipStream(p_) for p_ in streams]


def grade(f, gain):
    f = f * gain
    lum = (f @ np.array([0.299, 0.587, 0.114], F32))[..., None]
    f = f * 0.93 + lum * 0.07
    f = f + np.array([-0.006, 0.0, 0.016], F32) * (1 - lum) ** 2
    f = f + np.array([0.018, 0.007, -0.010], F32) * lum ** 2
    return np.clip(f, 0, 1) ** 1.06


def bloom(f):
    hi = np.clip(f - 0.60, 0, 1)
    sm = cv2.GaussianBlur(cv2.resize(hi, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 9)
    return f + 0.55 * cv2.resize(sm, (W, H), interpolation=cv2.INTER_LINEAR) * np.array([1.0, 0.88, 0.70], F32)


def zoom(f, z):
    if abs(z - 1) < 1e-3:
        return f
    M = np.array([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H / 2]], F32)
    return cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


top_dark = (1 - 0.40 * (1 - np.clip(np.arange(H, dtype=F32) / 780.0, 0, 1)) ** 1.5)[:, None, None]
vig = (1 - 0.30 * (((np.arange(W)[None, :] - W / 2) ** 2 + (np.arange(H)[:, None] - H / 2) ** 2) /
                   ((W / 2) ** 2 + (H / 2) ** 2)) ** 1.1)[..., None].astype(F32)

# ---------------- words: each one arrives as it is spoken ----------------
def build_words(text, gold, ty, max_h, s_start, s_dur):
    fname = "CormorantGaramond-Italic" if gold else "CormorantGaramond"
    size, f, lines = rf.choose_size_f(text, fname, 860, max_h)
    lh = size * 1.18
    top = ty - len(lines) * lh / 2
    sp = f.getlength(" ")
    allw = [w for ln, _ in lines for w, _ in ln]
    total = sum(len(w) + 1.5 for w in allw)
    acc, words = 0.0, []
    for li, (ln, cw) in enumerate(lines):
        x = (W - cw) / 2
        base = top + size * 0.95 + li * lh
        for w, wl in ln:
            m, b = rf.tmask(w, f, 0, 40)
            frac = acc / total
            acc += len(w) + 1.5
            words.append(dict(m=m, sh=cv2.GaussianBlur(m, (0, 0), 7), x=int(x) - 40, y=int(base - b),
                              t0=s_start - 0.10 + frac * s_dur * 0.92))
            x += wl + sp
    return words, (G if gold else P)


lines_w = []
for k, ln in enumerate(CFG["lines"]):
    ws, col = build_words(ln["text"], ln.get("gold", False), ln.get("y", 430), ln.get("max_h", 430), starts[k], durs[k])
    lines_w.append((ws, col, ln.get("y", 430)))

# ---------------- brand mark ----------------
BUG = CFG.get("bug", {})
BUG_ON = BUG.get("enabled", True)
if BUG_ON:
    bug_w = BUG.get("width", 240)
    bug_arr, bug_ox, bug_oy = rf.lock_scaled(bug_w)
    bug_sh = cv2.GaussianBlur(bug_arr.max(axis=2), (0, 0), 7)
    bug_cx, bug_cy = 62 + bug_w / 2, BUG.get("y", 1405)
    hm, hb = rf.tmask(BUG.get("handle", "@mentormindhq"), rf.font("Inter", 23, 500), 3)
    hm_sh = cv2.GaussianBlur(hm, (0, 0), 5)
    hx, hy = 62 - 40, int(bug_cy + 62 - hb)

# ---------------- closing card ----------------
LSb, lox, loy = rf.lock_scaled(700)
tag_m, tag_b = rf.tmask("The counsel of the ages, kept for you.", rf.font("CormorantGaramond-Italic", 52, 500), 0)
cta_m, cta_b = rf.tmask(CFG["closing"].get("cta", "FOLLOW FOR LAUNCH DAY"), rf.font("Inter", 34, 500), 8)
hnd_m, hnd_b = rf.tmask("@mentormindhq", rf.font("Inter", 28, 500), 4)
close_bg = glow_layer(W, H, [(540, 1250, 520, 0.28, GOLD), (800, 260, 640, 0.17, (63, 90, 140)), (120, 1600, 520, 0.22, (40, 52, 80))])
GR = 500
gy_, gx_ = np.mgrid[-GR:GR + 1, -GR:GR + 1].astype(F32)
gspr = np.exp(-((gx_ ** 2 + gy_ ** 2) / (GR * 0.42) ** 2)).astype(F32)
moon = CFG.get("moon_xy", [778, 570])


def add_glow(frame, cx, cy, R, amt):
    s = cv2.resize(gspr, (2 * R, 2 * R), interpolation=cv2.INTER_LINEAR)
    x0, y0 = int(cx - R), int(cy - R)
    x1, y1, x2, y2 = max(x0, 0), max(y0, 0), min(x0 + 2 * R, W), min(y0 + 2 * R, H)
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] += s[y1 - y0:y2 - y0, x1 - x0:x2 - x0, None] * WARM * amt


rnd = random.Random(CFG["day"] * 7)
dust = []
for _ in range(110):
    spr, s = sprite(rnd.uniform(1.3, 3.0))
    dust.append(dict(x=rnd.uniform(0, W), y=rnd.uniform(0, H), sp=rnd.uniform(14, 50), ph=rnd.uniform(0, 6.28),
                     tw=rnd.uniform(0, 6.28), spr=spr, s=s))
rng = np.random.RandomState(3)

# ---------------- sound design (all synthesised, nothing borrowed) ----------------
def make_fx(sr=24000):
    nS = int(T * sr)
    t = np.arange(nS, dtype=F32) / sr
    rs = np.random.RandomState(7)
    pad = np.zeros(nS, F32)
    for f0, ph, lfo in ((110.0, 0.0, 0.11), (164.81, 1.3, 0.07), (220.0, 2.1, 0.09), (261.63, 0.7, 0.05)):
        pad += np.sin(2 * np.pi * f0 * t + ph) * (0.6 + 0.4 * np.sin(2 * np.pi * lfo * t + ph))
    pad *= 0.020 * np.clip(t / 1.8, 0, 1) * np.clip((T - t) / 2.2, 0, 1)
    sos = butter(2, [70, 700], btype="band", fs=sr, output="sos")
    wind = sosfilt(sos, rs.randn(nS).astype(F32))
    wind = wind / np.max(np.abs(wind)) * 0.045 * (0.65 + 0.35 * np.sin(2 * np.pi * 0.13 * t)) * np.clip(t / 1.2, 0, 1)
    fx = pad + wind

    def burst(center, width, lo, hi, amp, shape="sym"):
        sos2 = butter(2, [lo, hi], btype="band", fs=sr, output="sos")
        nz = sosfilt(sos2, rs.randn(nS).astype(F32))
        nz = nz / np.max(np.abs(nz))
        x = (t - center) / width
        env = np.where(np.abs(x) < 1, np.cos(x * np.pi / 2) ** 2, 0) if shape == "sym" else \
              np.where((x > -1) & (x < 0.1), np.clip((x + 1) / 1.1, 0, 1) ** 2, np.where((x >= 0.1) & (x < 0.5), np.cos((x - 0.1) / 0.4 * np.pi / 2) ** 2, 0))
        return (nz * env * amp).astype(F32)

    for b in bounds:                                   # a breath of air on each cut
        fx += burst(b, 0.35, 350, 2400, 0.05)
    fx += burst(t_close - 0.1, 0.9, 400, 3200, 0.07, shape="riser")     # the light swelling
    tb = t_close + 0.15                                 # a soft bell as the logo appears
    bell = np.zeros(nS, F32)
    for ratio, a, tau in ((1.0, 1.0, 1.7), (2.76, 0.45, 1.0), (5.40, 0.22, 0.6), (8.93, 0.10, 0.4)):
        dt = np.clip(t - tb, 0, None)
        bell += a * np.sin(2 * np.pi * 523.25 * ratio * dt) * np.exp(-dt / tau) * (t >= tb)
    fx += bell * 0.10
    return fx.astype(F32), sr


fx, fx_sr = make_fx()
sf_path = "/home/claude/social/_fx_day.wav"
import soundfile as sf
sf.write(sf_path, fx, fx_sr)

# ---------------- render ----------------
SILENT = f"/home/claude/social/_silent2_{CFG['day']}.mp4"
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
       "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", SILENT]
pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
still_idx = {int(round(float(x) * FPS)): float(x) for x in CFG.get("stills", [])}

for i in range(int(T * FPS)):
    t = i / FPS
    a = [ss((t - (b - XF / 2)) / XF) for b in bounds]
    comp = None
    for k in range(n):
        a_in = 1.0 if k == 0 else a[k - 1]
        a_out = 0.0 if k == n - 1 else a[k]
        if not (a_in > 0 and a_out < 1):
            continue
        fr = grade(cs[k].get(int(round((t - seg[k][0]) * FPS))), (auto[k] * np.array(CFG["clips"][k].get("gain", (1, 1, 1)), F32)))
        z = 1.0
        if 0 < a_in < 1:
            z *= 1 + 0.05 * (1 - a_in)
        if 0 < a_out < 1:
            z *= 1 + 0.035 * a_out
        fr = zoom(fr, z)
        comp = fr if comp is None else comp * (1 - a_in) + fr * a_in
    for av in a:                                                  # a flash of warm light on each cut
        if 0 < av < 1:
            comp = comp + 0.085 * math.sin(math.pi * av) * WARM
    comp = bloom(comp) * top_dark * vig

    fade = ss(t / 1.2)
    for p in dust:
        x = (p["x"] - p["sp"] * 0.8 * t) % (W + 60) - 30
        y = p["y"] - 9 * t + 12 * math.sin(p["ph"] + t * 0.6)
        al = (0.20 + 0.20 * math.sin(p["tw"] + t * 1.7)) * fade
        s = p["s"]
        xi, yi = int(x) - s // 2, int(y) - s // 2
        if 0 <= xi < W - s and 0 <= yi < H - s:
            comp[yi:yi + s, xi:xi + s] += p["spr"][..., None] * G * (al * 1.3)

    if BUG_ON:
        bo = ss(t / 0.8) * BUG.get("opacity", 0.92)
        h_, w_ = bug_arr.shape[:2]
        x0, y0 = int(bug_cx - bug_ox - w_ / 2), int(bug_cy - bug_oy - h_ / 2)
        x1, y1, x2, y2 = max(x0, 0), max(y0, 0), min(x0 + w_, W), min(y0 + h_, H)
        comp[y1:y2, x1:x2] *= (1 - 0.55 * bo * bug_sh[y1 - y0:y2 - y0, x1 - x0:x2 - x0, None])
        rf.add_light(comp, bug_arr, bug_ox, bug_oy, bug_cx, bug_cy, bo)
        comp[hy:hy + hm_sh.shape[0], hx:hx + hm_sh.shape[1]] *= (1 - 0.5 * bo * hm_sh[..., None])
        rf.blend(comp, hm, P, hx, hy, bo * 0.9)

    # the light of the tiny moon swells and becomes the card
    swell = ss((t - (t_close - 1.0)) / 0.9) * (1 - ss((t - (t_close + 0.15)) / 0.9))
    if swell > 0:
        add_glow(comp, moon[0], moon[1], int(150 + 650 * swell), 0.95 * swell)
        comp = comp + 0.10 * swell * WARM
    ca = ss((t - (t_close - XF2 / 2)) / XF2)
    if ca > 0:
        cb = close_bg.copy()
        sc_ = 1.05 - 0.05 * ss((t - (t_close - XF2 / 2)) / 1.4)
        arr = cv2.resize(LSb, (int(LSb.shape[1] * sc_), int(LSb.shape[0] * sc_)), interpolation=cv2.INTER_LINEAR)
        rf.add_light(cb, arr, lox * sc_, loy * sc_, 540, 690, 1.0)
        for p in dust:
            x = (p["x"] - p["sp"] * 0.8 * t) % (W + 60) - 30
            y = (p["y"] - 12 * t) % H
            s = p["s"]
            xi, yi = int(x) - s // 2, int(y) - s // 2
            if 0 <= xi < W - s and 0 <= yi < H - s:
                cb[yi:yi + s, xi:xi + s] += p["spr"][..., None] * G * 0.35
        for (m_, b_, col, base_y, t_on, dur_) in ((tag_m, tag_b, PD, 860, t_close + 0.85, 0.8),
                                                    (cta_m, cta_b, G, 985, t_close + 1.55, 0.8),
                                                    (hnd_m, hnd_b, PD, 1055, t_close + 2.1, 0.8)):
            pc = ss((t - t_on) / dur_)
            if pc > 0:
                mm = m_
                if (1 - pc) * 6 > 0.4:
                    mm = cv2.GaussianBlur(m_, (0, 0), (1 - pc) * 6)
                rf.blend(cb, mm, col, (W - m_.shape[1]) // 2, int(base_y - b_ + (1 - pc) * 12), pc * (0.9 if col is PD else 1.0))
        comp = comp * (1 - ca) + cb * ca

    # words, each as it is spoken
    for k, (ws, col, ty) in enumerate(lines_w):
        e = (starts[k + 1] - 0.10) if k < n - 1 else (t_close - 0.05)
        pout = 1 - ss((t - (e - 0.35)) / 0.35)
        if pout <= 0:
            continue
        for w in ws:
            pin = ss((t - w["t0"]) / 0.35)
            op = pin * pout
            if op <= 0:
                continue
            mm = w["m"]
            sg = (1 - pin) * 6
            if sg > 0.4:
                mm = cv2.GaussianBlur(mm, (0, 0), sg)
            yv = w["y"] + int((1 - pin) * 14)
            hh, ww = mm.shape
            if yv >= 0 and yv + hh <= H and w["x"] >= 0 and w["x"] + ww <= W:
                comp[yv:yv + hh, w["x"]:w["x"] + ww] *= (1 - 0.40 * op * w["sh"][..., None])
            rf.blend(comp, mm, col, w["x"], yv, op)

    comp = comp + rng.normal(0, 0.006, (H, W, 1)).astype(F32)
    o8 = (np.clip(comp, 0, 1) * 255 + 0.5).astype(np.uint8)
    if i in still_idx:
        Image.fromarray(o8).save(f"/home/claude/social/_still2_{CFG['day']}_{still_idx[i]:.2f}.png")
    pipe.stdin.write(o8.tobytes())

pipe.stdin.close()
pipe.wait()

af = ("asetrate=24000*0.985,atempo=1.0152,aresample=24000,equalizer=f=170:t=q:w=1:g=3.5,equalizer=f=3800:t=q:w=1:g=-2,"
      "acompressor=threshold=-20dB:ratio=2.5:attack=15:release=180:makeup=2,aecho=0.8:0.7:45:0.10")
fc = f"[1:a]{af}[v];[v][2:a]amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7,alimiter=limit=0.95[a]"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", SILENT, "-i", CFG["voice_wav"], "-i", sf_path,
                "-filter_complex", fc, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                "-t", f"{T:.2f}", CFG["out"]], check=True)
print("done", CFG["out"], f"({T:.1f}s)")
