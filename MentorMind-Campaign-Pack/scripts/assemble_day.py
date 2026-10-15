import sys, json, math, random, subprocess, os
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, "/home/claude/social")
import render_feeling as rf                      # lockup, text and drawing helpers
from social_lib import ss, glow_layer, sprite, GOLD, PARCH, PARCH_DIM

CFG = json.load(open(sys.argv[1]))
W, H = 1080, 1920
FPS = int(CFG.get("fps", 30))
G = np.array(GOLD, np.float32) / 255
P = np.array(PARCH, np.float32) / 255
PD = np.array(PARCH_DIM, np.float32) / 255
V = json.load(open(CFG["voice_json"]))
starts, durs = V["starts"], V["durs"]
n = len(starts)
XF, XF2 = 0.50, 0.80                              # crossfade between clips, and into the closing card

# ---------------- timeline ----------------
bounds = [(starts[k] + durs[k] + starts[k + 1]) / 2 for k in range(n - 1)]          # cut points between lines
t_close = starts[-1] + durs[-1] + 0.55                                              # closing card arrives
T = t_close + 0.4 + 2.8
seg = []
for i in range(n):
    s = 0.0 if i == 0 else bounds[i - 1] - XF / 2
    e = (bounds[i] + XF / 2) if i < n - 1 else (t_close + XF2 / 2)
    seg.append((s, e))

# ---------------- normalise every clip: 1080x1920, 30 fps, right length, no audio ----------------
def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,duration,nb_frames", "-of", "json", path],
                         capture_output=True, text=True).stdout
    return json.loads(out)["streams"][0]

streams = []
for i, c in enumerate(CFG["clips"]):
    info = probe(c["path"])
    src_dur = float(info.get("duration") or 0)
    sp = float(c.get("speed", 1.0))
    need = seg[i][1] - seg[i][0] + 0.15
    avail = (src_dur - float(c.get("in", 0))) / sp
    print(f"clip {i + 1}: {info['width']}x{info['height']} @ {info['r_frame_rate']} , {src_dur:.2f}s source, need {need:.2f}s, available {avail:.2f}s" +
          ("  <-- SHORT, last frame will be held" if avail < need else ""))
    norm = f"/home/claude/social/_norm_{CFG['day']}_{i}.mp4"
    vf = (f"setpts=PTS/{sp},fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}")
    if c.get("zoom", 1.0) > 1.0:
        vf += f",scale=iw*{c['zoom']}:ih*{c['zoom']}:flags=lanczos,crop={W}:{H}:(iw-{W})/2:(ih-{H})*{c.get('anchor_y', 0.5)}"
    vf += ",format=yuv420p"
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
        return cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


cs = [ClipStream(p) for p in streams]

# ---------------- look ----------------
def grade(f, gain=(1.0, 1.0, 1.0)):
    f = f * np.array(gain, np.float32)
    lum = (f @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    f = f * 0.93 + lum * 0.07                                                   # a touch less saturated
    f = f + np.array([-0.006, 0.0, 0.016], np.float32) * (1 - lum) ** 2        # cool, deep shadows
    f = f + np.array([0.018, 0.007, -0.010], np.float32) * lum ** 2            # warm highlights
    f = np.clip(f, 0, 1) ** 1.06                                                # slightly moodier
    return f


top_dark = (1 - 0.40 * (1 - np.clip(np.arange(H, dtype=np.float32) / 780.0, 0, 1)) ** 1.5)[:, None, None]
vig = (1 - 0.30 * (((np.arange(W)[None, :] - W / 2) ** 2 + (np.arange(H)[:, None] - H / 2) ** 2) /
                   ((W / 2) ** 2 + (H / 2) ** 2)) ** 1.1)[..., None].astype(np.float32)

# words on screen, one per line of the voice
blocks = []
for k, ln in enumerate(CFG["lines"]):
    gold = ln.get("gold", False)
    m, h, *_ = rf.block_mask(ln["text"], "CormorantGaramond-Italic" if gold else "CormorantGaramond", W, ln.get("max_h", 430))
    blocks.append((m, h, G if gold else P, ln.get("y", 430), ln.get("scrim", 0.40 if ln.get("y", 430) > 1000 else 0.0)))

# permanent brand mark (logo + handle) so no part of the video is unbranded
BUG = CFG.get("bug", {})
BUG_ON = BUG.get("enabled", True)
if BUG_ON:
    bug_w = BUG.get("width", 230)
    bug_arr, bug_ox, bug_oy = rf.lock_scaled(bug_w)
    bug_sh = cv2.GaussianBlur(bug_arr.max(axis=2), (0, 0), 7)
    bug_cx, bug_cy = 62 + bug_w / 2, BUG.get("y", 1400)
    hm, hb = rf.tmask(BUG.get("handle", "@mentormindhq"), rf.font("Inter", 23, 500), 3)
    hm_sh = cv2.GaussianBlur(hm, (0, 0), 5)
    hx, hy = 62 - 40, int(bug_cy + 62 - hb)

# closing card
LS, lox, loy = rf.lock_scaled(700)
cta_m, cta_b = rf.tmask(CFG["closing"].get("cta", "FOLLOW FOR LAUNCH DAY"), rf.font("Inter", 34, 500), 8)
close_bg = glow_layer(W, H, [(540, 1250, 520, 0.28, GOLD), (800, 260, 640, 0.17, (63, 90, 140)), (120, 1600, 520, 0.22, (40, 52, 80))])
rnd = random.Random(CFG["day"] * 7)
dust = []
for _ in range(110):
    spr, s = sprite(rnd.uniform(1.3, 3.0))
    dust.append(dict(x=rnd.uniform(0, W), y=rnd.uniform(0, H), sp=rnd.uniform(14, 50), ph=rnd.uniform(0, 6.28),
                     tw=rnd.uniform(0, 6.28), spr=spr, s=s))
rng = np.random.RandomState(3)

SILENT = f"/home/claude/social/_silent_{CFG['day']}.mp4"
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
       "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", SILENT]
pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
still_times = [float(x) for x in CFG.get("stills", [])]
still_idx = {int(round(x * FPS)): x for x in still_times}

for i in range(int(T * FPS)):
    t = i / FPS
    a = [ss((t - (b - XF / 2)) / XF) for b in bounds]            # crossfade amount at each cut
    # which clips are visible right now
    comp = None
    for k in range(n):
        vis_in = (k == 0) or a[k - 1] > 0
        vis_out = (k == n - 1) or a[k] < 1
        if not (vis_in and vis_out):
            continue
        fr = grade(cs[k].get(int(round((t - seg[k][0]) * FPS))), CFG["clips"][k].get("gain", (1.0, 1.0, 1.0)))
        if comp is None:
            comp = fr
        else:
            w = a[k - 1]
            comp = comp * (1 - w) + fr * w
    ca = ss((t - (t_close - XF2 / 2)) / XF2)

    comp = comp * top_dark * vig

    # drifting dust over the footage
    fade = ss(t / 1.2)
    dl = np.zeros((H, W, 3), np.float32) if False else None
    for p in dust:
        x = (p["x"] - p["sp"] * 0.8 * t) % (W + 60) - 30
        y = p["y"] - 9 * t + 12 * math.sin(p["ph"] + t * 0.6)
        al = (0.20 + 0.20 * math.sin(p["tw"] + t * 1.7)) * fade
        s = p["s"]
        xi, yi = int(x) - s // 2, int(y) - s // 2
        if 0 <= xi < W - s and 0 <= yi < H - s:
            comp[yi:yi + s, xi:xi + s] += p["spr"][..., None] * G * (al * 1.3)

    if BUG_ON:
        bo = ss(t / 0.8) * BUG.get("opacity", 0.85)
        h_, w_ = bug_arr.shape[:2]
        x0, y0 = int(bug_cx - bug_ox - w_ / 2), int(bug_cy - bug_oy - h_ / 2)
        x1, y1, x2, y2 = max(x0, 0), max(y0, 0), min(x0 + w_, W), min(y0 + h_, H)
        comp[y1:y2, x1:x2] *= (1 - 0.55 * bo * bug_sh[y1 - y0:y2 - y0, x1 - x0:x2 - x0, None])    # soft shadow for contrast
        rf.add_light(comp, bug_arr, bug_ox, bug_oy, bug_cx, bug_cy, bo)
        sh_reg = comp[hy:hy + hm_sh.shape[0], hx:hx + hm_sh.shape[1]]
        sh_reg *= (1 - 0.5 * bo * hm_sh[..., None])
        rf.blend(comp, hm, P, hx, hy, bo * 0.9)

    if ca > 0:                                                    # closing card
        cb = close_bg.copy()
        rf.add_light(cb, LS, lox, loy, 540, 700, 1.0)
        for p in dust:
            x = (p["x"] - p["sp"] * 0.8 * t) % (W + 60) - 30
            y = p["y"] - 9 * t
            s = p["s"]
            xi, yi = int(x) - s // 2, int(y % H) - s // 2
            if 0 <= xi < W - s and 0 <= yi < H - s:
                cb[yi:yi + s, xi:xi + s] += p["spr"][..., None] * G * 0.35
        pc = ss((t - t_close - 0.5) / 0.8)
        if pc > 0:
            rf.blend(cb, cta_m, G, (W - cta_m.shape[1]) // 2, int(905 - cta_b + (1 - pc) * 12), pc)
        comp = comp * (1 - ca) + cb * ca

    # the words, timed to the voice
    for k, (m, h, c, ty, scrim) in enumerate(blocks):
        s0 = starts[k] - 0.12
        e = (starts[k + 1] - 0.10) if k < n - 1 else (t_close - 0.05)
        pin = ss((t - s0) / 0.55)
        pout = 1 - ss((t - (e - 0.35)) / 0.35)
        op = pin * pout
        if op > 0:
            mm = m
            sg = (1 - pin) * 8
            if sg > 0.4:
                mm = cv2.GaussianBlur(m, (0, 0), sg)
            if scrim > 0:                                          # a soft shadow behind the words
                band = np.exp(-(((np.arange(H, dtype=np.float32) - ty) / 230.0) ** 2))[:, None, None]
                comp = comp * (1 - scrim * band * op)
            yv = int(ty - h / 2 + (1 - pin) * 20)
            reg = comp[yv:yv + h]
            reg[:] = reg * (1 - (mm * op)[..., None]) + c * (mm * op)[..., None]

    comp = comp + rng.normal(0, 0.006, (H, W, 1)).astype(np.float32)
    o8 = (np.clip(comp, 0, 1) * 255 + 0.5).astype(np.uint8)
    if i in still_idx:
        Image.fromarray(o8).save(f"/home/claude/social/_still_{CFG['day']}_{still_idx[i]:.2f}.png")
    pipe.stdin.write(o8.tobytes())

pipe.stdin.close()
pipe.wait()
af = CFG.get("voice_af", ("asetrate=24000*0.985,atempo=1.0152,aresample=24000,equalizer=f=170:t=q:w=1:g=3.5,equalizer=f=3800:t=q:w=1:g=-2,"
      "acompressor=threshold=-20dB:ratio=2.5:attack=15:release=180:makeup=2,aecho=0.8:0.7:45:0.10"))
wind = (f"lowpass=f=800,highpass=f=70,volume=0.05,afade=t=in:d=1.2,afade=t=out:st={max(T - 1.6, 0):.2f}:d=1.6")
fc = (f"[1:a]{af}[v];[2:a]{wind}[w];[v][w]amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7,alimiter=limit=0.95[a]")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", SILENT, "-i", CFG["voice_wav"],
                "-f", "lavfi", "-i", f"anoisesrc=d={T:.2f}:c=pink:r=24000:a=0.5",
                "-filter_complex", fc, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                "-t", f"{T:.2f}", CFG["out"]], check=True)
print("done", CFG["out"], f"({T:.1f}s)")
