import sys, math, random, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw

src = open("/home/claude/social/render_profile2.py").read()
exec(src[:src.index("a1080, a400")])        # loaders, put_add, canvas_bg, lock_c, helpers

G = np.array(GOLD, np.float32) / 255
P = np.array(PARCH, np.float32) / 255
PD = np.array(PARCH_DIM, np.float32) / 255
GS = np.array(GOLD_SOFT, np.float32) / 255

DAYS = {
    1: dict(lines=["You are carrying a lot.", "Mostly quietly.", "Some of it nobody sees."],
            flames=[dict(x=540, h=190, at=0.5)], dust="up", closing="plain", dur=9.6),
    2: dict(lines=["You know something has to change.", "You just do not know what.", "Or where to start."],
            flames=[dict(x=540, h=240, at=0.5)], dust="side", closing="plain", dur=9.6),
    3: dict(lines=["Everyone expects you to have it figured out.", "You are still figuring it out.", "That is normal."],
            flames=[dict(x=540, h=240, at=0.5)], dust="press", closing="ai3", dur=10.8),
    4: dict(lines=["Everyone asks how you are.", "Few wait for the answer.", "You do not have to think alone."],
            flames=[dict(x=400, h=230, at=0.5), dict(x=690, h=230, at="line3")], dust="up", closing="ai4", dur=12.2),
}
CLOSE = {
    "plain": dict(sub=None, small=None),
    "ai3": dict(sub="AI mentors, built from real writings.", small=None),
    "ai4": dict(sub="AI mentors, built from what the wisest people wrote.",
                small=["Not therapy. Not professional advice.", "In a crisis, contact local emergency services."]),
}


# ---------------- shared helpers ----------------
def choose_size_f(text, fname, max_w, max_h, wght=500):
    for s in [124, 114, 104, 96, 90, 84, 78, 72, 66, 60]:
        f = font(fname, s, wght)
        for mw in range(int(max_w), int(max_w) - 330, -30):
            lines = layout_words(text, f, mw)
            if len(lines) > 1 and len(lines[-1][0]) < 2:
                continue
            if len(lines) * s * 1.18 <= max_h:
                return s, f, lines
            break
    f = font(fname, 56, wght)
    return 56, f, layout_words(text, f, max_w)


def block_mask(text, fname, Wd, max_h):
    size, f, lines = choose_size_f(text, fname, 860, max_h)
    lh = size * 1.18
    pad = 50
    h = int(lh * len(lines) + 2 * pad)
    im = Image.new("L", (Wd, h), 0)
    d = ImageDraw.Draw(im)
    base = pad + size * 0.95
    sp = f.getlength(" ")
    for ln, cw in lines:
        x = (Wd - cw) / 2
        for w, wl in ln:
            d.text((x, base), w, font=f, fill=255, anchor="ls")
            x += wl + sp
        base += lh
    return np.asarray(im, np.float32) / 255.0, h, size, lines, f


def tmask(txt, f, tracking=0, pad=40):
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


def blend(frame, m, color, x, y, op):
    Hh, Ww = frame.shape[:2]
    h, w = m.shape
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, Ww), min(y + h, Hh)
    if x2 <= x1 or y2 <= y1 or op <= 0:
        return
    mm = (m[y1 - y:y2 - y, x1 - x:x2 - x] * op)[..., None]
    reg = frame[y1:y2, x1:x2]
    reg[:] = reg * (1 - mm) + color * mm


def lock_scaled(width):
    sc = width / lw_
    nw, nh = int(lock_c.shape[1] * sc), int(lock_c.shape[0] * sc)
    arr = cv2.resize(lock_c, (nw, nh), interpolation=cv2.INTER_AREA)
    ox = ((lx0 + lx1) / 2 - dx0 - lock_c.shape[1] / 2) * sc
    oy = ((ly0 + ly1) / 2 - dy0 - lock_c.shape[0] / 2) * sc
    return arr, ox, oy


def add_light(frame, arr, ox, oy, cx, cy, op=1.0):
    h, w = arr.shape[:2]
    x0, y0 = int(cx - ox - w / 2), int(cy - oy - h / 2)
    Hh, Ww = frame.shape[:2]
    x1, y1 = max(x0, 0), max(y0, 0)
    x2, y2 = min(x0 + w, Ww), min(y0 + h, Hh)
    if x2 <= x1 or y2 <= y1 or op <= 0:
        return
    frame[y1:y2, x1:x2] += arr[y1 - y0:y2 - y0, x1 - x0:x2 - x0] * op


# ---------------- reels ----------------
def reel(day):
    W, H, FPS = 1080, 1920, 30
    cfg = DAYS[day]
    n = len(cfg["lines"])
    DUR = cfg["dur"]
    step = 1.9
    s0 = 0.9
    starts = [s0 + i * step for i in range(n)]
    tc = s0 + n * step + 0.2
    blocks = []
    for i, ln in enumerate(cfg["lines"]):
        last = i == n - 1
        m, h, size, _, _ = block_mask(ln, "CormorantGaramond-Italic" if last else "CormorantGaramond", W, 430)
        blocks.append(dict(m=m, h=h, c=G if last else P))
    LS, lox, loy = lock_scaled(700)
    fc, fb = font("Inter", 33, 500), None
    cta_m, cta_b = tmask("FOLLOW FOR LAUNCH DAY", font("Inter", 34, 500), 8)
    cl = CLOSE[cfg["closing"]]
    sc_m = tmask("Something is coming.", font("CormorantGaramond-Italic", 60, 500)) if cl["sub"] else None
    sub_m = tmask(cl["sub"], font("Inter", 32, 500), 0) if cl["sub"] else None
    small_m = [tmask(t, font("Inter", 29, 500), 0) for t in (cl["small"] or [])]

    # flames
    flames = []
    for fl in cfg["flames"]:
        at = starts[2] if fl["at"] == "line3" else fl["at"]
        flames.append(dict(x=fl["x"], k=fl["h"] / 44, at=at))
    gx = sum(f["x"] for f in flames[:1]) if len(flames) == 1 else sum(f["x"] for f in flames) / len(flames)

    rnd = random.Random(day * 13)
    dust = []
    for _ in range(150):
        spr, s = sprite(rnd.uniform(1.3, 3.2))
        dust.append(dict(x=rnd.uniform(0, W), y=rnd.uniform(0, H), sp=rnd.uniform(22, 80), ph=rnd.uniform(0, 6.28),
                         tw=rnd.uniform(0, 6.28), spr=spr, s=s))

    out = f"/home/claude/social/feel_day{day:02d}.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", out]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    stills = {int((starts[0] + 1.1) * FPS): "a", int((starts[2] + 1.1) * FPS): "b", int((tc + 2.4) * FPS): "c",
              int((DUR - 0.2) * FPS): "d"}

    for i in range(int(DUR * FPS)):
        t = i / FPS
        ap = ss(t / 2.0)
        specs = [(gx, 1310, 360 + 80 * ss(t / 3), 0.30 * ap * (1 + 0.04 * math.sin(t * 2.3)), GOLD),
                 (gx, 1330, 140, 0.12 * ap, (255, 226, 168)),
                 (770 + 30 * math.sin(t * 0.5), 300, 640, 0.17, (63, 90, 140)),
                 (120 + 40 * math.cos(t * 0.4), 1700, 560, 0.24, (40, 52, 80))]
        frame = glow_layer(W, H, specs)

        fade = ss(t / 1.6)
        for p in dust:
            if cfg["dust"] == "up":
                y = (p["y"] - p["sp"] * t) % (H + 60) - 30
                x = p["x"] + 16 * math.sin(p["ph"] + t * 0.6)
            elif cfg["dust"] == "side":
                x = (p["x"] + p["sp"] * 0.9 * t) % (W + 60) - 30
                y = p["y"] + 14 * math.sin(p["ph"] + t * 0.6)
            else:  # press: drifts down under pressure, then lifts at the last line
                ts = starts[2]
                yy = p["y"] + p["sp"] * min(t, ts) - p["sp"] * 1.1 * max(t - ts, 0)
                y = yy % (H + 60) - 30
                x = p["x"] + 14 * math.sin(p["ph"] + t * 0.6)
            a = (0.28 + 0.28 * math.sin(p["tw"] + t * 1.7)) * fade
            s = p["s"]
            xi, yi = int(x) - s // 2, int(y) - s // 2
            if 0 <= xi < W - s and 0 <= yi < H - s:
                frame[yi:yi + s, xi:xi + s] += p["spr"][..., None] * G * (a * 1.6)

        # flames fade out as the closing card arrives
        fout = 1 - ss((t - tc) / 0.8)
        for fl in flames:
            fo = ss((t - fl["at"]) / 1.3) * fout
            if fo > 0:
                im, anc = flame_img(fl["k"], t + fl["x"] * 0.003)
                paste_flame_np(frame, im, anc, fl["x"], 1360, fo)

        # lines, one at a time
        for k, b in enumerate(blocks):
            e = starts[k] + step if k < n - 1 else tc - 0.1
            pin = ss((t - starts[k]) / 0.7)
            pout = 1 - ss((t - (e - 0.5)) / 0.5)
            op = pin * pout
            if op <= 0:
                continue
            m = b["m"]
            sig = (1 - pin) * 8
            if sig > 0.4:
                m = cv2.GaussianBlur(m, (0, 0), sig)
            y = int(820 - b["h"] / 2 + (1 - pin) * 22)
            blend(frame, m, b["c"], 0, y, op)

        # closing card
        lo = ss((t - tc) / 1.0)
        if lo > 0:
            add_light(frame, LS, lox, loy, 540, 700, lo)
            base_cta = 905
            if cl["sub"]:
                m1, b1 = sc_m
                p1 = ss((t - tc - 0.5) / 0.8)
                blend(frame, m1, G if False else P, (W - m1.shape[1]) // 2, int(880 - b1 + (1 - p1) * 14), p1)
                m2, b2 = sub_m
                p2 = ss((t - tc - 1.0) / 0.8)
                blend(frame, m2, PD, (W - m2.shape[1]) // 2, int(945 - b2 + (1 - p2) * 14), p2)
                base_cta = 1075 if not cl["small"] else 1180
                for j, (ms, bs) in enumerate(small_m):
                    pj = ss((t - tc - 2.0 - 0.3 * j) / 0.8)
                    blend(frame, ms, P, (W - ms.shape[1]) // 2, int(1025 + j * 48 - bs + (1 - pj) * 10), pj * 0.92)
            p3 = ss((t - tc - (1.8 if cl["sub"] else 0.6)) / 0.8)
            blend(frame, cta_m, G, (W - cta_m.shape[1]) // 2, int(base_cta - cta_b + (1 - p3) * 12), p3)

        o8 = (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
        if i in stills:
            Image.fromarray(o8).save(f"/home/claude/social/fstill_{day}_{stills[i]}.png")
        pipe.stdin.write(o8.tobytes())
    pipe.stdin.close()
    pipe.wait()
    print("reel done", out, pipe.returncode)


# ---------------- carousels ----------------
def corners_img(img):
    Wd, Hd = img.size
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    c = GOLD_DIM + (190,)
    d.line([(64, 144), (64, 64), (144, 64)], fill=c, width=2)
    d.line([(Wd - 144, Hd - 64), (Wd - 64, Hd - 64), (Wd - 64, Hd - 144)], fill=c, width=2)
    return Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")


def carousel(day):
    W, H = 1080, 1350
    cfg = DAYS[day]
    lines = cfg["lines"]
    cl = CLOSE[cfg["closing"]]
    total = len(lines) + 1
    LS, lox, loy = lock_scaled(780)
    fcount = [dict(x=f["x"], h=int(f["h"] * 0.75)) for f in cfg["flames"]]
    for idx in range(total):
        specs = [(sum(f["x"] for f in fcount) / len(fcount), 1140, 480, 0.30, GOLD), (540, 1150, 170, 0.12, (255, 226, 168)),
                 (800, 200, 600, 0.18, (63, 90, 140)), (120, 1150, 520, 0.24, (40, 52, 80))]
        arr = dust_static(glow_layer(W, H, specs), W, H, day * 31 + idx, n=90)
        closing = idx == len(lines)
        shift = {"plain": 120, "ai3": 85, "ai4": 50}[cfg["closing"]]
        if closing:
            add_light(arr, LS, lox, loy, 540, 470 + shift, 1.0)
        img = Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))
        img = corners_img(img)
        d = ImageDraw.Draw(img)
        draw_tracked(d, f"{idx + 1:02d}  /  {total:02d}", font("Inter", 30, 500), 150 + 62, 142, GOLD_SOFT, 5)
        if not closing:
            # flames for this slide (the second flame of day 4 lights on the last line)
            for j, f in enumerate(fcount):
                if cfg["flames"][j]["at"] == "line3" and idx < 2:
                    continue
                im, anc = flame_img(f["h"] / 44, 0.9 + j * 0.4)
                img.paste(im, (f["x"] - anc[0], 1190 - anc[1]), im)
            last = idx == len(lines) - 1
            fname = "CormorantGaramond-Italic" if last else "CormorantGaramond"
            size, f, ls = choose_size_f(lines[idx], fname, 860, 560)
            lh = size * 1.18
            top, bottom = 250, 960
            base = top + (bottom - top - lh * len(ls)) / 2 + size * 0.95
            sp = f.getlength(" ")
            for ln, cw in ls:
                x = (W - cw) / 2
                for w, wl in ln:
                    d.text((x, base), w, font=f, fill=GOLD if last else PARCH, anchor="ls")
                    x += wl + sp
                base += lh
        else:
            fi = font("CormorantGaramond-Italic", 62, 500)
            if cl["sub"]:
                t1 = "Something is coming."
                d.text((W / 2 - fi.getlength(t1) / 2, 700 + shift), t1, font=fi, fill=PARCH, anchor="ls")
                fs = font("Inter", 32, 500)
                d.text((W / 2 - fs.getlength(cl["sub"]) / 2, 762 + shift), cl["sub"], font=fs, fill=PARCH_DIM, anchor="ls")
                yy = 840 + shift
                for t in (cl["small"] or []):
                    fsm = font("Inter", 30, 500)
                    d.text((W / 2 - fsm.getlength(t) / 2, yy), t, font=fsm, fill=PARCH, anchor="ls")
                    yy += 46
                cta_y = yy + 70 if cl["small"] else 880 + shift
            else:
                cta_y = 700 + shift
            draw_tracked(d, "FOLLOW FOR LAUNCH DAY", font("Inter", 34, 500), W / 2, cta_y, GOLD, 8)
        img.save(f"/home/claude/social/feel_day{day:02d}_slide{idx + 1}.png")
    print("carousel done", day, total)


# ---------------- announcement graphics ----------------
def announce():
    sizes = {
        "instagram_1080x1350": dict(W=1080, H=1350, A=(62, 470), L=(780, 650), S=(32, 910), C=(28, 980)),
        "x_1600x900": dict(W=1600, H=900, A=(66, 270), L=(780, 430), S=(34, 640), C=(30, 705)),
        "linkedin_1200x627": dict(W=1200, H=627, A=(48, 190), L=(580, 300), S=(26, 440), C=(23, 492)),
    }
    for name, c in sizes.items():
        W, H = c["W"], c["H"]
        specs = [(W / 2, H * 0.9, W * 0.42, 0.26, GOLD), (W * 0.75, H * 0.1, W * 0.5, 0.16, (63, 90, 140)),
                 (W * 0.1, H * 0.85, W * 0.4, 0.22, (40, 52, 80))]
        arr = dust_static(glow_layer(W, H, specs), W, H, 77, n=int(70 * W * H / (1080 * 1350)) + 25)
        LSa, ox, oy = lock_scaled(c["L"][0])
        add_light(arr, LSa, ox, oy, W / 2, c["L"][1], 1.0)
        img = Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))
        d = ImageDraw.Draw(img)
        fa = font("CormorantGaramond-Italic", c["A"][0], 500)
        t = "This account is now"
        d.text((W / 2 - fa.getlength(t) / 2, c["A"][1]), t, font=fa, fill=PARCH_DIM, anchor="ls")
        fs = font("Inter", c["S"][0], 400)
        t2 = "AI mentors built from what history's wisest people wrote."
        d.text((W / 2 - fs.getlength(t2) / 2, c["S"][1]), t2, font=fs, fill=PARCH, anchor="ls")
        draw_tracked(d, "OPENING SOON.  FOLLOW FOR LAUNCH DAY.", font("Inter", c["C"][0], 500), W / 2, c["C"][1], GOLD, 7)
        img.save(f"/home/claude/social/announce_{name}.png")
    print("announce done")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "reel":
        reel(int(sys.argv[2]))
    elif what == "cards":
        for dd in DAYS:
            carousel(dd)
        announce()
