import re, math, random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

GOLD = (216, 166, 87)
GOLD_SOFT = (202, 162, 94)
GOLD_DIM = (138, 116, 68)
PARCH = (236, 227, 210)
PARCH_DIM = (169, 162, 144)
FF = "/tmp/fonts/"


def font(name, size, wght=500):
    f = ImageFont.truetype(FF + name + ".ttf", size)
    vals = []
    for a in f.get_variation_axes():
        n = a["name"].decode() if isinstance(a["name"], bytes) else a["name"]
        vals.append(wght if "eight" in n else a["default"])
    f.set_variation_by_axes(vals)
    return f


def ss(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def draw_tracked(d, txt, f, cx, base, fill, tr):
    ws = [f.getlength(c) for c in txt]
    total = sum(ws) + tr * (len(txt) - 1)
    x = cx - total / 2
    for c, w in zip(txt, ws):
        d.text((x, base), c, font=f, fill=fill, anchor="ls")
        x += w + tr


def glow_layer(W, H, specs):
    hw, hh = W // 2, H // 2
    Y, X = np.mgrid[0:hh, 0:hw].astype(np.float32) * 2
    lay = np.ones((hh, hw, 3), np.float32) * (np.array((14, 17, 26), np.float32) / 255)
    for cx, cy, r, i, c in specs:
        g = i * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (r * r)))
        lay += g[..., None] * (np.array(c, np.float32) / 255)
    vig = 1 - 0.36 * (((X - W / 2) ** 2 + (Y - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)) ** 1.1
    lay *= vig[..., None]
    return cv2.resize(lay, (W, H), interpolation=cv2.INTER_LINEAR)


def sprite(r):
    s = int(r * 4) | 1
    ax = np.arange(s) - s // 2
    return np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (r * r * 0.9)).astype(np.float32), s


def dust_static(img, W, H, seed, n=110):
    rnd = random.Random(seed)
    for _ in range(n):
        spr, s = sprite(rnd.uniform(1.2, 3.0))
        x, y = rnd.randint(0, W - s), rnd.randint(0, H - s)
        a = rnd.uniform(0.15, 0.55)
        img[y:y + s, x:x + s] += spr[..., None] * (np.array(GOLD, np.float32) / 255) * a * 1.6
    return img


def layout_words(text, f, max_w):
    space = f.getlength(" ")
    lines, cur, cw = [], [], 0
    for w in text.split():
        wl = f.getlength(w)
        add = wl if not cur else wl + space
        if cur and cw + add > max_w:
            lines.append((cur, cw))
            cur, cw = [(w, wl)], wl
        else:
            cur.append((w, wl))
            cw += add
    lines.append((cur, cw))
    return lines


def choose_size(text, max_w, max_h):
    for s in [124, 114, 104, 96, 90, 84, 78, 72, 66, 60]:
        f = font("CormorantGaramond", s, 500)
        for mw in range(int(max_w), int(max_w) - 330, -30):
            lines = layout_words(text, f, mw)
            if len(lines) > 1 and len(lines[-1][0]) < 2:
                continue  # no single word alone on the last line
            if len(lines) * s * 1.18 <= max_h:
                return s, f, lines
            break
    f = font("CormorantGaramond", 56, 500)
    return 56, f, layout_words(text, f, max_w)


# ---- the real flame from Home's streak card, drawn from its SVG path data
def _parse(d):
    toks = re.findall(r"[MCZ]|-?\d+\.?\d*", d)
    pts, i, cur = [], 0, (0.0, 0.0)
    while i < len(toks):
        c = toks[i]
        i += 1
        if c == "M":
            cur = (float(toks[i]), float(toks[i + 1]))
            i += 2
            pts.append(cur)
        elif c == "C":
            x1, y1, x2, y2, x, y = [float(v) for v in toks[i:i + 6]]
            i += 6
            p0 = cur
            for s in range(1, 25):
                t = s / 24
                m = 1 - t
                pts.append((m ** 3 * p0[0] + 3 * m * m * t * x1 + 3 * m * t * t * x2 + t ** 3 * x,
                            m ** 3 * p0[1] + 3 * m * m * t * y1 + 3 * m * t * t * y2 + t ** 3 * y))
            cur = (x, y)
    return pts


_D1 = "M15 43 C6 38 3 28 8 18 C10 14 9 9 12 3 C13 10 18 12 20 19 C24 14 23 7 21 1 C29 9 30 22 25 30 C29 27 30 21 29 17 C31 27 27 38 15 43 Z"
_D2 = "M15 41 C8 36 6 27 10 19 C11.5 15.5 11 11 12.5 6 C14 12 17.5 13.5 19 19 C22 15 21 8.5 19.5 4 C25 11 25.5 21 22 27 C25 25 25.5 20 24.5 17 C26.5 25 23.5 34 15 41 Z"
_D3 = "M15 38 C10 34 9 27 11.5 21 C12.5 18.5 12.5 15 13.5 11 C14.5 15.5 16.5 16.5 17.5 21 C19.5 18 19 13.5 18 10 C21.5 15 22 22 19.5 26 C21.5 24.5 22 20.5 21 18 C22.5 24 20.5 31 15 38 Z"
# (points, colour, opacity, period s, scaleY at peak, rotation at peak)  - from fmFlick1/2/3
PATHS = [(_parse(_D1), (168, 72, 58), 0.5, 2.6, 1.06, -2.0),
         (_parse(_D2), (216, 166, 87), 0.85, 2.1, 0.94, 2.0),
         (_parse(_D3), (240, 196, 121), 1.0, 1.7, 1.08, 0.0)]


def flame_img(k, t):
    """RGBA flame image, k = pixels per SVG unit. Returns (image, (base_x, base_y)) anchor of the flame base."""
    S, cw, ch, ox, oy = 4, 36, 51, 3, 5
    Wd, Hd = int(cw * k * S), int(ch * k * S)
    base = Image.new("RGBA", (Wd, Hd), (0, 0, 0, 0))
    for pts, col, op, per, sy, rot in PATHS:
        f = (1 - math.cos(2 * math.pi * t / per)) / 2
        s = 1 + (sy - 1) * f
        th = math.radians(rot * f)
        ct, st = math.cos(th), math.sin(th)
        poly = []
        for x, y in pts:
            dx, dy = x - 15, y - 44
            rx, ry = dx * ct - dy * st, dx * st + dy * ct
            poly.append(((15 + rx + ox) * k * S, (44 + ry * s + oy) * k * S))
        layer = Image.new("RGBA", (Wd, Hd), (0, 0, 0, 0))
        ImageDraw.Draw(layer).polygon(poly, fill=col + (int(255 * op),))
        base = Image.alpha_composite(base, layer)
    layer = Image.new("RGBA", (Wd, Hd), (0, 0, 0, 0))
    cx, cy, rx, ry = 15, 34, 2, 3.5
    ImageDraw.Draw(layer).ellipse([(cx - rx + ox) * k * S, (cy - ry + oy) * k * S,
                                   (cx + rx + ox) * k * S, (cy + ry + oy) * k * S], fill=(255, 248, 232, 230))
    base = Image.alpha_composite(base, layer)
    out = base.convert("RGBa").resize((int(cw * k), int(ch * k)), Image.LANCZOS).convert("RGBA")
    return out, (int((15 + ox) * k), int((44 + oy) * k))


def paste_flame_np(frame, img, anchor, bx, by, op=1.0):
    arr = np.asarray(img, np.float32) / 255.0
    h, w = arr.shape[:2]
    x0, y0 = bx - anchor[0], by - anchor[1]
    x1, y1 = max(x0, 0), max(y0, 0)
    x2, y2 = min(x0 + w, frame.shape[1]), min(y0 + h, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    a = arr[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
    al = a[..., 3:4] * op
    reg = frame[y1:y2, x1:x2]
    reg[:] = reg * (1 - al) + a[..., :3] * al
