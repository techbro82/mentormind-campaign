import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1350
GOLD = (216, 166, 87)
GOLD_SOFT = (202, 162, 94)
GOLD_DIM = (138, 116, 68)
PARCH = (236, 227, 210)
PARCH_DIM = (169, 162, 144)
FF = "/tmp/fonts/"
OUTDIR = "/home/claude/social/"


def font(name, size, wght=500):
    f = ImageFont.truetype(FF + name + ".ttf", size)
    vals = []
    for a in f.get_variation_axes():
        n = a["name"].decode() if isinstance(a["name"], bytes) else a["name"]
        vals.append(wght if "eight" in n else a["default"])
    f.set_variation_by_axes(vals)
    return f


def background(seed):
    hw, hh = W // 2, H // 2
    Y, X = np.mgrid[0:hh, 0:hw].astype(np.float32) * 2

    def glow(cx, cy, r, inten, col):
        g = inten * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (r * r)))
        return g[..., None] * (np.array(col, np.float32) / 255)

    lay = np.ones((hh, hw, 3), np.float32) * (np.array((14, 17, 26), np.float32) / 255)
    lay += glow(540, 1230, 520, 0.34, GOLD)
    lay += glow(560, 1260, 200, 0.14, (255, 226, 168))
    lay += glow(800, 220, 600, 0.20, (63, 90, 140))
    lay += glow(120, 1200, 520, 0.26, (40, 52, 80))
    vig = 1 - 0.36 * (((X - W / 2) ** 2 + (Y - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)) ** 1.1
    lay *= vig[..., None]
    img = cv2.resize(lay, (W, H), interpolation=cv2.INTER_LINEAR)
    rnd = random.Random(seed)
    for _ in range(110):
        r = rnd.uniform(1.2, 3.0)
        s = int(r * 4) | 1
        ax = np.arange(s) - s // 2
        spr = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (r * r * 0.9)).astype(np.float32)
        x, y = rnd.randint(0, W - s), rnd.randint(0, H - s)
        a = rnd.uniform(0.15, 0.55)
        img[y:y + s, x:x + s] += spr[..., None] * (np.array(GOLD, np.float32) / 255) * a * 1.6
    return Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))


def tracked(d, txt, f, cx, base, fill, tracking):
    ws = [f.getlength(c) for c in txt]
    total = sum(ws) + tracking * (len(txt) - 1)
    x = cx - total / 2
    for c, w in zip(txt, ws):
        d.text((x, base), c, font=f, fill=fill, anchor="ls")
        x += w + tracking


def draw_runs(d, runs, size, area_top, area_bottom, max_w=860):
    fr, fg = font("CormorantGaramond", size, 500), font("CormorantGaramond-Italic", size, 500)
    words = []
    for text, style in runs:
        for tok in text.split():
            words.append((tok, style))
    space = fr.getlength(" ")
    lines, cur, cw = [], [], 0
    for tok, style in words:
        f = fg if style == "g" else fr
        w = f.getlength(tok)
        add = w if not cur else w + space
        if cur and cw + add > max_w:
            lines.append((cur, cw))
            cur, cw = [(tok, style, w)], w
        else:
            cur.append((tok, style, w))
            cw += add
    lines.append((cur, cw))
    lh = size * 1.2
    block = lh * len(lines)
    base = area_top + (area_bottom - area_top - block) / 2 + size * 0.95
    for ln, cw in lines:
        x = (W - cw) / 2
        for tok, style, w in ln:
            if style == "g":
                d.text((x, base), tok, font=fg, fill=GOLD, anchor="ls")
            else:
                d.text((x, base), tok, font=fr, fill=PARCH, anchor="ls")
            x += w + space
        base += lh


def corners(img):
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    c = GOLD_DIM + (190,)
    a, m = 80, 64
    d.line([(m, m + a), (m, m), (m + a, m)], fill=c, width=2)
    d.line([(W - m - a, H - m), (W - m, H - m), (W - m, H - m - a)], fill=c, width=2)
    return Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")


SLIDES = [
    ([("Not an AI in a ", "r"), ("costume.", "g")], 150),
    ([("Every mentor is built only from what the real person ", "r"), ("actually wrote.", "g")], 104),
    ([("Marcus Aurelius", "g"), (" draws on his Meditations. ", "r"), ("Augustine,", "g"), (" on his Confessions.", "r")], 100),
    ([("And they ", "r"), ("remember you.", "g"), (" Return, and they ask what came of your last resolve.", "r")], 96),
    ([("Plainly: these are ", "r"), ("AI voices", "g"), (" built from the writings, not the real people.", "r")], 90),
]

f_count = font("Inter", 30, 500)
f_word = font("CormorantGaramond", 60, 500)
f_wordg = font("CormorantGaramond", 60, 500)
f_cta = font("Inter", 31, 500)

for i, (runs, size) in enumerate(SLIDES, 1):
    img = corners(background(i * 11))
    d = ImageDraw.Draw(img)
    tracked(d, f"{i:02d}  /  {len(SLIDES):02d}", f_count, 150 + 62, 142, GOLD_SOFT, 5)
    draw_runs(d, runs, size, 220, H - 300)
    # wordmark
    wa, wb = f_word.getlength("Mentor"), f_wordg.getlength("Mind")
    x = (W - (wa + wb)) / 2
    d.text((x, H - 150), "Mentor", font=f_word, fill=PARCH, anchor="ls")
    d.text((x + wa, H - 150), "Mind", font=f_wordg, fill=GOLD, anchor="ls")
    if i == len(SLIDES):
        tracked(d, "FOLLOW FOR LAUNCH DAY", f_cta, W / 2, H - 96, GOLD_SOFT, 7)
    img.save(f"{OUTDIR}day03_slide{i}.png")
print("slides done")
