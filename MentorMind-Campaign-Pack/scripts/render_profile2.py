import numpy as np
import cv2
from PIL import Image, ImageDraw
from social_lib import *

OUT = "/home/claude/social/"
MARK = "/mnt/project/MentorMind_Logo_s.png"
SHEET = "/mnt/project/MentorMind_Logo.png"


def load_add(path, box=None, border=6, floor=0.05, taper=0.0):
    from scipy.signal.windows import tukey
    im = Image.open(path).convert("RGB")
    if box:
        im = im.crop(box)
    a = np.asarray(im).astype(np.float32)
    # trim white edges left over from the export (right and bottom of the small mark file)
    while a.shape[1] > 20 and a[:, -1].mean() > 150:
        a = a[:, :-1]
    while a.shape[0] > 20 and a[-1, :].mean() > 150:
        a = a[:-1, :]
    a = a[:, :-3][:-3, :] if box is None else a
    edge = np.concatenate([a[:border].reshape(-1, 3), a[-border:].reshape(-1, 3),
                           a[:, :border].reshape(-1, 3), a[:, -border:].reshape(-1, 3)])
    ref = np.median(edge, axis=0)
    add = np.clip((a - ref) / 255.0, 0, 1)
    add = np.clip((add - floor) / (1 - floor), 0, 1)       # drop the faint background halo
    h, w = add.shape[:2]
    return add.astype(np.float32)


def fade_edges(arr, px=40):
    """Fade the outer px pixels to nothing, so a cropped glow never shows a hard edge."""
    h, w = arr.shape[:2]

    def ramp(n):
        k = min(px, n // 2)
        r = np.ones(n, np.float32)
        t = np.linspace(0, 1, k, dtype=np.float32)
        r[:k] = 0.5 - 0.5 * np.cos(np.pi * t)
        r[n - k:] = r[:k][::-1]
        return r

    return (arr * np.outer(ramp(h), ramp(w))[..., None]).astype(np.float32)


def bbox_of(add, thr=0.2):
    m = add.max(axis=2) > thr
    ys, xs = np.where(m)
    return xs.min(), ys.min(), xs.max(), ys.max()


def put_add(canvas, add, cx, cy, scale):
    """Add a light layer onto canvas (float HxWx3), centred at cx,cy, scaled."""
    h, w = add.shape[:2]
    nw, nh = int(w * scale), int(h * scale)
    big = cv2.resize(add, (nw, nh), interpolation=cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA)
    x0, y0 = int(cx - nw / 2), int(cy - nh / 2)
    H, W = canvas.shape[:2]
    x1, y1 = max(x0, 0), max(y0, 0)
    x2, y2 = min(x0 + nw, W), min(y0 + nh, H)
    canvas[y1:y2, x1:x2] += big[y1 - y0:y2 - y0, x1 - x0:x2 - x0]


mark = load_add(MARK)
mx0, my0, mx1, my1 = bbox_of(mark)
print("mark bbox", mx0, my0, mx1, my1, "of", mark.shape)
tw, th = mx1 - mx0, my1 - my0                      # tight size of the artwork
pad = 70
cy0, cx0 = max(my0 - pad, 0), max(mx0 - pad, 0)
cy1, cx1 = min(my1 + pad, mark.shape[0] - 1), min(mx1 + pad, mark.shape[1] - 1)
mark_c = fade_edges(mark[cy0:cy1 + 1, cx0:cx1 + 1], 55)
mark_off = ((mx0 + mx1) / 2 - cx0 - mark_c.shape[1] / 2, (my0 + my1) / 2 - cy0 - mark_c.shape[0] / 2)

# the horizontal lockup (mark + wordmark) from the dark panel of the brand sheet
lock = load_add(SHEET, box=(60, 60, 1330, 440))
lx0, ly0, lx1, ly1 = bbox_of(lock, 0.12)
lw_, lh_ = lx1 - lx0, ly1 - ly0
lpad = 60
dx0, dy0 = max(lx0 - lpad, 0), max(ly0 - lpad, 0)
dx1, dy1 = min(lx1 + lpad, lock.shape[1] - 1), min(ly1 + lpad, lock.shape[0] - 1)
lock_c = fade_edges(lock[dy0:dy1 + 1, dx0:dx1 + 1], 40)
print("lockup crop", lock_c.shape)


def canvas_bg(W, H, specs, seed, dust):
    arr = glow_layer(W, H, specs)
    return dust_static(arr, W, H, seed, n=dust)


def avatar(S):
    specs = [(S / 2, S * 0.52, S * 0.50, 0.10, GOLD)]
    c = canvas_bg(S, S, specs, 5, 26)
    rad = 0.40 * S
    scale = rad / np.hypot(tw / 2, th / 2) * 1.22
    lift = 0.030 * S
    put_add(c, mark_c, S / 2 - mark_off[0] * scale + 0.005 * S, S * 0.50 - lift - mark_off[1] * scale, scale)
    return Image.fromarray((np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8))


def x_header():
    W, H = 1500, 500
    specs = [(1100, 250, 520, 0.12, GOLD), (350, 60, 620, 0.14, (63, 90, 140)), (60, 480, 420, 0.18, (40, 52, 80))]
    c = canvas_bg(W, H, specs, 9, 60)
    scale = 640 / lw_
    put_add(c, lock_c, 750 - ((lx0 + lx1) / 2 - dx0 - lock_c.shape[1] / 2) * scale, 205 - ((ly0 + ly1) / 2 - dy0 - lock_c.shape[0] / 2) * scale, scale)
    img = Image.fromarray((np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8))
    d = ImageDraw.Draw(img)
    f1 = font("CormorantGaramond-Italic", 42, 500)
    t = "The counsel of the ages, kept for you."
    d.text((750 - f1.getlength(t) / 2, 380), t, font=f1, fill=PARCH_DIM, anchor="ls")
    fw = font("Inter", 24, 500)
    draw_tracked(d, "MENTORMINDHQ.COM", fw, 750, 428, GOLD_SOFT, 6)
    return img


def li_cover():
    W, H = 2256, 382
    specs = [(1900, 300, 560, 0.12, GOLD), (700, 40, 700, 0.14, (63, 90, 140))]
    c = canvas_bg(W, H, specs, 11, 70)
    scale = 215 / lh_
    put_add(c, lock_c, 1128 - ((lx0 + lx1) / 2 - dx0 - lock_c.shape[1] / 2) * scale, 135 - ((ly0 + ly1) / 2 - dy0 - lock_c.shape[0] / 2) * scale, scale)
    img = Image.fromarray((np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8))
    d = ImageDraw.Draw(img)
    f1 = font("CormorantGaramond-Italic", 58, 500)
    t = "The counsel of the ages, kept for you."
    d.text((1128 - f1.getlength(t) / 2, 330), t, font=f1, fill=PARCH_DIM, anchor="ls")
    return img


a1080, a400 = avatar(800), avatar(400)
a1080.save(OUT + "avatar_logo_800.png")
a400.save(OUT + "avatar_logo_400.png")
xh = x_header()
xh.save(OUT + "x_header_1500x500.png")
lc = li_cover()
lc.save(OUT + "linkedin_cover_2256x382.png")


def circle(im, s):
    im = im.resize((s, s), Image.LANCZOS).convert("RGBA")
    m = Image.new("L", (s, s), 0)
    ImageDraw.Draw(m).ellipse([0, 0, s - 1, s - 1], fill=255)
    out = Image.new("RGBA", (s, s), (30, 30, 34, 255))
    out.paste(im, (0, 0), m)
    return out


sheet = Image.new("RGB", (1180, 800), (30, 30, 34))
sheet.paste(circle(a1080, 300), (30, 20))
sheet.paste(circle(a1080, 110), (360, 20))
sheet.paste(circle(a1080, 64), (500, 20))
sheet.paste(a1080.resize((300, 300), Image.LANCZOS), (600, 20))
sheet.paste(xh.resize((1120, 373), Image.LANCZOS), (30, 340))
sheet.paste(lc.resize((1120, 189), Image.LANCZOS), (30, 600))
sheet.save(OUT + "profile_sheet2.png")
print("done")
