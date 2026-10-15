import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from social_lib import *

OUT = "/home/claude/social/"


def base_img(W, H, specs, seed, dust=60):
    arr = glow_layer(W, H, specs)
    arr = dust_static(arr, W, H, seed, n=dust)
    return Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))


# ---------- avatar (flame) ----------
def avatar_flame(S=1080):
    specs = [(S / 2, S * 0.60, S * 0.46, 0.46, GOLD), (S / 2, S * 0.62, S * 0.20, 0.20, (255, 226, 168)),
             (S * 0.8, S * 0.15, S * 0.55, 0.12, (63, 90, 140))]
    img = base_img(S, S, specs, 5, dust=45)
    fl, anc = flame_img(S * 0.56 / 44, 0.9)
    bx, by = S // 2, int(S * 0.80)
    img.paste(fl, (bx - anc[0], by - anc[1]), fl)
    return img


# ---------- avatar (logo cut-out) ----------
def avatar_logo(S=1080):
    specs = [(S / 2, S * 0.55, S * 0.46, 0.40, GOLD), (S / 2, S * 0.55, S * 0.20, 0.14, (255, 226, 168))]
    img = base_img(S, S, specs, 6, dust=45).convert("RGBA")
    lg = Image.open(OUT + "logo_cut.png").convert("RGBA")
    w = int(S * 0.58)
    lg = lg.resize((w, int(lg.height * w / lg.width)), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.7))
    img.alpha_composite(lg, ((S - lg.width) // 2, int(S * 0.52 - lg.height / 2)))
    return img.convert("RGB")


# ---------- X header 1500x500 ----------
def x_header():
    W, H = 1500, 500
    specs = [(1280, 470, 520, 0.34, GOLD), (1280, 460, 200, 0.14, (255, 226, 168)),
             (350, 60, 620, 0.16, (63, 90, 140)), (60, 480, 420, 0.20, (40, 52, 80))]
    img = base_img(W, H, specs, 9, dust=70)
    d = ImageDraw.Draw(img)
    fl, anc = flame_img(250 / 44, 1.2)
    img.paste(fl, (1255 - anc[0], 420 - anc[1]), fl)
    f1 = font("CormorantGaramond", 80, 500)
    f2 = font("CormorantGaramond-Italic", 80, 500)
    f3 = font("Inter", 25, 400)
    x = 410
    d.text((x, 205), "The counsel of the ages,", font=f1, fill=PARCH, anchor="ls")
    d.text((x, 295), "kept for you.", font=f2, fill=GOLD, anchor="ls")
    draw_tracked(d, "AI MENTORS BUILT FROM WHAT HISTORY'S WISEST PEOPLE WROTE", font("Inter", 19, 500), 0, 0, GOLD, 0) if False else None
    d.text((x, 350), "AI mentors built from what history's wisest people wrote.", font=f3, fill=PARCH_DIM, anchor="ls")
    d.text((x, 392), "mentormindhq.com", font=font("Inter", 25, 500), fill=GOLD_SOFT, anchor="ls")
    return img


# ---------- LinkedIn cover 2256x382 (2x of 1128x191) ----------
def li_cover():
    W, H = 2256, 382
    specs = [(1950, 400, 520, 0.30, GOLD), (1950, 390, 200, 0.12, (255, 226, 168)),
             (700, 40, 700, 0.15, (63, 90, 140))]
    img = base_img(W, H, specs, 11, dust=80)
    d = ImageDraw.Draw(img)
    fl, anc = flame_img(210 / 44, 0.7)
    img.paste(fl, (2050 - anc[0], 350 - anc[1]), fl)
    f1 = font("CormorantGaramond", 88, 500)
    f2 = font("CormorantGaramond-Italic", 88, 500)
    x = 600
    t1 = "The counsel of the ages, "
    d.text((x, 200), t1, font=f1, fill=PARCH, anchor="ls")
    d.text((x + f1.getlength(t1), 200), "kept for you.", font=f2, fill=GOLD, anchor="ls")
    d.text((x, 275), "AI mentors built from what history's wisest people wrote.", font=font("Inter", 36, 400), fill=PARCH_DIM, anchor="ls")
    d.text((x, 330), "mentormindhq.com", font=font("Inter", 36, 500), fill=GOLD_SOFT, anchor="ls")
    return img


af = avatar_flame()
al = avatar_logo()
af.save(OUT + "avatar_flame_1080.png")
af.resize((400, 400), Image.LANCZOS).save(OUT + "avatar_flame_400.png")
al.save(OUT + "avatar_logo_1080.png")
al.resize((400, 400), Image.LANCZOS).save(OUT + "avatar_logo_400.png")
xh = x_header()
xh.save(OUT + "x_header_1500x500.png")
lc = li_cover()
lc.save(OUT + "linkedin_cover_2256x382.png")


# ---------- review sheet with circle previews ----------
def circle(im, s):
    im = im.resize((s, s), Image.LANCZOS).convert("RGBA")
    m = Image.new("L", (s, s), 0)
    ImageDraw.Draw(m).ellipse([0, 0, s - 1, s - 1], fill=255)
    out = Image.new("RGBA", (s, s), (30, 30, 34, 255))
    out.paste(im, (0, 0), m)
    return out


sheet = Image.new("RGB", (1180, 880), (30, 30, 34))
sheet.paste(circle(af, 300), (30, 30))
sheet.paste(circle(al, 300), (360, 30))
sheet.paste(circle(af, 64), (700, 30))
sheet.paste(circle(al, 64), (790, 30))
sheet.paste(circle(af, 110), (700, 120))
sheet.paste(circle(al, 110), (830, 120))
sheet.paste(xh.resize((1120, 373), Image.LANCZOS), (30, 360))
sheet.paste(lc.resize((1120, 190), Image.LANCZOS), (30, 760 - 0))
sheet.save(OUT + "profile_sheet.png")
print("done", sheet.size)
