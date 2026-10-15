import numpy as np
from PIL import Image, ImageDraw

# reuse the loading and helper code from the profile script, without running its renders
src = open("/home/claude/social/render_profile2.py").read()
exec(src[:src.index("a1080, a400")])

W, H = 1584, 396


def personal_banner():
    specs = [(1180, 210, 520, 0.12, GOLD), (400, 30, 620, 0.14, (63, 90, 140)), (80, 380, 420, 0.18, (40, 52, 80))]
    c = canvas_bg(W, H, specs, 21, 60)
    scale = 560 / lw_
    cx = 792 - ((lx0 + lx1) / 2 - dx0 - lock_c.shape[1] / 2) * scale
    cy = 118 - ((ly0 + ly1) / 2 - dy0 - lock_c.shape[0] / 2) * scale
    put_add(c, lock_c, cx, cy, scale)
    img = Image.fromarray((np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8))
    d = ImageDraw.Draw(img)
    f1 = font("CormorantGaramond-Italic", 40, 500)
    t = "The counsel of the ages, kept for you."
    d.text((792 - f1.getlength(t) / 2, 230), t, font=f1, fill=PARCH_DIM, anchor="ls")
    return img


b = personal_banner()
b.save(OUT + "linkedin_personal_banner_1584x396.png")

# review copy: show where a profile photo can cover the banner (desktop bottom-left, some phones bottom-centre)
chk = b.copy().convert("RGBA")
ov = Image.new("RGBA", chk.size, (0, 0, 0, 0))
od = ImageDraw.Draw(ov)
for cx_ in (200, 792):
    od.ellipse([cx_ - 150, 246, cx_ + 150, 546], fill=(255, 60, 60, 70), outline=(255, 60, 60, 220), width=3)
od.rectangle([117, 88, 1467, 308], outline=(80, 200, 255, 220), width=2)   # central safe band
chk = Image.alpha_composite(chk, ov).convert("RGB")
chk.save(OUT + "personal_banner_check.png")
print("done", b.size)
