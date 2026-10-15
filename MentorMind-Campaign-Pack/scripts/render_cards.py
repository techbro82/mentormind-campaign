import numpy as np
from PIL import Image, ImageDraw
from social_lib import *

W, H = 1080, 1350
OUT = "/home/claude/social/"

QUOTES = [
    dict(day=4, name="EPICTETUS", cat="ADVERSITY", era="a Stoic school at Nicopolis, around 108 AD",
         text="Remember that it is not he who reviles you or strikes you, who insults you, but it is your opinion about these things as being insulting."),
    dict(day=5, name="MARCUS AURELIUS", cat="MORTALITY", era="the Roman frontier, around 175 AD, on campaign",
         text="Take pains therefore to know what it is that thy nature requireth, and let nothing else distract thee."),
    dict(day=6, name="FREDERICK DOUGLASS", cat="FREEDOM", era="the American North, 1845, after escaping slavery",
         text="Freedom now appeared, to disappear no more forever."),
    dict(day=7, name="BENJAMIN FRANKLIN", cat="CRAFT", era="Philadelphia and London, late 1700s",
         text="Human felicity is produced not so much by great pieces of good fortune that seldom happen, as by little advantages that occur every day."),
    dict(day=8, name="RALPH WALDO EMERSON", cat="THINKING", era="New England, mid-1800s",
         text="We are afraid of truth, afraid of fortune, afraid of death and afraid of each other."),
]

f_cat = font("Inter", 28, 500)
f_name = font("Inter", 30, 500)
f_era = font("CormorantGaramond-Italic", 40, 500)
f_word = font("CormorantGaramond", 56, 500)

FL, FL_ANCHOR = flame_img(170 / 44, 0.9)

for q in QUOTES:
    specs = [(540, 270, 330, 0.34, GOLD), (540, 310, 130, 0.14, (255, 226, 168)),
             (800, 200, 600, 0.18, (63, 90, 140)), (120, 1150, 520, 0.24, (40, 52, 80)),
             (540, 1340, 420, 0.18, GOLD)]
    arr = dust_static(glow_layer(W, H, specs), W, H, q["day"] * 17)
    img = Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))

    # corner brackets
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    c = GOLD_DIM + (190,)
    d.line([(64, 144), (64, 64), (144, 64)], fill=c, width=2)
    d.line([(W - 144, H - 64), (W - 64, H - 64), (W - 64, H - 144)], fill=c, width=2)
    img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")

    # flame
    img.paste(FL, (540 - FL_ANCHOR[0], 300 - FL_ANCHOR[1]), FL)
    d = ImageDraw.Draw(img)

    # category and rule
    draw_tracked(d, q["cat"], f_cat, W / 2, 378, GOLD_SOFT, 9)
    d.rectangle([W / 2 - 45, 404, W / 2 + 45, 405], fill=GOLD_DIM)

    # quote
    text = "\u201c" + q["text"] + "\u201d"
    s, f, lines = choose_size(text, 860, 540)
    lh = s * 1.18
    top, bottom = 450, 1010
    base = top + (bottom - top - lh * len(lines)) / 2 + s * 0.95
    for ln, cw in lines:
        x = (W - cw) / 2
        sp = f.getlength(" ")
        for w, wl in ln:
            d.text((x, base), w, font=f, fill=PARCH, anchor="ls")
            x += wl + sp
        base += lh

    # name and era
    draw_tracked(d, q["name"], f_name, W / 2, 1085, GOLD, 9)
    era = q["era"]
    fe = f_era
    if fe.getlength(era) > 860:
        fe = font("CormorantGaramond-Italic", 34, 500)
    d.text((W / 2 - fe.getlength(era) / 2, 1145), era, font=fe, fill=PARCH_DIM, anchor="ls")

    # wordmark
    wa, wb = f_word.getlength("Mentor"), f_word.getlength("Mind")
    x = (W - (wa + wb)) / 2
    d.text((x, H - 112), "Mentor", font=f_word, fill=PARCH, anchor="ls")
    d.text((x + wa, H - 112), "Mind", font=f_word, fill=GOLD, anchor="ls")

    img.save(f"{OUT}day{q['day']:02d}_quote.png")
    print("day", q["day"], "size", s, "lines", len(lines))
