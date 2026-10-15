import numpy as np
import cv2
from PIL import Image

SRC = "/mnt/project/logo-mark.png"
OUT = "/home/claude/social/logo_hires.png"
SCALE = 8

im = Image.open(SRC).convert("RGB")
a = np.asarray(im).astype(np.float32)

# Gold is darkest in the blue channel, white background is 255 everywhere.
d = (255.0 - a[..., 2]) / 255.0           # 0 = background, higher = more gold
d = np.clip((d - 0.10) / (0.75 - 0.10), 0, 1)   # drop JPEG noise, stretch contrast

# Remove isolated specks (the faint grey smudges near the bottom) before upscaling.
bw = (d > 0.18).astype(np.uint8)
n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
keep = np.zeros_like(bw)
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] >= 14:
        keep[lab == i] = 1
keep = cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)
d = d * keep

big = cv2.resize(d, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
big = cv2.GaussianBlur(big, (0, 0), 2.4)

# crisp but anti-aliased edge
alpha = np.clip((big - 0.20) / (0.42 - 0.20), 0, 1)
alpha = alpha * alpha * (3 - 2 * alpha)
alpha = cv2.GaussianBlur(alpha, (0, 0), 0.9)

# colour: deeper gold where the original was denser, lighter gold on thin or pale parts
t = np.clip(big / 0.85, 0, 1)
stops = np.array([[190, 140, 62], [216, 166, 87], [240, 196, 121], [255, 232, 176]], np.float32)
# t high = dense gold = rich gold; low = pale. map low->light, high->deep
pos = 1 - t
idx = pos * (len(stops) - 1)
i0 = np.clip(np.floor(idx).astype(int), 0, len(stops) - 2)
f = (idx - i0)[..., None]
rgb = stops[i0] * (1 - f) + stops[i0 + 1] * f

h, w = alpha.shape
out = np.zeros((h, w, 4), np.uint8)
out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
out[..., 3] = (alpha * 255).astype(np.uint8)

# crop to the artwork with a margin
ys, xs = np.where(out[..., 3] > 8)
y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
pad = 24
y0, x0 = max(y0 - pad, 0), max(x0 - pad, 0)
y1, x1 = min(y1 + pad, h - 1), min(x1 + pad, w - 1)
crop = out[y0:y1 + 1, x0:x1 + 1]
Image.fromarray(crop, "RGBA").save(OUT)
print("saved", OUT, crop.shape)

# review sheet: original (nearest-neighbour x6 on white) vs rebuilt on ink
orig = im.resize((im.width * 6, im.height * 6), Image.LANCZOS)
rb = Image.fromarray(crop, "RGBA")
H = 640
rb2 = rb.resize((int(rb.width * H / rb.height), H), Image.LANCZOS)
sheet = Image.new("RGB", (orig.width + rb2.width + 60, H + 40), (14, 17, 26))
sheet.paste(orig.resize((int(orig.width * H / orig.height), H)), (20, 20))
sheet.paste(rb2, (orig.width + 40, 20), rb2)
sheet.save("/home/claude/social/logo_compare.png")
print(sheet.size)
