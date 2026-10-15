import numpy as np
import cv2
from PIL import Image, ImageFilter

SRC = "/mnt/project/logo-mark.png"
im = Image.open(SRC).convert("RGB")
P = np.asarray(im).astype(np.float32)

B = P[..., 2]
a = np.clip((255.0 - B) / 165.0, 0, 1)            # alpha from how far the blue channel dropped
a = np.where(a < 0.05, 0, a)

# remove isolated specks (JPEG smudges)
bw = (a > 0.12).astype(np.uint8)
n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
keep = np.zeros_like(bw)
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] >= 14:
        keep[lab == i] = 1
keep = cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)
a = a * keep

# unmix: P = a*F + (1-a)*255  ->  F
aa = np.maximum(a, 1e-3)[..., None]
F = (P - (1 - aa) * 255.0) / aa
F = np.clip(F, 0, 255)
gold = np.array([216, 166, 87], np.float32)
w = np.clip((a - 0.12) / 0.35, 0, 1)[..., None]    # trust the estimate only where alpha is solid
F = F * w + gold * (1 - w)
# keep colours inside the gold family so noise cannot turn them green or pink
F[..., 0] = np.clip(F[..., 0], 150, 255)
F[..., 1] = np.clip(F[..., 1], 100, 235)
F[..., 2] = np.clip(F[..., 2], 40, 190)

SC = 4
a_big = cv2.resize(a, None, fx=SC, fy=SC, interpolation=cv2.INTER_LANCZOS4)
F_big = cv2.resize(F * a[..., None], None, fx=SC, fy=SC, interpolation=cv2.INTER_LANCZOS4)  # premultiplied
a_big = np.clip(a_big, 0, 1)
F_big = np.clip(F_big, 0, 255)
F_un = F_big / np.maximum(a_big, 1e-3)[..., None]
F_un = np.clip(F_un, 0, 255)

# gentle sharpening of the alpha so thin strokes read clearly
blur = cv2.GaussianBlur(a_big, (0, 0), 1.6)
a_sh = np.clip(a_big + 0.9 * (a_big - blur), 0, 1)

out = np.zeros(a_big.shape + (4,), np.uint8)
out[..., :3] = F_un.astype(np.uint8)
out[..., 3] = (a_sh * 255).astype(np.uint8)

ys, xs = np.where(out[..., 3] > 6)
y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
pad = 12
crop = out[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]
Image.fromarray(crop, "RGBA").save("/home/claude/social/logo_cut.png")
print("logo_cut", crop.shape)

H = 560
orig = im.resize((int(im.width * H / im.height), H), Image.LANCZOS)
cut = Image.fromarray(crop, "RGBA")
cut2 = cut.resize((int(cut.width * H / cut.height), H), Image.LANCZOS)
sheet = Image.new("RGB", (orig.width + cut2.width + 70, H + 40), (14, 17, 26))
sheet.paste(orig, (20, 20))
sheet.paste(cut2, (orig.width + 50, 20), cut2)
sheet.save("/home/claude/social/logo_compare2.png")
print(sheet.size)
