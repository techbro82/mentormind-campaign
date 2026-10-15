import sys, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, "/home/claude/social")
import render_feeling as rf
from social_lib import font, ss, GOLD

F32 = np.float32
S0 = 1.9                                   # scene resolution (so a page can be filled close to 1:1 on screen)
PWu, PHu, BWu, BHu = 500, 1078, 522, 1122  # page and board size, in scene units
PW, PH, BW, BH = int(PWu * S0), int(PHu * S0), int(BWu * S0), int(BHu * S0)
YOFF = 260                                  # spare rows above and below, so a lifted page is never cut off
CWs, CHs = int(1120 * S0) + 300, int(1180 * S0) + 2 * YOFF
SPX = CWs // 2                             # x of the spine, in canvas pixels
PAGE_Y0, BOARD_Y0 = int(51 * S0) + YOFF, int(29 * S0) + YOFF
PR = "/mnt/project/"

CREAM = np.array([0.925, 0.890, 0.815], F32)
NAVY = np.array([0.075, 0.100, 0.175], F32)
GOLDF = np.array(GOLD, F32) / 255.0


def noise(h, w, seed, sigma_fine=0.010, sigma_low=0.020, low=60):
    rs = np.random.RandomState(seed)
    fine = rs.randn(h, w).astype(F32) * sigma_fine
    lo = cv2.resize(rs.randn(h // low + 2, w // low + 2).astype(F32), (w, h), interpolation=cv2.INTER_CUBIC)
    lo = cv2.GaussianBlur(lo, (0, 0), low / 2.5) * sigma_low * 2.2
    return (fine + lo)[..., None]


def gutter(w, side):
    """Brightness across a page: darker toward the spine, a touch darker at the outer edge."""
    x = np.arange(w, dtype=F32)
    d = x if side == "R" else (w - 1 - x)                  # distance from the spine
    g = 0.46 + 0.54 * np.clip(d / (120 * S0), 0, 1) ** 0.8
    edge = 0.90 + 0.10 * np.clip((w - 1 - d) / (14 * S0), 0, 1)
    return (g * edge)[None, :, None]


def plate_page(screen_file, numeral, side, seed):
    page = np.ones((PH, PW, 3), F32) * CREAM + noise(PH, PW, seed)
    im = Image.open(PR + screen_file).convert("RGB")
    pw, ph = int(440 * S0), int(948 * S0)
    scr = np.asarray(im.resize((pw, ph), Image.LANCZOS), F32) / 255.0
    x0, y0 = int(30 * S0), int(92 * S0)
    # a soft shadow under the plate, then the plate with a thin gold edge
    sh = np.zeros((PH, PW), F32)
    sh[y0 + int(7 * S0):y0 + ph + int(7 * S0), x0 + int(5 * S0):x0 + pw + int(5 * S0)] = 1
    sh = cv2.GaussianBlur(sh, (0, 0), 7 * S0)
    page = page * (1 - 0.30 * sh[..., None])
    b = max(int(3 * S0), 2)
    page[y0 - b:y0 + ph + b, x0 - b:x0 + pw + b] = GOLDF * 0.80
    page[y0:y0 + ph, x0:x0 + pw] = scr
    if numeral:
        pil = Image.fromarray(np.clip(page * 255, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(pil)
        f = font("Inter", int(62 * S0), 300)
        d.text((x0 + 2, int(20 * S0)), numeral, font=f, fill=(150, 108, 44), anchor="la")
        page = np.asarray(pil, F32) / 255.0
    return page * gutter(PW, side)


def endpaper():
    img = np.ones((BH, BW, 3), F32) * NAVY + noise(BH, BW, 3, 0.006, 0.006, 90)
    g = int(26 * S0)
    pil = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    d.rectangle([g, g, BW - g, BH - g], outline=(168, 132, 78), width=int(3 * S0))
    d.rectangle([g + int(9 * S0), g + int(9 * S0), BW - g - int(9 * S0), BH - g - int(9 * S0)], outline=(110, 86, 46), width=max(int(1.5 * S0), 1))
    return np.asarray(pil, F32) / 255.0


def cover():
    img = np.ones((BH, BW, 3), F32) * np.array([0.080, 0.105, 0.185], F32) + noise(BH, BW, 5, 0.012, 0.009, 70)
    g = int(26 * S0)
    pil = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    d.rectangle([g, g, BW - g, BH - g], outline=(188, 148, 84), width=int(4 * S0))
    d.rectangle([g + int(11 * S0), g + int(11 * S0), BW - g - int(11 * S0), BH - g - int(11 * S0)], outline=(120, 92, 48), width=max(int(2 * S0), 1))
    img = np.asarray(pil, F32) / 255.0
    # the MentorMind mark, embossed in gold
    LS, lox, loy = rf.lock_scaled(int(380 * S0))
    h, w = LS.shape[:2]
    cx, cy = BW // 2, int(BH * 0.42)
    x0, y0 = int(cx - w / 2 - lox), int(cy - h / 2 - loy)
    m = LS.max(axis=2)
    shd = cv2.GaussianBlur(m, (0, 0), 5 * S0)
    reg = img[y0:y0 + h, x0:x0 + w]
    reg2 = np.roll(np.roll(shd, int(4 * S0), axis=0), int(3 * S0), axis=1)
    reg[:] = reg * (1 - 0.55 * reg2[..., None]) + LS * 1.25
    return np.clip(img, 0, 1)


def build_assets():
    A = {}
    A["end"] = endpaper()
    A["cover"] = cover()
    A["P1"] = plate_page("screen-04-onboarding-struggle.png", "1", "R", 11)
    A["P2b"] = plate_page("screen-05-onboarding-aspiration.png", "2", "L", 12)
    A["P3"] = plate_page("screen-07-mentor-browse.png", "3", "R", 13)
    A["P4b"] = plate_page("screen-11-empty-chat.png", "", "L", 14)
    return A


# ---------------------------------------------------------------- static book
_VOID = None


def void_canvas():
    global _VOID
    if _VOID is None:
        yy, xx = np.mgrid[0:CHs:8, 0:CWs:8].astype(F32)
        v = 1 - 0.55 * (((xx - CWs / 2) / (CWs / 2)) ** 2 + ((yy - CHs / 2) / (CHs / 2)) ** 2)
        v = cv2.resize(np.clip(v, 0.25, 1), (CWs, CHs), interpolation=cv2.INTER_LINEAR)
        _VOID = (np.array([0.030, 0.036, 0.056], F32)[None, None, :] * v[..., None]).astype(F32)
    return _VOID.copy()


def paste(canvas, img, x, y):
    h, w = img.shape[:2]
    canvas[y:y + h, x:x + w] = img


def stack_edges(canvas, side):
    """A few thin lines at the outer edge of a page, to suggest the thickness of the paper."""
    n = 9
    for i in range(n):
        shade = 0.86 - 0.05 * i
        off = int((2 + i * 1.6) * S0 * 0.5)
        if side == "R":
            x = SPX + PW + off
        else:
            x = SPX - PW - off - 2
        canvas[PAGE_Y0 + 4:PAGE_Y0 + PH - 4, x:x + 2] = np.array([shade, shade * 0.97, shade * 0.9], F32)


def compose(left=None, right=None, left_board=False, right_board=False, A=None):
    c = void_canvas()
    # shadow of the book on the table
    sh = np.zeros((CHs, CWs), F32)
    if left_board or left is not None:
        sh[BOARD_Y0 + int(14 * S0):BOARD_Y0 + BH + int(14 * S0), SPX - BW:SPX] = 1
    if right_board or right is not None:
        sh[BOARD_Y0 + int(14 * S0):BOARD_Y0 + BH + int(14 * S0), SPX:SPX + BW] = 1
    sh = cv2.GaussianBlur(sh, (0, 0), 24 * S0 / 1.9)
    c *= (1 - 0.6 * sh[..., None])
    if left_board or left is not None:
        paste(c, A["end"], SPX - BW, BOARD_Y0)
        if left is not None:
            stack_edges(c, "L")
            paste(c, left, SPX - PW, PAGE_Y0)
    if right_board or right is not None:
        paste(c, A["end"], SPX, BOARD_Y0)
        if right is not None:
            stack_edges(c, "R")
            paste(c, right, SPX, PAGE_Y0)
    # the crease down the middle
    gx = np.arange(-int(26 * S0), int(26 * S0))
    crease = 1 - 0.35 * np.exp(-(gx / (9.0 * S0)) ** 2)
    c[BOARD_Y0:BOARD_Y0 + BH, SPX - int(26 * S0):SPX + int(26 * S0)] *= crease[None, :, None]
    return c


# ---------------------------------------------------------------- the page turn
def apply_turn(canvas, front, back, p, y0, stiff=1.0, Dz=4200.0):
    """A sheet hinged at the spine, rotating from the right page to the left page.
    It bends (the free edge leads), is seen in perspective from above, shaded by its angle, and casts a shadow."""
    h, w = front.shape[:2]
    e = p * p * (3 - 2 * p)
    kap = max(2.0 * stiff * (1 - p) ** 1.2, 0.02)
    N = 4 * w
    u = np.linspace(0, w - 1, N)
    sn = np.maximum(u / (w - 1), 1e-6)
    phi = math.pi * e * np.power(sn, kap)                 # the tangent angle along the sheet
    du = u[1] - u[0]
    x = np.concatenate([[0.0], np.cumsum(np.cos(phi[:-1]) * du)])
    z = np.concatenate([[0.0], np.cumsum(np.sin(phi[:-1]) * du)])
    persp = 1 + z / Dz
    xi = np.round(x * persp).astype(np.int64) + w
    n = 2 * w
    ok = (xi >= 0) & (xi < n)
    order = np.lexsort((z, xi))                           # for each screen column, keep the highest part of the sheet
    xs_, = (xi[order],)
    last = np.r_[xs_[1:] != xs_[:-1], True] & ok[order]
    sel = order[last]
    cols = xi[sel]
    cosp, sinp = np.cos(phi[sel]), np.sin(phi[sel])
    is_front = cosp > 0
    M = 250
    Hout = h + 2 * M
    ncol = len(sel)
    stack = np.concatenate([front, back], axis=1)         # one source image: front, then back
    map_x = np.where(is_front, u[sel], w + (w - 1 - u[sel])).astype(F32)
    sy = persp[sel].astype(F32)
    yy = np.arange(Hout, dtype=F32)[:, None]
    map_y = ((yy - M) - h / 2) / sy[None, :] + h / 2
    mx = np.broadcast_to(map_x[None, :], (Hout, ncol)).astype(F32).copy()
    patch = cv2.remap(stack, mx, map_y.astype(F32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    inside = ((map_y >= 0) & (map_y <= h - 1)).astype(F32)[..., None]
    # light from the top left: lit by the angle of the surface
    Lx, Lz = -0.35, 0.94
    nf = np.where(is_front, -sinp * Lx + cosp * Lz, sinp * Lx - cosp * Lz)
    bright = 0.50 + 0.55 * np.clip(nf, 0, 1)
    bright = bright * (1 - 0.18 * np.clip(z[sel] / 900.0, 0, 1))
    patch = patch * bright[None, :, None].astype(F32)
    out = np.zeros((Hout, n, 3), F32)
    alpha = np.zeros((Hout, n), F32)
    out[:, cols] = patch
    alpha[:, cols] = inside[..., 0]
    col_a = np.zeros(n, F32)
    col_a[cols] = 1
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8)
    # shadow on the page underneath: falls to the right and down, wider while the sheet is high
    sm = cv2.resize(np.roll(alpha, int(70 * (0.4 + math.sin(math.pi * p))), axis=1), (n // 4, Hout // 4), interpolation=cv2.INTER_AREA)
    sh = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 10), (n, Hout), interpolation=cv2.INTER_LINEAR)
    amp = 0.55 * math.sin(math.pi * min(max(p, 0), 1)) ** 0.7
    ry0 = y0 - M
    cx0 = SPX - w
    reg = canvas[ry0:ry0 + Hout, cx0:cx0 + n]
    reg *= (1 - amp * sh * (1 - alpha))[..., None]
    a3 = alpha[..., None]
    reg[:] = reg * (1 - a3) + out * a3
    return canvas
