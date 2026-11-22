"""Locate the toy in a recorded Toybox frame (plain wall background), so crops can be centred on the OBJECT.

mask = (far from the wall colour) AND (not skin), cleaned with morphology; the object is the largest component.
Returns the centre of the mask's bounding box in full-frame pixels. Always checked visually (`box_sheet`).
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw


def object_mask(img: Image.Image, bg_thresh: float = 38.0):
    a = np.asarray(img)
    small = cv2.resize(a, (480, 270))
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate([lab[:12].reshape(-1, 3), lab[:, :12].reshape(-1, 3), lab[:, -12:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    far = np.linalg.norm(lab - bg, axis=2) > bg_thresh
    ycc = cv2.cvtColor(small, cv2.COLOR_RGB2YCrCb)
    skin = (ycc[..., 1] >= 135) & (ycc[..., 1] <= 180) & (ycc[..., 2] >= 85) & (ycc[..., 2] <= 135)
    m = (far & ~skin).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = 1 + int(np.argmax(areas))
    bx, by, bw, bh = stats[big, :4]
    keep = []
    for k in range(1, n):          # the largest piece plus pieces of a reasonable size close to it (same object)
        x, y, w, h, area = stats[k]
        gap = max(bx - (x + w), x - (bx + bw), by - (y + h), y - (by + bh), 0)
        if k == big or (area >= 0.08 * areas.max() and gap <= 25):
            keep.append((x, y, x + w, y + h))
    x = min(b[0] for b in keep); y = min(b[1] for b in keep)
    w = max(b[2] for b in keep) - x; h = max(b[3] for b in keep) - y
    sx, sy = a.shape[1] / 480, a.shape[0] / 270
    return (x * sx, y * sy, (x + w) * sx, (y + h) * sy)


def center(img):
    b = object_mask(img)
    return None if b is None else ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def box_sheet(items, width=300):
    """items: [(label, PIL frame)] -> sheet with the detected box and centre drawn."""
    h = int(width * 9 / 16)
    sheet = Image.new("RGB", (width * len(items), h + 18), "white")
    d = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(items):
        b = object_mask(im)
        t = im.resize((width, h))
        dd = ImageDraw.Draw(t)
        if b:
            s = width / im.width
            dd.rectangle([b[0] * s, b[1] * s, b[2] * s, b[3] * s], outline=(255, 0, 0), width=2)
            cx, cy = (b[0] + b[2]) / 2 * s, (b[1] + b[3]) / 2 * s
            dd.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 0, 0))
        sheet.paste(t, (i * width, 18))
        d.text((i * width + 4, 3), label, fill="black")
    return sheet
