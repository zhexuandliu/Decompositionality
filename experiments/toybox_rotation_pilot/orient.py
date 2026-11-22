"""Track the in-plane orientation of the object through a Toybox `ry` video and propose keyframes at roughly
0/90/180/270 degrees (counter-clockwise, relative to frame 0).

Method: SIFT + RANSAC similarity between frames `step` apart, accumulated over the video. Estimates with too few
inliers are skipped (the angle is then carried over). Proposed keyframes are always CHECKED VISUALLY afterwards
(contact sheet from `keyframe_sheet`); the tracker only narrows the search.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from angle import rotation_deg


def track(path, step: int = 3, min_inliers: int = 12):
    """[(frame, cumulative_deg_ccw, inliers)] for frames 0, step, 2*step, ..."""
    cap = cv2.VideoCapture(str(path))
    out, prev, cum, f = [], None, 0.0, 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if f % step == 0:
            cur = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            k = 0
            if prev is not None:
                a, k = rotation_deg(prev, cur)
                if a is not None and k >= min_inliers and abs(a) < 45:
                    cum += a
            out.append((f, cum, k))
            prev = cur
        f += 1
    cap.release()
    return out


def keyframes(trace, targets=(0, 90, 180, 270), tol: float = 10.0, min_gap: int = 45, skip_start: int = 30):
    """For each target orientation (mod 360): candidate frames whose tracked angle is within ``tol`` degrees,
    one per visit (frames closer than ``min_gap`` are the same visit). Target 0 ignores the first ``skip_start``
    frames, so the 0-degree keyframes include later, re-recorded visits."""
    res = {}
    for t in targets:
        hits = []
        for f, a, _ in trace:
            d = (a - t + 180) % 360 - 180
            if abs(d) <= tol and not (t == 0 and f < skip_start):
                hits.append((f, d))
        visits, cur = [], []
        for f, d in hits:
            if cur and f - cur[-1][0] > min_gap:
                visits.append(cur)
                cur = []
            cur.append((f, d))
        if cur:
            visits.append(cur)
        res[t] = [min(v, key=lambda x: abs(x[1]))[0] for v in visits]
    return res


def keyframe_sheet(read_frame, path, kf: dict, width: int = 300) -> Image.Image:
    cells = [(t, f) for t in sorted(kf) for f in kf[t]]
    h = int(width * 9 / 16)
    sheet = Image.new("RGB", (width * max(1, len(cells)), h + 18), "white")
    d = ImageDraw.Draw(sheet)
    for i, (t, f) in enumerate(cells):
        sheet.paste(read_frame(path, f).resize((width, h)), (i * width, 18))
        d.text((i * width + 4, 3), f"{t} deg  f={f}", fill="black")
    return sheet
