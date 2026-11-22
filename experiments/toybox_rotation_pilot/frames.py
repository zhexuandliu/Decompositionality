"""Frame utilities for the Toybox rotation pilot: read frames by index, and build timestamped contact sheets."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    info = {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def read_frame(path, index: int) -> Image.Image:
    """The recorded frame at ``index`` (0-based), as an RGB image. No geometric change is applied."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read frame {index} of {path}")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def contact_sheet(path, step_s: float = 0.5, cols: int = 8, thumb_w: int = 240, start_s: float = 0.0,
                  end_s: float | None = None) -> Image.Image:
    """Thumbnails every ``step_s`` seconds, each labelled with its timestamp and frame index (inspection only)."""
    info = video_info(path)
    fps = info["fps"]
    end_s = end_s if end_s is not None else info["frames"] / fps
    idx = [int(round(t * fps)) for t in np.arange(start_s, end_s, step_s) if int(round(t * fps)) < info["frames"] - 2]
    th = int(thumb_w * info["height"] / info["width"])
    rows = (len(idx) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (th + 16)), "white")
    d = ImageDraw.Draw(sheet)
    for k, i in enumerate(idx):
        im = read_frame(path, i).resize((thumb_w, th))
        x, y = (k % cols) * thumb_w, (k // cols) * (th + 16)
        sheet.paste(im, (x, y + 16))
        d.text((x + 3, y + 2), f"t={i / fps:5.2f}s f={i}", fill="black")
    return sheet
