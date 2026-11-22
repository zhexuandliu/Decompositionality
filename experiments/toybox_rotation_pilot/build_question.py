"""Build the single Toybox rotation question from recorded frames.

Usage:  python build_question.py --videos <dir with the .mp4 files> [--out question]

Every image is a recorded video frame, resized only (no rotation, flip, crop alignment or generation).
Writes (into --out):
  images/demo{1,2,3}_{before,after}.jpg, images/query.jpg, images/candidate_{A,B,C,D}.jpg   (model inputs)
  composite.jpg   readable sheet with neutral labels only
  prompt.txt      the exact model-facing prompt
  ../metadata.json   answer + provenance (kept OUTSIDE the question folder; never sent to the model)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from frames import read_frame, video_info

HERE = Path(__file__).resolve().parent
PROMPT = ("The three example pairs show the same transformation applied to different objects. Infer the shared "
          "transformation from the examples, then apply it to the query image. Which candidate shows the result? "
          "Answer with exactly one letter: A, B, C, or D.")


def _font(size):
    for f in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def frame(videos: Path, video: str, idx: int, width: int) -> Image.Image:
    im = read_frame(videos / Path(video).name, idx)          # full recorded frame, same framing for every object
    return im.resize((width, round(width * im.height / im.width)), Image.LANCZOS)


def composite(rows, width) -> Image.Image:
    """rows: list of lists of (label, image)."""
    f = _font(26)
    h = rows[0][0][1].height
    ncol = max(len(r) for r in rows)
    pad, lab = 16, 38
    W = ncol * width + (ncol + 1) * pad
    H = len(rows) * (h + lab + pad) + pad
    out = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(out)
    for r, row in enumerate(rows):
        y = pad + r * (h + lab + pad)
        for c, (label, im) in enumerate(row):
            x = pad + c * (width + pad)
            d.text((x, y + 4), label, fill="black", font=f)
            out.paste(im, (x, y + lab))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--out", default=HERE / "question", type=Path)
    a = ap.parse_args()
    sel = json.loads((HERE / "selection.json").read_text())
    W = sel["image_width"]
    img_dir = a.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    files, rows = [], []
    for k, dm in enumerate(sel["demos"], 1):
        b, af = frame(a.videos, dm["video"], dm["before_frame"], W), frame(a.videos, dm["video"], dm["after_frame"], W)
        for tag, im in (("before", b), ("after", af)):
            p = img_dir / f"demo{k}_{tag}.jpg"
            im.save(p, quality=90)
            files.append({"label": f"Example {k} - {tag.capitalize()}", "file": f"images/{p.name}"})
        rows.append([(f"Example {k}: Before", b), (f"Example {k}: After", af)])

    q = sel["query"]
    qim = frame(a.videos, q["video"], q["frame"], W)
    qim.save(img_dir / "query.jpg", quality=90)
    files.append({"label": "Query", "file": "images/query.jpg"})
    rows.append([("Query", qim)])

    order = list(range(len(sel["candidates"])))
    random.Random(sel["shuffle_seed"]).shuffle(order)
    letters = "ABCD"
    cand_rows, final = [], []
    for letter, ci in zip(letters, order):
        c = sel["candidates"][ci]
        im = frame(a.videos, q["video"], c["frame"], W)
        im.save(img_dir / f"candidate_{letter}.jpg", quality=90)
        files.append({"label": f"Candidate {letter}", "file": f"images/candidate_{letter}.jpg"})
        cand_rows.append((f"Candidate {letter}", im))
        final.append({"letter": letter, **c})
    rows.append(cand_rows)

    composite(rows, W).save(a.out / "composite.jpg", quality=90)
    (a.out / "prompt.txt").write_text(PROMPT + "\n")
    (a.out / "inputs.json").write_text(json.dumps({"prompt": PROMPT, "images_in_order": files}, indent=2))

    answer = next(x["letter"] for x in final if x["intended_rotation_deg_ccw"] == sel["correct_rotation_deg_ccw"])
    meta = {
        "dataset": "Toybox (Wang et al. 2018), https://aivaslab.github.io/toybox/ ; videos from the Zenodo archives "
                   "(vehicles: record 1290757, animals: record 1289936), streamed and only the needed files kept",
        "intended_transformation": sel["transformation"],
        "labeling": "approximate, visually checked pilot question; NOT calibrated geometric ground truth. " + sel["note"],
        "fps": {d["object"]: round(video_info(a.videos / Path(d["video"]).name)["fps"], 3)
                for d in sel["demos"] + [q]},
        "demos": sel["demos"],
        "query": q,
        "candidate_shuffle_seed": sel["shuffle_seed"],
        "candidates_in_presented_order": final,
        "intended_answer": answer,
        "uncertainty": [
            "Rotations are performed by hand; angles are approximate (roughly +-15 degrees).",
            "Toybox uses a head-mounted camera; the camera may move slightly between frames.",
            "The truck_28 'before' frame is tilted by roughly 10-15 degrees from vertical.",
            "Hands partly occlude the objects; hand positions differ between frames.",
            "Some rotation may be slightly out of plane.",
        ],
    }
    for d in meta["demos"] + [meta["query"]]:
        f_ = meta["fps"][d["object"]]
        for key in ("before_frame", "after_frame", "frame"):
            if key in d:
                d[key.replace("frame", "time_s")] = round(d[key] / f_, 2)
    for c in meta["candidates_in_presented_order"]:
        c["time_s"] = round(c["frame"] / meta["fps"][q["object"]], 2)
    (HERE / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("question ->", a.out, "| intended answer (metadata only):", answer)


if __name__ == "__main__":
    main()
