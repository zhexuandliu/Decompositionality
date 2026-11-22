"""Generate the batch: 10 rotation-only, 10 translation-only and 10 rotation+translation questions.

Usage:  python make_questions.py --videos <dir> [--out batch1] [--seed 0]

Each question = 3 before/after demonstrations (3 different objects) + 1 query (a 4th object) + 4 candidates,
the same format and prompt as the single pilot question.

* Rotation: recorded frames only (Toybox `ry` videos; orientations from the visually checked keyframes.json).
* Translation: made by moving the crop window over the recorded frame (equivalent to the camera panning).
  Every image in every question is a crop of the same size, so the framing is identical across tasks.
* Answers and provenance go to <out>/answers.json, which is never sent to the model.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

from build_question import PROMPT, composite
from frames import read_frame

HERE = Path(__file__).resolve().parent
FRAME_W, FRAME_H = 1920, 1080
CROP_W, CROP_H = 1344, 756                     # 70% window
CX, CY = (FRAME_W - CROP_W) // 2, (FRAME_H - CROP_H) // 2
SHIFT = {"right": (240, 0), "left": (-240, 0), "down": (0, 150), "up": (0, -150)}   # object displacement in frame px
OUT_W = 768
DIRS = ["left", "right", "up", "down"]


def crop(video_dir, video, frame, move=None):
    """Recorded frame -> fixed-size crop. ``move`` = direction the OBJECT moves (the window moves the other way)."""
    im = read_frame(video_dir / video, frame)
    dx, dy = SHIFT[move] if move else (0, 0)
    x, y = CX - dx, CY - dy
    return im.crop((x, y, x + CROP_W, y + CROP_H)).resize((OUT_W, round(OUT_W * CROP_H / CROP_W)), Image.LANCZOS)


def pick(rng, table, obj, orient, avoid=()):
    fs = [f for f in table[obj]["frames"].get(str(orient % 360), []) if (obj, f) not in avoid]
    return rng.choice(fs) if fs else None


def rotation_questions(rng, table, n):
    objs = list(table)
    rules = ([90, -90, 180] * n)[:n]
    rng.shuffle(rules)
    qs = []
    for r in rules:
        for _ in range(200):
            o4 = rng.sample(objs, 4)
            starts = rng.sample([0, 90, 180, 270], 3)
            demos, used = [], set()
            for ob, o in zip(o4[:3], starts):
                b = pick(rng, table, ob, o, used); used.add((ob, b))
                a = pick(rng, table, ob, o + r, used); used.add((ob, a))
                demos.append((ob, b, a, o))
            q = rng.choice([0, 90, 180, 270]); qo = o4[3]
            qf = pick(rng, table, qo, q); used.add((qo, qf))
            cands = [(k, pick(rng, table, qo, q + k, used)) for k in (0, 90, 180, 270)]
            if None in [x for d in demos for x in d[1:3]] + [qf] + [c[1] for c in cands]:
                continue
            ans_k = r % 360
            qs.append({"task": "rotation", "rule": {"rotation_deg_ccw": r},
                       "demos": [{"object": ob, "before": {"frame": b, "orient": o}, "after": {"frame": a, "orient": (o + r) % 360}}
                                 for ob, b, a, o in demos],
                       "query": {"object": qo, "frame": qf, "orient": q},
                       "candidates": [{"kind": f"rotate {k}", "frame": f, "orient": (q + k) % 360, "move": None,
                                       "correct": k == ans_k} for k, f in cands]})
            break
        else:
            raise RuntimeError("could not build a rotation question")
    return qs


def translation_questions(rng, table, n):
    objs = list(table)
    rules = (DIRS * n)[:n]
    rng.shuffle(rules)
    qs = []
    for t in rules:
        o4 = rng.sample(objs, 4)
        demos = []
        for ob in o4[:3]:
            o = rng.choice([0, 90, 180, 270]); f = pick(rng, table, ob, o)
            demos.append({"object": ob, "before": {"frame": f, "orient": o}, "after": {"frame": f, "orient": o, "move": t}})
        q = rng.choice([0, 90, 180, 270]); qf = pick(rng, table, o4[3], q)
        qs.append({"task": "translation", "rule": {"move": t}, "demos": demos,
                   "query": {"object": o4[3], "frame": qf, "orient": q},
                   "candidates": [{"kind": f"move {d}", "frame": qf, "orient": q, "move": d, "correct": d == t} for d in DIRS]})
    return qs


def combined_questions(rng, table, n):
    objs = list(table)
    rules = [(r, t) for r in (90, -90) for t in DIRS]
    rules = (rules * 2)[:n]
    rng.shuffle(rules)
    qs = []
    for r, t in rules:
        for _ in range(200):
            o4 = rng.sample(objs, 4)
            starts = rng.sample([0, 90, 180, 270], 3)
            demos, used = [], set()
            for ob, o in zip(o4[:3], starts):
                b = pick(rng, table, ob, o, used); used.add((ob, b))
                a = pick(rng, table, ob, o + r, used); used.add((ob, a))
                demos.append({"object": ob, "before": {"frame": b, "orient": o},
                              "after": {"frame": a, "orient": (o + r) % 360, "move": t}})
            q = rng.choice([0, 90, 180, 270]); qo = o4[3]
            qf = pick(rng, table, qo, q); used.add((qo, qf))
            fr = pick(rng, table, qo, q + r, used)                 # correct rotation
            f0 = pick(rng, table, qo, q, used) or qf               # unrotated (another recorded frame if available)
            fw = pick(rng, table, qo, q - r, used)                 # rotated the wrong way
            if None in [d["before"]["frame"] for d in demos] + [d["after"]["frame"] for d in demos] + [qf, fr, fw]:
                continue
            qs.append({"task": "rotation+translation", "rule": {"rotation_deg_ccw": r, "move": t}, "demos": demos,
                       "query": {"object": qo, "frame": qf, "orient": q},
                       "candidates": [
                           {"kind": "rotate + move (correct)", "frame": fr, "orient": (q + r) % 360, "move": t, "correct": True},
                           {"kind": "rotate only", "frame": fr, "orient": (q + r) % 360, "move": None, "correct": False},
                           {"kind": "move only", "frame": f0, "orient": q, "move": t, "correct": False},
                           {"kind": "rotate the other way + move", "frame": fw, "orient": (q - r) % 360, "move": t, "correct": False}]})
            break
        else:
            raise RuntimeError("could not build a combined question")
    return qs


def render(q, qid, video_dir, table, out: Path):
    d = out / qid
    (d / "images").mkdir(parents=True, exist_ok=True)
    vid = lambda ob: table[ob]["video"]
    files, rows = [], []
    for k, dm in enumerate(q["demos"], 1):
        b = crop(video_dir, vid(dm["object"]), dm["before"]["frame"], dm["before"].get("move"))
        a = crop(video_dir, vid(dm["object"]), dm["after"]["frame"], dm["after"].get("move"))
        for tag, im in (("before", b), ("after", a)):
            im.save(d / "images" / f"demo{k}_{tag}.jpg", quality=85)
            files.append({"label": f"Example {k} - {tag.capitalize()}", "file": f"images/demo{k}_{tag}.jpg"})
        rows.append([(f"Example {k}: Before", b), (f"Example {k}: After", a)])
    qim = crop(video_dir, vid(q["query"]["object"]), q["query"]["frame"])
    qim.save(d / "images" / "query.jpg", quality=85)
    files.append({"label": "Query", "file": "images/query.jpg"})
    rows.append([("Query", qim)])
    rng = random.Random(f"{qid}-order")
    order = list(range(4)); rng.shuffle(order)
    cand_row = []
    for letter, ci in zip("ABCD", order):
        c = q["candidates"][ci]
        im = crop(video_dir, vid(q["query"]["object"]), c["frame"], c["move"])
        im.save(d / "images" / f"candidate_{letter}.jpg", quality=85)
        files.append({"label": f"Candidate {letter}", "file": f"images/candidate_{letter}.jpg"})
        cand_row.append((f"Candidate {letter}", im))
        c["letter"] = letter
    rows.append(cand_row)
    q["candidates"] = sorted(q["candidates"], key=lambda c: c["letter"])
    cm = composite(rows, OUT_W)
    cm.resize((cm.width // 2, cm.height // 2), Image.LANCZOS).save(d / "composite.jpg", quality=80)   # for viewing only
    (d / "inputs.json").write_text(json.dumps({"prompt": PROMPT, "images_in_order": files}, indent=2))
    return next(c["letter"] for c in q["candidates"] if c["correct"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--out", default=HERE / "batch1", type=Path)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--n", default=10, type=int)
    a = ap.parse_args()
    table = json.loads((HERE / "keyframes.json").read_text())["objects"]
    rng = random.Random(a.seed)
    qs = (rotation_questions(rng, table, a.n) + translation_questions(rng, table, a.n)
          + combined_questions(rng, table, a.n))
    answers = {"seed": a.seed, "prompt": PROMPT, "crop": {"frame": [FRAME_W, FRAME_H], "window": [CROP_W, CROP_H],
               "object_shift_px": SHIFT, "output_width": OUT_W},
               "note": "orientations are approximate (hand-rotated, visually checked keyframes); translations are "
                       "crop-window shifts of recorded frames (camera-pan equivalent)", "questions": {}}
    for i, q in enumerate(qs):
        qid = {"rotation": "R", "translation": "T", "rotation+translation": "RT"}[q["task"]] + f"{i % a.n:02d}"
        q["answer"] = render(q, qid, a.videos, table, a.out)
        answers["questions"][qid] = q
    (a.out / "answers.json").write_text(json.dumps(answers, indent=2))
    print(len(qs), "questions ->", a.out)


if __name__ == "__main__":
    main()
