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
SHIFT = {"right": (240, 0), "left": (-240, 0), "down": (0, 150), "up": (0, -150)}   # object displacement in frame px
OUT_W = 768
DIRS = ["left", "right", "up", "down"]


def crop(video_dir, video, frame, origin):
    """Recorded frame -> fixed-size crop whose top-left corner is ``origin`` (full-frame pixels)."""
    im = read_frame(video_dir / video, frame)
    x, y = origin
    return im.crop((x, y, x + CROP_W, y + CROP_H)).resize((OUT_W, round(OUT_W * CROP_H / CROP_W)), Image.LANCZOS)


def place(table, obj, items):
    """Crop origins for a group of frames of ONE object: items = [(frame, move)], the first being the reference
    (move None). Each crop is placed so that the object's position in it equals its position in the reference crop
    plus SHIFT[move] -- this cancels the incidental motion of the hand between recorded frames. The reference crop is
    as close to object-centred as the frame allows. Returns a list of origins, or None if the frame is too small."""
    c = table[obj]["centers"]
    c0 = c[str(items[0][0])]
    origins = []
    lims = (FRAME_W - CROP_W, FRAME_H - CROP_H)
    ref = []
    for ax in (0, 1):
        lo, hi = 0, lims[ax]
        for f, mv in items:
            off = c[str(f)][ax] - c0[ax] - (SHIFT[mv][ax] if mv else 0)     # origin_k = origin_ref + off
            lo, hi = max(lo, -off), min(hi, lims[ax] - off)
        if lo > hi:
            return None
        want = c0[ax] - (CROP_W, CROP_H)[ax] / 2
        ref.append(int(round(min(max(want, lo), hi))))
    for f, mv in items:
        origins.append([ref[ax] + c[str(f)][ax] - c0[ax] - (SHIFT[mv][ax] if mv else 0) for ax in (0, 1)])
    return origins


def placement(q, table):
    """Origins for every image of a question, or None if some group cannot be placed."""
    pl = {"demos": []}
    for dm in q["demos"]:
        o = place(table, dm["object"], [(dm["before"]["frame"], None), (dm["after"]["frame"], dm["after"].get("move"))])
        if o is None:
            return None
        pl["demos"].append(o)
    o = place(table, q["query"]["object"], [(q["query"]["frame"], None)] + [(c["frame"], c["move"]) for c in q["candidates"]])
    if o is None:
        return None
    pl["query"], pl["candidates"] = o[0], o[1:]
    return pl


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
            cand_q = {"task": "rotation", "rule": {"rotation_deg_ccw": r},
                       "demos": [{"object": ob, "before": {"frame": b, "orient": o}, "after": {"frame": a, "orient": (o + r) % 360}}
                                 for ob, b, a, o in demos],
                       "query": {"object": qo, "frame": qf, "orient": q},
                       "candidates": [{"kind": f"rotate {k}", "frame": f, "orient": (q + k) % 360, "move": None,
                                       "correct": k == ans_k} for k, f in cands]}
            if placement(cand_q, table) is None:
                continue
            qs.append(cand_q)
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
        for _ in range(200):
            o4 = rng.sample(objs, 4)
            demos = []
            for ob in o4[:3]:
                o = rng.choice([0, 90, 180, 270]); f = pick(rng, table, ob, o)
                demos.append({"object": ob, "before": {"frame": f, "orient": o}, "after": {"frame": f, "orient": o, "move": t}})
            q = rng.choice([0, 90, 180, 270]); qf = pick(rng, table, o4[3], q)
            cand_q = {"task": "translation", "rule": {"move": t}, "demos": demos,
                      "query": {"object": o4[3], "frame": qf, "orient": q},
                      "candidates": [{"kind": f"move {d}", "frame": qf, "orient": q, "move": d, "correct": d == t} for d in DIRS]}
            if placement(cand_q, table) is not None:
                qs.append(cand_q)
                break
        else:
            raise RuntimeError("could not build a translation question")
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
            cand_q = {"task": "rotation+translation", "rule": {"rotation_deg_ccw": r, "move": t}, "demos": demos,
                       "query": {"object": qo, "frame": qf, "orient": q},
                       "candidates": [
                           {"kind": "rotate + move (correct)", "frame": fr, "orient": (q + r) % 360, "move": t, "correct": True},
                           {"kind": "rotate only", "frame": fr, "orient": (q + r) % 360, "move": None, "correct": False},
                           {"kind": "move only", "frame": f0, "orient": q, "move": t, "correct": False},
                           {"kind": "rotate the other way + move", "frame": fw, "orient": (q - r) % 360, "move": t, "correct": False}]}
            if placement(cand_q, table) is None:
                continue
            qs.append(cand_q)
            break
        else:
            raise RuntimeError("could not build a combined question")
    return qs


def render(q, qid, video_dir, table, out: Path):
    d = out / qid
    (d / "images").mkdir(parents=True, exist_ok=True)
    vid = lambda ob: table[ob]["video"]
    pl = placement(q, table)
    q["crop_origins"] = {"demos": pl["demos"], "query": pl["query"]}
    files, rows = [], []
    for k, dm in enumerate(q["demos"], 1):
        b = crop(video_dir, vid(dm["object"]), dm["before"]["frame"], pl["demos"][k - 1][0])
        a = crop(video_dir, vid(dm["object"]), dm["after"]["frame"], pl["demos"][k - 1][1])
        for tag, im in (("before", b), ("after", a)):
            im.save(d / "images" / f"demo{k}_{tag}.jpg", quality=85)
            files.append({"label": f"Example {k} - {tag.capitalize()}", "file": f"images/demo{k}_{tag}.jpg"})
        rows.append([(f"Example {k}: Before", b), (f"Example {k}: After", a)])
    qim = crop(video_dir, vid(q["query"]["object"]), q["query"]["frame"], pl["query"])
    qim.save(d / "images" / "query.jpg", quality=85)
    files.append({"label": "Query", "file": "images/query.jpg"})
    rows.append([("Query", qim)])
    rng = random.Random(f"{qid}-order")
    order = list(range(4)); rng.shuffle(order)
    cand_row = []
    for letter, ci in zip("ABCD", order):
        c = q["candidates"][ci]
        im = crop(video_dir, vid(q["query"]["object"]), c["frame"], pl["candidates"][ci])
        im.save(d / "images" / f"candidate_{letter}.jpg", quality=85)
        files.append({"label": f"Candidate {letter}", "file": f"images/candidate_{letter}.jpg"})
        cand_row.append((f"Candidate {letter}", im))
        c["letter"] = letter
        c["crop_origin"] = pl["candidates"][ci]
    rows.append(cand_row)
    q["candidates"] = sorted(q["candidates"], key=lambda c: c["letter"])
    cm = composite(rows, OUT_W)
    cm.resize((cm.width // 2, cm.height // 2), Image.LANCZOS).save(d / "composite.jpg", quality=80)   # for viewing only
    (d / "inputs.json").write_text(json.dumps({"prompt": PROMPT, "images_in_order": files}, indent=2))
    return next(c["letter"] for c in q["candidates"] if c["correct"])


def main():
    global CROP_W, CROP_H, SHIFT
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--out", default=HERE / "batch1", type=Path)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--n", default=10, type=int)
    ap.add_argument("--tasks", default="R,T,RT", help="subset of R,T,RT to (re)build")
    ap.add_argument("--objects", default=None, help="comma-separated objects to use (default: all in keyframes.json)")
    ap.add_argument("--crop", default=None, help="crop window WxH in frame pixels, e.g. 1200x675")
    ap.add_argument("--shift", default=None, help="object shift dx,dy in frame pixels, e.g. 330,210")
    ap.add_argument("--merge", action="store_true", help="keep the other tasks already in <out>/answers.json")
    a = ap.parse_args()
    if a.crop:
        CROP_W, CROP_H = map(int, a.crop.split("x"))
    if a.shift:
        dx, dy = map(int, a.shift.split(","))
        SHIFT = {"right": (dx, 0), "left": (-dx, 0), "down": (0, dy), "up": (0, -dy)}
    table = json.loads((HERE / "keyframes.json").read_text())["objects"]
    if a.objects:
        table = {k: table[k] for k in a.objects.split(",")}
    tasks = a.tasks.split(",")
    rng = random.Random(a.seed)
    qs = []
    if "R" in tasks:
        qs += rotation_questions(rng, table, a.n)
    if "T" in tasks:
        qs += translation_questions(rng, table, a.n)
    if "RT" in tasks:
        qs += combined_questions(rng, table, a.n)
    ans_path = a.out / "answers.json"
    answers = json.loads(ans_path.read_text()) if (a.merge and ans_path.exists()) else {"prompt": PROMPT, "questions": {}}
    answers["note"] = ("orientations are approximate (hand-rotated, visually checked keyframes); translations are "
                       "crop-window shifts of recorded frames (camera-pan equivalent). Crops are placed from the object "
                       "centres in keyframes.json so that the object's displacement relative to the reference image "
                       "(demo 'before' / query) equals the intended shift (0 when there is no shift). Each question "
                       "records its own crop window, shift, objects and generation seed.")
    answers.pop("crop", None); answers.pop("seed", None)
    cfg = {"frame": [FRAME_W, FRAME_H], "window": [CROP_W, CROP_H], "object_shift_px": SHIFT, "output_width": OUT_W,
           "seed": a.seed, "objects": sorted(table)}
    for i, q in enumerate(qs):
        qid = {"rotation": "R", "translation": "T", "rotation+translation": "RT"}[q["task"]] + f"{i % a.n:02d}"
        if (a.out / qid).exists():
            import shutil
            shutil.rmtree(a.out / qid)
        q["answer"] = render(q, qid, a.videos, table, a.out)
        q["config"] = cfg
        answers["questions"][qid] = q
    answers["questions"] = dict(sorted(answers["questions"].items(), key=lambda kv: (kv[0].rstrip("0123456789"), kv[0])))
    ans_path.write_text(json.dumps(answers, indent=2))
    print(len(qs), "questions ->", a.out)


if __name__ == "__main__":
    main()
