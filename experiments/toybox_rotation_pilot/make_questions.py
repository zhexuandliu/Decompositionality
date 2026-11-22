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

from PIL import Image, ImageOps

from build_question import PROMPT, composite
from frames import read_frame

HERE = Path(__file__).resolve().parent
FRAME_W, FRAME_H = 1920, 1080
CROP_W, CROP_H = 1344, 756                     # 70% window
SHIFT = {"right": (240, 0), "left": (-240, 0), "down": (0, 150), "up": (0, -150)}   # object displacement in frame px
OUT_W = 768
DIRS = ["left", "right", "up", "down"]


def crop(video_dir, video, frame, origin, mirror=None):
    """Recorded frame -> fixed-size crop whose top-left corner is ``origin`` (full-frame pixels). ``mirror`` = "h"
    (left-right) or "v" (top-bottom) flips the CROP digitally (mirror images are not physically recordable)."""
    im = read_frame(video_dir / video, frame)
    x, y = origin
    im = im.crop((x, y, x + CROP_W, y + CROP_H))
    if mirror == "h":
        im = ImageOps.mirror(im)
    elif mirror == "v":
        im = ImageOps.flip(im)
    return im.resize((OUT_W, round(OUT_W * CROP_H / CROP_W)), Image.LANCZOS)


def place(table, obj, items):
    """Crop origins for a group of frames of ONE object: items = [(frame, move[, mirror])], the first being the
    reference (no move, no mirror). Each crop is placed so that the object's position in the FINAL image (after an
    optional flip of the crop) equals its position in the reference image plus SHIFT[move] -- this cancels the
    incidental motion of the hand between recorded frames, and makes a mirror act about the object's own centre.
    The reference position is as close to the image centre as the frame allows. Returns origins or None."""
    c = table[obj]["centers"]
    items = [tuple(it) + (None,) * (3 - len(it)) for it in items]
    lims = (FRAME_W - CROP_W, FRAME_H - CROP_H)
    dims = (CROP_W, CROP_H)
    u = []
    for ax in (0, 1):
        L, lim = dims[ax], lims[ax]
        lo, hi = -1e9, 1e9
        for f, mv, mr in items:
            ck, dk = c[str(f)][ax], (SHIFT[mv][ax] if mv else 0)
            flipped = (mr == "h" and ax == 0) or (mr == "v" and ax == 1)
            a_, b_ = ((L - dk - ck, L - dk - ck + lim) if flipped else (ck - dk - lim, ck - dk))
            lo, hi = max(lo, a_), min(hi, b_)
        if lo > hi:
            return None
        u.append(min(max(L / 2, lo), hi))
    origins = []
    for f, mv, mr in items:
        o = []
        for ax in (0, 1):
            ck, dk, L = c[str(f)][ax], (SHIFT[mv][ax] if mv else 0), dims[ax]
            flipped = (mr == "h" and ax == 0) or (mr == "v" and ax == 1)
            pre = (L - (u[ax] + dk)) if flipped else (u[ax] + dk)          # object position in the crop before the flip
            o.append(int(round(ck - pre)))
        o = [min(max(o[0], 0), lims[0]), min(max(o[1], 0), lims[1])]        # rounding guard
        origins.append(o)
    return origins


def placement(q, table):
    """Origins for every image of a question, or None if some group cannot be placed."""
    pl = {"demos": []}
    for dm in q["demos"]:
        o = place(table, dm["object"], [(dm["before"]["frame"], None, None),
                                         (dm["after"]["frame"], dm["after"].get("move"), dm["after"].get("mirror"))])
        if o is None:
            return None
        pl["demos"].append(o)
    o = place(table, q["query"]["object"], [(q["query"]["frame"], None, None)] +
              [(c["frame"], c["move"], c.get("mirror")) for c in q["candidates"]])
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



def _demo_pairs(rng, table, objs3, starts, r, used):
    out = []
    for ob, o in zip(objs3, starts):
        b = pick(rng, table, ob, o, used); used.add((ob, b))
        a = pick(rng, table, ob, o + r, used) if r else b; used.add((ob, a))
        out.append((ob, o, b, a))
    return out


def mirror_questions(rng, table, n):
    """M: left-right mirror about the object's own vertical axis (digital flip of a recorded frame)."""
    objs, qs = list(table), []
    seen_queries = set()          # no two questions of this task share a query frame
    for _ in range(n):
        for _ in range(200):
            o4 = rng.sample(objs, 4)
            used = set()
            demos = [{"object": ob, "before": {"frame": b, "orient": o}, "after": {"frame": b, "orient": o, "mirror": "h"}}
                     for ob, o, b, _ in _demo_pairs(rng, table, o4[:3], rng.sample([0, 90, 180, 270], 3), 0, used)]
            q = rng.choice([0, 90, 180, 270]); qo = o4[3]
            qf = pick(rng, table, qo, q); used.add((qo, qf))
            f0 = pick(rng, table, qo, q, used)
            f180 = pick(rng, table, qo, q + 180, used)
            if None in [d["before"]["frame"] for d in demos] + [qf, f0, f180]:
                continue
            cq = {"task": "mirror", "rule": {"mirror": "left-right"}, "demos": demos,
                  "query": {"object": qo, "frame": qf, "orient": q},
                  "candidates": [
                      {"kind": "mirror left-right (correct)", "frame": qf, "orient": q, "move": None, "mirror": "h", "correct": True},
                      {"kind": "mirror top-bottom", "frame": qf, "orient": q, "move": None, "mirror": "v", "correct": False},
                      {"kind": "rotate 180 (recorded)", "frame": f180, "orient": (q + 180) % 360, "move": None, "correct": False},
                      {"kind": "unchanged (another recorded frame)", "frame": f0, "orient": q, "move": None, "correct": False}]}
            if (qo, qf) in seen_queries:
                continue
            if placement(cq, table) is not None:
                seen_queries.add((qo, qf))
                qs.append(cq)
                break
        else:
            raise RuntimeError("could not build a mirror question")
    return qs


def rotation_mirror_questions(rng, table, n):
    """RM: rotate by r (recorded) THEN mirror left-right (digital)."""
    objs, qs = list(table), []
    seen_queries = set()          # no two questions of this task share a query frame
    rules = ([90, -90] * n)[:n]
    rng.shuffle(rules)
    for r in rules:
        for _ in range(200):
            o4 = rng.sample(objs, 4)
            used = set()
            demos = [{"object": ob, "before": {"frame": b, "orient": o}, "after": {"frame": a, "orient": (o + r) % 360, "mirror": "h"}}
                     for ob, o, b, a in _demo_pairs(rng, table, o4[:3], rng.sample([0, 90, 180, 270], 3), r, used)]
            q = rng.choice([0, 90, 180, 270]); qo = o4[3]
            qf = pick(rng, table, qo, q); used.add((qo, qf))
            fr = pick(rng, table, qo, q + r, used)
            f0 = pick(rng, table, qo, q, used) or qf
            fw = pick(rng, table, qo, q - r, used)
            if None in [d["before"]["frame"] for d in demos] + [d["after"]["frame"] for d in demos] + [qf, fr, fw]:
                continue
            cq = {"task": "rotation+mirror", "rule": {"rotation_deg_ccw": r, "then": "mirror left-right"}, "demos": demos,
                  "query": {"object": qo, "frame": qf, "orient": q},
                  "candidates": [
                      {"kind": "rotate + mirror (correct)", "frame": fr, "orient": (q + r) % 360, "move": None, "mirror": "h", "correct": True},
                      {"kind": "rotate only", "frame": fr, "orient": (q + r) % 360, "move": None, "correct": False},
                      {"kind": "mirror only", "frame": f0, "orient": q, "move": None, "mirror": "h", "correct": False},
                      {"kind": "rotate the other way + mirror (= mirror first, then rotate: wrong order)", "frame": fw,
                       "orient": (q - r) % 360, "move": None, "mirror": "h", "correct": False}]}
            if (qo, qf) in seen_queries:
                continue
            if placement(cq, table) is not None:
                seen_queries.add((qo, qf))
                qs.append(cq)
                break
        else:
            raise RuntimeError("could not build a rotation+mirror question")
    return qs


def translation_mirror_questions(rng, table, n):
    """TM: mirror left-right about the object's own axis AND shift."""
    objs, qs = list(table), []
    seen_queries = set()          # no two questions of this task share a query frame
    rules = (DIRS * n)[:n]
    rng.shuffle(rules)
    for t in rules:
        for _ in range(200):
            o4 = rng.sample(objs, 4)
            used = set()
            demos = [{"object": ob, "before": {"frame": b, "orient": o}, "after": {"frame": b, "orient": o, "move": t, "mirror": "h"}}
                     for ob, o, b, _ in _demo_pairs(rng, table, o4[:3], rng.sample([0, 90, 180, 270], 3), 0, used)]
            q = rng.choice([0, 90, 180, 270]); qo = o4[3]
            qf = pick(rng, table, qo, q)
            if None in [d["before"]["frame"] for d in demos] + [qf]:
                continue
            cq = {"task": "translation+mirror", "rule": {"move": t, "mirror": "left-right"}, "demos": demos,
                  "query": {"object": qo, "frame": qf, "orient": q},
                  "candidates": [
                      {"kind": "mirror + move (correct)", "frame": qf, "orient": q, "move": t, "mirror": "h", "correct": True},
                      {"kind": "mirror only", "frame": qf, "orient": q, "move": None, "mirror": "h", "correct": False},
                      {"kind": "move only", "frame": qf, "orient": q, "move": t, "correct": False},
                      {"kind": "mirror top-bottom + move", "frame": qf, "orient": q, "move": t, "mirror": "v", "correct": False}]}
            if (qo, qf) in seen_queries:
                continue
            if placement(cq, table) is not None:
                seen_queries.add((qo, qf))
                qs.append(cq)
                break
        else:
            raise RuntimeError("could not build a translation+mirror question")
    return qs


def render(q, qid, video_dir, table, out: Path, answer_letter=None):
    d = out / qid
    (d / "images").mkdir(parents=True, exist_ok=True)
    vid = lambda ob: table[ob]["video"]
    pl = placement(q, table)
    q["crop_origins"] = {"demos": pl["demos"], "query": pl["query"]}
    files, rows = [], []
    for k, dm in enumerate(q["demos"], 1):
        b = crop(video_dir, vid(dm["object"]), dm["before"]["frame"], pl["demos"][k - 1][0])
        a = crop(video_dir, vid(dm["object"]), dm["after"]["frame"], pl["demos"][k - 1][1], dm["after"].get("mirror"))
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
    if answer_letter is not None:            # put the correct candidate at a pre-assigned (balanced) letter
        ci = next(i for i, c in enumerate(q["candidates"]) if c["correct"])
        order.remove(ci)
        order.insert("ABCD".index(answer_letter), ci)
    cand_row = []
    for letter, ci in zip("ABCD", order):
        c = q["candidates"][ci]
        im = crop(video_dir, vid(q["query"]["object"]), c["frame"], pl["candidates"][ci], c.get("mirror"))
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
    ap.add_argument("--tasks", default="R,T,RT", help="subset of R,T,RT,M,RM,TM to (re)build")
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
    if "M" in tasks:
        qs += mirror_questions(rng, table, a.n)
    if "RM" in tasks:
        qs += rotation_mirror_questions(rng, table, a.n)
    if "TM" in tasks:
        qs += translation_mirror_questions(rng, table, a.n)
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
    balanced = {}                              # M / RM / TM: correct letters balanced within each task
    for t in ("mirror", "rotation+mirror", "translation+mirror"):
        idx = [i for i, q in enumerate(qs) if q["task"] == t]
        letters = (list("ABCD") * len(idx))[:len(idx)]
        random.Random(f"{a.seed}-{t}-letters").shuffle(letters)
        balanced.update(zip(idx, letters))
    for i, q in enumerate(qs):
        qid = {"rotation": "R", "translation": "T", "rotation+translation": "RT", "mirror": "M",
               "rotation+mirror": "RM", "translation+mirror": "TM"}[q["task"]] + f"{i % a.n:02d}"
        if (a.out / qid).exists():
            import shutil
            shutil.rmtree(a.out / qid)
        q["answer"] = render(q, qid, a.videos, table, a.out, balanced.get(i))
        q["config"] = cfg
        answers["questions"][qid] = q
    answers["questions"] = dict(sorted(answers["questions"].items(), key=lambda kv: (["R", "T", "RT", "M", "RM", "TM"].index(kv[0].rstrip("0123456789")), kv[0])))
    ans_path.write_text(json.dumps(answers, indent=2))
    print(len(qs), "questions ->", a.out)


if __name__ == "__main__":
    main()
