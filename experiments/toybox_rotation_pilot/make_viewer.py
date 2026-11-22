"""Write <batch>/viewer.html: every question's composite, with the answer hidden behind a toggle."""
import html
import json
import sys
from pathlib import Path

batch = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "batch1")
a = json.loads((batch / "answers.json").read_text())["questions"]
names = {"rotation": "Rotation", "translation": "Translation", "rotation+translation": "Rotation + translation",
         "mirror": "Mirror", "rotation+mirror": "Rotation then mirror", "translation+mirror": "Translation + mirror"}
ROT = {90: "rotate 90° counter-clockwise", -90: "rotate 90° clockwise", 180: "rotate 180°"}


def rule(q):
    r = q["rule"]
    s = [ROT[r["rotation_deg_ccw"]]] if "rotation_deg_ccw" in r else []
    s += ["move " + r["move"]] if "move" in r else []
    s += ["then mirror left-right"] if "then" in r else (["mirror left-right"] if "mirror" in r else [])
    return " + ".join(s)


cards = []
for qid, q in a.items():
    cfg = q.get("config", {})
    shift = cfg.get("object_shift_px", {})
    cand = "".join(f"<li><b>{c['letter']}</b>: {html.escape(c['kind'])}{' ✅' if c['correct'] else ''}</li>"
                   for c in q["candidates"])
    cards.append(f'''<section id="{qid}"><h2>{qid} <span class="t">{names[q["task"]]}</span></h2>
<img src="{qid}/composite.jpg" loading="lazy">
<details><summary>Show answer</summary><p><b>Rule:</b> {rule(q)} &nbsp; <b>Answer:</b> {q["answer"]}<br>
<b>Demo objects:</b> {", ".join(d["object"] for d in q["demos"])} &nbsp; <b>Query:</b> {q["query"]["object"]} at about {q["query"]["orient"]}°<br>
<b>Crop:</b> {cfg.get("window")} &nbsp; <b>Shift (px, frame):</b> {shift.get("right", ["?"])[0]} horizontal / {shift.get("down", [0, "?"])[1]} vertical</p>
<ul>{cand}</ul></details></section>''')
nav = " ".join(f'<a href="#{k}">{k}</a>' for k in a)
page = f'''<!doctype html><meta charset="utf-8"><title>Toybox {batch.name}</title>
<style>body{{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:1200px;margin:20px auto;padding:0 16px;color:#222}}
nav{{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ddd;line-height:1.9}}nav a{{margin-right:8px}}
section{{margin:28px 0;border-bottom:1px solid #eee;padding-bottom:16px}}img{{max-width:100%;border:1px solid #ddd}}
.t{{font-size:15px;color:#666;font-weight:normal}}details{{margin-top:8px}}summary{{cursor:pointer;color:#0645ad}}</style>
<h1>Toybox {batch.name} ({len(a)} questions)</h1>
<p>Each composite shows 3 example pairs, the query and candidates A-D, exactly as sent to the model (as separate images).
Answers are hidden; click "Show answer".</p><nav>{nav}</nav>{"".join(cards)}'''
(batch / "viewer.html").write_text(page)
print("->", batch / "viewer.html")
