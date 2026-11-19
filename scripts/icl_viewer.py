"""Turn an in-context API results file (responses.jsonl / repR_demosN.jsonl) into a readable HTML page.

    python scripts/icl_viewer.py path/to/folder                       # index.html + one page per .jsonl
    python scripts/icl_viewer.py path/to/rep0_demos3.jsonl            # tasks.json is looked up next to it
    python scripts/icl_viewer.py path/to/responses.jsonl --tasks path/to/tasks.json

Each call is shown as a card: the demonstrations, the query, the expected grid, the model's grid (wrong cells
highlighted), the stated operations, what the answer actually corresponds to (one of the 288 transformations or
none), and the model's reasoning summary. Filters: experiment, correct / wrong, error type, transformation.
The page is written next to the .jsonl file (``<name>_viewer.html``) and opened in the browser.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import webbrowser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decompositionality import icl_api as IA  # noqa: E402

CSS = """
:root{--bg:#ffffff;--fg:#1d1d1f;--muted:#6e6e73;--card:#f7f7f8;--line:#d9d9de;--ok:#1a7f4b;--okbg:#e3f4ea;
--bad:#b42318;--badbg:#fde7e4;--warn:#9a6700;--warnbg:#fff4d6;--cell:#ffffff;--accent:#0b63c5}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141416;--fg:#ececf0;--muted:#a0a0a8;
--card:#1e1e22;--line:#38383f;--ok:#5fd394;--okbg:#173524;--bad:#ff8a7a;--badbg:#3d1c18;--warn:#f0c058;
--warnbg:#3a2f12;--cell:#26262b;--accent:#6aa9ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,
BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1200px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:4px 0 4px}.sub{color:var(--muted);margin-bottom:12px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;position:sticky;top:0;background:var(--bg);
padding:8px 0;border-bottom:1px solid var(--line);z-index:5}
select,input{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 6px}
.count{color:var(--muted);margin-left:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:12px 0}
.head{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px}
.tid{font-weight:600}.pill{border-radius:999px;padding:1px 9px;font-size:12px;border:1px solid var(--line)}
.ok{color:var(--ok);background:var(--okbg);border-color:transparent}.bad{color:var(--bad);background:var(--badbg);
border-color:transparent}.warn{color:var(--warn);background:var(--warnbg);border-color:transparent}
.grids{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start}
.g{display:flex;flex-direction:column;gap:3px}.g .lab{font-size:12px;color:var(--muted)}
table.grid{border-collapse:collapse;font:13px/1 ui-monospace,SFMono-Regular,Menlo,monospace}
table.grid td{width:20px;height:20px;text-align:center;border:1px solid var(--line);background:var(--cell)}
table.grid td.x{background:var(--badbg);color:var(--bad);font-weight:700}
.pair{display:flex;gap:6px;align-items:center}.arrow{color:var(--muted)}
.sep{width:1px;align-self:stretch;background:var(--line)}
details{margin-top:8px}summary{cursor:pointer;color:var(--accent)}
pre{white-space:pre-wrap;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px;
font:12px/1.45 ui-monospace,Menlo,monospace;max-height:420px;overflow:auto}
.meta{color:var(--muted);font-size:12px}
@media (max-width:600px){main{padding:12px 16px}table.grid td{width:16px;height:16px;font-size:11px}}
"""

JS = """
const cards=[...document.querySelectorAll('.card')];
const f={exp:document.getElementById('fexp'),res:document.getElementById('fres'),
err:document.getElementById('ferr'),word:document.getElementById('fword'),q:document.getElementById('fq')};
function apply(){let n=0;for(const c of cards){const d=c.dataset;
const show=(!f.exp.value||d.exp===f.exp.value)&&(!f.res.value||d.res===f.res.value)&&
(!f.err.value||d.err===f.err.value)&&(!f.word.value||d.word===f.word.value)&&
(!f.q.value||d.tid.toLowerCase().includes(f.q.value.toLowerCase()));c.style.display=show?'':'none';if(show)n++;}
document.getElementById('count').textContent=n+' of '+cards.length+' calls';}
for(const k in f){f[k].addEventListener('input',apply);}apply();
"""


def grid_html(g, wrong=None, label=""):
    g = np.asarray(g).reshape(6, 6)
    rows = []
    for r in range(6):
        tds = "".join(f'<td class="{"x" if wrong is not None and wrong[r, c] else ""}">{int(g[r, c])}</td>'
                      for c in range(6))
        rows.append(f"<tr>{tds}</tr>")
    return f'<div class="g"><span class="lab">{html.escape(label)}</span><table class="grid">{"".join(rows)}</table></div>'


def render_file(path: Path, tasks_path: Path = None, replicate: int = None, back_link: str = None):
    """Write <name>_viewer.html next to ``path``; returns (output path, scored DataFrame)."""
    tasks_path = tasks_path or path.parent / "tasks.json"
    T = json.loads(tasks_path.read_text())
    m = re.match(r"rep(\d+)_demos(\d+)\.jsonl$", path.name)
    if isinstance(T, dict):                                   # demo sweep: {replicate: [tasks with 10 demos]}
        rep = replicate if replicate is not None else (int(m.group(1)) if m else 0)
        tasks = T[str(rep)]
        if m:
            tasks = IA.with_demos(tasks, int(m.group(2)))
    else:
        tasks = T
    by_id = {t["task_id"]: t for t in tasks}
    df = IA.score(tasks, path)
    scored = {(r.task_id, r.experiment): r for r in df.itertuples()}
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("error") is None and r["task_id"] in by_id]
    recs.sort(key=lambda r: (r["task_id"], r["experiment"]))
    cards = []
    for r in recs:
        t = by_id[r["task_id"]]
        s = scored.get((r["task_id"], r["experiment"]))
        exp = np.array(t["expected"]).reshape(6, 6)
        ans = IA.parse_answer(r.get("content"))
        demos = "".join(
            f'<div class="pair">{grid_html(x, label=f"demo {i + 1} input")}<span class="arrow">→</span>'
            f'{grid_html(y, label="output")}</div>'
            for i, (x, y) in enumerate(zip(t["demo_inputs"], t["demo_outputs"])))
        answer = (grid_html(ans, wrong=(ans != exp), label="model answer (wrong cells red)") if ans is not None
                  else '<div class="g"><span class="lab">model answer</span><em>could not parse</em></div>')
        ok = bool(s and s.correct)
        err = getattr(s, "error_type", "unparsed") if s is not None else "unparsed"
        pills = [f'<span class="pill {"ok" if ok else "bad"}">{"correct" if ok else "wrong"}</span>',
                 f'<span class="pill">true: {t["word"]}</span>']
        if s is not None and not ok:
            pills.append(f'<span class="pill warn">{html.escape(err)}</span>')
            pills.append(f'<span class="pill">answer corresponds to: {html.escape(str(s.applied))}</span>')
        if r["experiment"] == "ops" and s is not None:
            eq = bool(s.ops_equivalent)
            pills.append(f'<span class="pill {"ok" if eq else "bad"}">stated ops: '
                         f'{html.escape(str(s.ops_stated)) or "—"}</span>')
        rule_block = ""
        if r["experiment"] == "describe" and s is not None:
            rule_block = f'<div class="meta"><b>stated rule:</b> {html.escape(str(s.rule))}</div>'
        u = r.get("usage") or {}
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        meta = (f'{r["seconds"]} s · reasoning tokens {rt} · cost ${u.get("cost", 0) or 0:.5f} · '
                f'model {html.escape(str(r.get("model")))}')
        reasoning = html.escape(r.get("reasoning") or "(no reasoning summary returned)")
        content = html.escape(r.get("content") or "")
        cards.append(f"""
<div class="card" data-exp="{r['experiment']}" data-res="{'correct' if ok else 'wrong'}" data-err="{html.escape(err)}"
 data-word="{t['word']}" data-tid="{r['task_id']}">
 <div class="head"><span class="tid">{r['task_id']}</span><span class="pill">{r['experiment']}</span>{''.join(pills)}</div>
 <div class="grids">{demos}<div class="sep"></div>{grid_html(t['query'], label='query')}
 {grid_html(exp, label='expected')}{answer}</div>
 {rule_block}<div class="meta">{meta}</div>
 <details><summary>reasoning summary</summary><pre>{reasoning}</pre></details>
 <details><summary>raw answer text</summary><pre>{content}</pre></details>
</div>""")
    words = sorted({t["word"] for t in tasks})
    errs = sorted(df.error_type.unique()) if "error_type" in df else []
    acc = df.groupby("experiment").correct.mean().round(3).to_dict() if len(df) else {}
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>ICL Results Viewer</title>
<style>{CSS}</style></head><body><main>
{f'<div class="sub"><a href="{back_link}">← all files</a></div>' if back_link else ''}<h1>{html.escape(path.name)}</h1>
<div class="sub">{len(recs)} calls · accuracy by experiment: {html.escape(json.dumps(acc))} ·
hidden transformations: A = shift rows right, B = rotate 90° clockwise, C = mirror left–right (left to right)</div>
<div class="bar">
 <select id="fexp"><option value="">all experiments</option><option>answer</option><option>ops</option><option>describe</option></select>
 <select id="fres"><option value="">correct + wrong</option><option>correct</option><option>wrong</option></select>
 <select id="ferr"><option value="">all error types</option>{''.join(f'<option>{html.escape(e)}</option>' for e in errs)}</select>
 <select id="fword"><option value="">all transformations</option>{''.join(f'<option>{w}</option>' for w in words)}</select>
 <input id="fq" placeholder="task id, e.g. CA-003">
 <span class="count" id="count"></span>
</div>
{''.join(cards)}
</main><script>{JS}</script></body></html>"""
    out = path.with_name(path.stem + "_viewer.html")
    out.write_text(page)
    return out, df


def render_dir(d: Path) -> Path:
    """One viewer per .jsonl in ``d`` plus index.html: accuracy by demos x experiment, error types, figures, links."""
    import pandas as pd
    files = sorted(d.glob("*.jsonl"), key=lambda p: (p.name.split("_")[0],
                   int(re.search(r"demos(\d+)", p.name).group(1)) if re.search(r"demos(\d+)", p.name) else 0))
    parts, rows = [], []
    for f in files:
        out, df = render_file(f, back_link="index.html")
        m = re.match(r"rep(\d+)_demos(\d+)\.jsonl$", f.name)
        if m and len(df):
            parts.append(df.assign(replicate=int(m.group(1)), n_demos=int(m.group(2))))
        acc = df.groupby("experiment").correct.mean().round(3).to_dict() if len(df) else {}
        rows.append((out.name, f.name, len(df), acc))
        print("written:", out)
    exps = [e for e in ("answer", "describe", "ops") if any(e in r[3] for r in rows)]
    links = [f'<tr><td><a href="{o}">{html.escape(n)}</a></td><td>{c}</td>' + "".join(f"<td>{a.get(e, '')}</td>"
             for e in exps) + "</tr>" for o, n, c, a in rows]
    tables = ""
    if parts:
        D = pd.concat(parts, ignore_index=True)
        acc = D.pivot_table(index="n_demos", columns="experiment", values="correct", aggfunc="mean").round(3)
        err = (D.groupby(["experiment", "n_demos"]).error_type.value_counts(normalize=True)
               .unstack(fill_value=0).round(3))
        byw = D.pivot_table(index="word", columns=["experiment", "n_demos"], values="correct", aggfunc="mean").round(2)
        wt = D[D.error_type == "wrong transform"].groupby(["word", "applied"]).size().rename("count") \
            .reset_index().sort_values("count", ascending=False).head(15)
        sty = lambda t: t.to_html(classes="tab", border=0)  # noqa: E731
        tables = (f"<h2>Accuracy by number of demos</h2>{sty(acc)}<h2>Error types (share of answers)</h2>{sty(err)}"
                  f"<h2>Accuracy by transformation</h2>{sty(byw)}"
                  f"<h2>Wrong legal transforms (what was applied instead)</h2>{sty(wt)}")
        if "rule_keywords" in D:
            rk = (D[D.experiment == "describe"].groupby(["word", "rule_keywords"]).size().rename("count").reset_index()
                  .sort_values(["word", "count"], ascending=[True, False]).groupby("word").head(4))
            tables += f"<h2>Operation words in the stated rule (heuristic), top 4 per transformation</h2>{sty(rk)}"
    figs = "".join(f'<figure><img src="{p.name}" alt="{p.stem}"><figcaption>{p.stem}</figcaption></figure>'
                   for p in sorted(d.glob("*.png")))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>ICL Results Index</title>
<style>{CSS}
table.tab{{border-collapse:collapse;font-size:13px;margin:6px 0 14px}}table.tab th,table.tab td{{border:1px solid var(--line);
padding:3px 8px;text-align:right}}table.tab th{{background:var(--card)}}figure{{display:inline-block;margin:6px 10px 6px 0}}
figure img{{max-width:520px;width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}}
figcaption{{color:var(--muted);font-size:12px}}h2{{font-size:16px;margin:18px 0 6px}}
.wrap{{overflow-x:auto}}</style></head><body><main>
<h1>{html.escape(d.name)}</h1><div class="sub">{len(files)} result files · click a file to browse every call</div>
<div class="wrap"><table class="tab"><tr><th>file</th><th>calls</th>{''.join(f'<th>{e}</th>' for e in exps)}</tr>
{''.join(links)}</table></div>
<div class="wrap">{tables}</div><h2>Figures</h2>{figs}</main></body></html>"""
    out = d / "index.html"
    out.write_text(page)
    print("written:", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a .jsonl results file, or a folder of them (writes index.html + one page per file)")
    ap.add_argument("--tasks", default=None, help="tasks.json (default: next to the .jsonl)")
    ap.add_argument("--replicate", type=int, default=None, help="replicate key in tasks.json (sweep files)")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    path = Path(a.path).expanduser()
    if path.is_dir():
        out = render_dir(path)
    else:
        out, _ = render_file(path, Path(a.tasks).expanduser() if a.tasks else None, a.replicate)
        print("written:", out)
    if not a.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
