"""One call per question for a batch made by make_questions.py (resumable; failed calls are retried on a re-run).

Usage:  OPENROUTER_API_KEY=... python eval_batch.py --batch batch1 [--model openai/gpt-6-luna] [--workers 8]

The request for each question is built from <batch>/<qid>/inputs.json and images only (eval_once.build_request);
answers.json is read only when scoring.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval_once import URL, build_request, parse_choice

HERE = Path(__file__).resolve().parent


def call(body, key):
    import requests
    try:
        r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=900)
        d = r.json()
        if r.status_code != 200 or "error" in d:
            return {"error": str(d.get("error", r.text[:1000])), "http_status": r.status_code}
        m = d["choices"][0]["message"]
        return {"error": None, "model_returned": d.get("model"), "raw_response": m.get("content"),
                "reasoning_summary": m.get("reasoning"), "finish_reason": d["choices"][0].get("finish_reason"),
                "usage": d.get("usage")}
    except Exception as ex:
        return {"error": f"{type(ex).__name__}: {ex}"}


def score(batch: Path, out_jsonl: Path):
    import pandas as pd
    ans = json.loads((batch / "answers.json").read_text())["questions"]
    recs = {}
    for line in out_jsonl.read_text().splitlines():
        r = json.loads(line)
        if r.get("error") is None:
            recs[r["qid"]] = r
    rows = []
    for qid, r in recs.items():
        q = ans[qid]
        ch = parse_choice(r.get("raw_response"))
        kind = next((c["kind"] for c in q["candidates"] if c["letter"] == ch), None)
        rows.append({"qid": qid, "task": q["task"], "rule": json.dumps(q["rule"]), "answer": q["answer"],
                     "choice": ch, "correct": ch == q["answer"], "chosen_kind": kind,
                     "reasoning_tokens": ((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
                     "cost": (r.get("usage") or {}).get("cost"), "raw": r.get("raw_response")})
    return pd.DataFrame(rows).sort_values("qid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=HERE / "batch1", type=Path)
    ap.add_argument("--model", default="openai/gpt-6-luna")
    ap.add_argument("--workers", default=8, type=int)
    a = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set; no call made.")
    out = a.batch / f"responses_{a.model.replace('/', '_')}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["qid"] for l in out.read_text().splitlines() if json.loads(l).get("error") is None}
    qids = sorted(p.name for p in a.batch.iterdir() if (p / "inputs.json").exists())
    todo = [q for q in qids if q not in done]
    print(f"{len(done)} done, {len(todo)} to call ({a.model})")
    lock = threading.Lock()

    def one(qid):
        body = build_request(a.batch / qid)
        body["model"] = a.model
        rec = {"qid": qid, "model_requested": a.model, "submission_format": "ordered_images_with_labels", **call(body, key)}
        with lock, out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(qid, rec.get("raw_response"), rec.get("error") or "", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))
    df = score(a.batch, out)
    df.to_csv(a.batch / f"scores_{a.model.replace('/', '_')}.csv", index=False)
    print(df.groupby("task").correct.agg(["mean", "sum", "size"]))


if __name__ == "__main__":
    main()
