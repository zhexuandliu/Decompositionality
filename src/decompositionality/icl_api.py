"""In-context inference of grid transformations by an API model (OpenRouter), notebook 46.

A hidden transformation (a composition of A = shift every row right by one with wrap-around, B = rotate 90 degrees
clockwise, C = mirror left-right; executed left to right) is shown through a few input/output grid pairs. The model
must apply it to a new grid.

Pipeline (the "harness"):
  1. make_tasks: for every target word, N tasks with fresh random grids. The demos are checked so that EXACTLY ONE
     of the 288 transformations fits all of them; the query grid has 288 distinct images, so every answer grid
     identifies the transformation it corresponds to (or none).
  2. build_prompt: two experiments that differ ONLY in the final instruction:
       "answer" -- give the output grid;
       "ops"    -- first state the transformation as a sequence of the operations, then give the grid.
  3. run: call the model through OpenRouter (key from the environment variable OPENROUTER_API_KEY, never printed),
     with retries; every response is appended to a JSONL file and skipped on rerun (resumable, no double spending).
  4. score / summarize: exact correctness; for wrong answers, which legal transformation (shortest A/B/C word) the
     answer corresponds to; for "ops", whether the stated sequence equals the true transformation in the group and
     whether it matches the grid actually given.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import mirror_diag as M

N = 6
K2_WORDS = ["".join(p) for p in itertools.permutations("ABC", 2)]          # AB AC BA BC CA CB
EXPERIMENTS = ("answer", "ops")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADER = (
    "Each example below shows a 6x6 grid of digits (Input) and the grid obtained by applying the same hidden "
    "transformation to it (Output). The transformation is the same for every example. The hidden transformation is "
    "a sequence of one or more of these operations:\n"
    "(A) shift every row one cell to the right, wrapping around;\n"
    "(B) rotate the grid 90 degrees clockwise;\n"
    "(C) mirror the grid left-to-right."
)
GRID_FORMAT = "a 6x6 grid of digits in the same format (6 lines of 6 space-separated digits), between <answer> and </answer>."
INSTRUCTION = {
    "answer": "Give your answer as " + GRID_FORMAT,
    "ops": ("First, state the transformation as a sequence of these operations, between <ops> and </ops>.\n"
            "Then give your answer as " + GRID_FORMAT),
}


# ---------------------------------------------------------------------------------------------------------------------
# tasks and prompts
# ---------------------------------------------------------------------------------------------------------------------
def _fmt(g) -> str:
    return "\n".join(" ".join(str(int(v)) for v in row) for row in np.asarray(g).reshape(N, N))


def _consistent(G, xs, ys) -> List[int]:
    return [i for i in range(G.size)
            if all(np.array_equal(x.reshape(36)[G.perms[i]], y.reshape(36)) for x, y in zip(xs, ys))]


def make_tasks(words=K2_WORDS, n_per_word: int = 20, n_demos: int = 3, seed: int = 0) -> List[dict]:
    """Deterministic in (words, n_per_word, n_demos, seed). Every task: demos uniquely identify the transformation;
    the query grid has 288 distinct transformed images."""
    G = M.group()
    tasks = []
    for wi, w in enumerate(words):
        p = M.word_perm(w)
        rng = np.random.default_rng([seed, wi, n_demos])
        j = 0
        while j < n_per_word:
            xs = [rng.integers(0, 10, (N, N)) for _ in range(n_demos)]
            q = rng.integers(0, 10, (N, N))
            ys = [x.reshape(36)[p].reshape(N, N) for x in xs]
            if len(_consistent(G, xs, ys)) != 1:
                continue
            if len({q.reshape(36)[G.perms[i]].tobytes() for i in range(G.size)}) != G.size:
                continue
            tasks.append({"task_id": f"{w}-{j:03d}", "word": w, "group_id": int(G.word_id(w)),
                          "demo_inputs": [x.tolist() for x in xs], "demo_outputs": [y.tolist() for y in ys],
                          "query": q.tolist(), "expected": q.reshape(36)[p].reshape(N, N).tolist()})
            j += 1
    return tasks


def build_prompt(task: dict, experiment: str) -> str:
    demos = "\n\n".join(f"Example {i + 1}\nInput:\n{_fmt(x)}\nOutput:\n{_fmt(y)}"
                        for i, (x, y) in enumerate(zip(task["demo_inputs"], task["demo_outputs"])))
    return (f"{HEADER}\n\n{demos}\n\nApply the same transformation to this grid:\n{_fmt(task['query'])}\n\n"
            f"{INSTRUCTION[experiment]}")


# ---------------------------------------------------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------------------------------------------------
def call_openrouter(prompt: str, model: str, reasoning: Optional[dict] = None, max_tokens: Optional[int] = None,
                    timeout: float = 600, retries: int = 5) -> dict:
    """One chat completion. Returns {"content", "reasoning", "usage", "model", "error"}. The key is read from the
    environment and never logged."""
    import requests
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "usage": {"include": True}}
    if reasoning:
        body["reasoning"] = reasoning
    if max_tokens:
        body["max_tokens"] = max_tokens
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(min(60, 2 ** attempt * 3))
                continue
            if r.status_code != 200:
                return {"content": None, "reasoning": None, "usage": None, "model": None,
                        "error": f"HTTP {r.status_code}: {r.text[:500]}"}
            d = r.json()
            if "error" in d:
                last = str(d["error"])[:500]
                time.sleep(min(60, 2 ** attempt * 3))
                continue
            msg = d["choices"][0]["message"]
            return {"content": msg.get("content"), "reasoning": msg.get("reasoning"), "usage": d.get("usage"),
                    "model": d.get("model"), "error": None}
        except Exception as ex:                                   # network errors, timeouts
            last = f"{type(ex).__name__}: {str(ex)[:300]}"
            time.sleep(min(60, 2 ** attempt * 3))
    return {"content": None, "reasoning": None, "usage": None, "model": None, "error": f"failed after retries: {last}"}


def run(tasks: List[dict], out_jsonl, model: str, experiments=EXPERIMENTS, reasoning: Optional[dict] = None,
        max_tokens: Optional[int] = None, workers: int = 4, limit: Optional[int] = None,
        call: Callable = call_openrouter, log=print) -> int:
    """Calls the model for every (task, experiment) not yet in ``out_jsonl``; appends one JSON line per call.
    ``limit`` caps the number of NEW calls (pilot). Returns the number of new calls made."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out = Path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("error") is None:
                    done.add((r["task_id"], r["experiment"]))
    todo = [(t, e) for t in tasks for e in experiments if (t["task_id"], e) not in done]   # interleaved: pilot covers both
    if limit is not None:
        todo = todo[:limit]
    log(f"{len(done)} calls already done; {len(todo)} to run (model {model})")
    lock = threading.Lock()
    n_ok = 0

    def one(t, e):
        prompt = build_prompt(t, e)
        t0 = time.time()
        res = call(prompt, model, reasoning=reasoning, max_tokens=max_tokens)
        return {"task_id": t["task_id"], "word": t["word"], "experiment": e, "model_requested": model,
                "reasoning_setting": reasoning, "seconds": round(time.time() - t0, 1), **res}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(one, t, e) for t, e in todo]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            with lock:
                with out.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            n_ok += rec["error"] is None
            if i % 10 == 0 or i == len(futs):
                log(f"  {i}/{len(futs)} calls finished ({n_ok} ok)")
    return len(todo)


# ---------------------------------------------------------------------------------------------------------------------
# parsing and scoring
# ---------------------------------------------------------------------------------------------------------------------
def parse_answer(text: Optional[str]) -> Optional[np.ndarray]:
    if not text:
        return None
    m = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    if not m:
        return None
    digits = re.findall(r"\d", m[-1])
    return np.array([int(d) for d in digits]).reshape(N, N) if len(digits) == 36 else None


def parse_ops(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = re.findall(r"<ops>(.*?)</ops>", text, re.S)
    if not m:
        return None
    num = {"once": 1, "twice": 2, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
    pat = (r"(?<![A-Za-z])([ABC])(?![A-Za-z])"
           r"(?:\s*(?:x|×)\s*(\d)|\s+(\d|two|three|four|five|six|seven)\s+times|\s+(twice|once))?")
    toks = re.findall(pat, m[-1])
    if toks:   # letters, with repetitions such as "A five times", "A x3", "B twice"
        out = ""
        for ch, n1, n2, n3 in toks:
            n = n1 or n2 or n3
            out += ch * (int(n) if n.isdigit() else num.get(n, 1)) if n else ch
        return out
    # the model sometimes describes the operations in words instead of letters
    words = re.findall(r"shift|rotat|mirror|flip|reflect", m[-1].lower())
    return "".join({"shift": "A", "rotat": "B"}.get(w, "C") for w in words)


def score(tasks: List[dict], out_jsonl) -> "pd.DataFrame":
    """One row per successful call (the latest record per task/experiment)."""
    import pandas as pd
    from .shortcut_analysis import shortest_words
    G = M.group()
    SW = shortest_words(G)
    by_id = {t["task_id"]: t for t in tasks}
    recs = {}
    for line in Path(out_jsonl).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("error") is None and r["task_id"] in by_id:
                recs[(r["task_id"], r["experiment"])] = r
    rows = []
    for (tid, e), r in recs.items():
        t = by_id[tid]
        q = np.array(t["query"]).reshape(36)
        ans = parse_answer(r.get("content"))
        row = {"task_id": tid, "word": t["word"], "experiment": e, "parsed": ans is not None,
               "correct": bool(ans is not None and np.array_equal(ans, np.array(t["expected"])))}
        applied = []
        if ans is not None:
            applied = [i for i in range(G.size) if np.array_equal(q[G.perms[i]], ans.reshape(36))]
        row["applied"] = SW[applied[0]] if applied else ("unparsed" if ans is None else "not a legal transform")
        if e == "ops":
            ops = parse_ops(r.get("content"))
            row["ops_stated"] = ops if ops is not None else "unparsed"
            ok = ops is not None and len(ops) > 0
            row["ops_equivalent"] = bool(ok and G.word_id(ops) == t["group_id"])
            row["ops_literal"] = bool(ok and ops == t["word"])
            row["ops_consistent_with_answer"] = bool(ok and ans is not None and
                                                     np.array_equal(q[M.word_perm(ops)], ans.reshape(36)))
        u = r.get("usage") or {}
        row["prompt_tokens"] = u.get("prompt_tokens")
        row["completion_tokens"] = u.get("completion_tokens")
        row["reasoning_tokens"] = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        row["cost_usd"] = u.get("cost")
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df) -> Dict[str, "pd.DataFrame"]:
    import pandas as pd
    out = {}
    out["accuracy"] = (df.pivot_table(index="word", columns="experiment", values="correct", aggfunc="mean")
                       .round(3).reset_index())
    out["n_calls"] = df.groupby("experiment").size().rename("calls").reset_index()
    wrong = df[~df.correct]
    out["wrong_answers"] = (wrong.groupby(["experiment", "word", "applied"]).size().rename("count").reset_index()
                            .sort_values(["experiment", "word", "count"], ascending=[True, True, False]))
    if (df.experiment == "ops").any():
        o = df[df.experiment == "ops"]
        out["ops"] = o.groupby("word")[["ops_equivalent", "ops_literal", "ops_consistent_with_answer", "correct"]] \
            .mean().round(3).reset_index()
        out["ops_stated"] = (o.groupby(["word", "ops_stated"]).size().rename("count").reset_index()
                             .sort_values(["word", "count"], ascending=[True, False]))
    cost = df[["prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_usd"]].apply(
        pd.to_numeric, errors="coerce").sum().to_frame("total").T
    out["usage"] = cost
    return out
