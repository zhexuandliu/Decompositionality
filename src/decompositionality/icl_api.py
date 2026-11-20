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
EXPERIMENTS = ("answer", "ops")                 # with the operation hint (notebooks 46 / 47)
EXPERIMENTS_NEUTRAL = ("answer", "describe")    # no hint: give the grid / describe the rule in words first
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# NEUTRAL header (default): nothing about which operations exist or how many steps the transformation has.
HEADER_NEUTRAL = (
    "Each example below shows a 6x6 grid of digits (Input) and the grid obtained by applying the same hidden "
    "transformation to it (Output). The transformation is the same for every example."
)
# HINT header (notebooks 46 / 47 only): lists the three operations. Kept to reproduce those runs.
HEADER = (
    "Each example below shows a 6x6 grid of digits (Input) and the grid obtained by applying the same hidden "
    "transformation to it (Output). The transformation is the same for every example. The hidden transformation is "
    "a sequence of one or more of these operations:\n"
    "(A) shift every row one cell to the right, wrapping around;\n"
    "(B) rotate the grid 90 degrees clockwise;\n"
    "(C) mirror the grid left-to-right."
)
# STEPS header (notebook 52): says the transformation is a sequence of rigid transformations of a wrap-around grid,
# without saying which ones, how many, or naming any of them.
HEADER_STEPS = (
    HEADER_NEUTRAL + " The hidden transformation is a sequence of one or more rigid transformations (distance-preserving "
    "maps) of the grid, where the grid wraps around at its edges."
)
EXPERIMENTS_STEPS = ("answer", "steps")
GRID_FORMAT = "a 6x6 grid of digits in the same format (6 lines of 6 space-separated digits), between <answer> and </answer>."
INSTRUCTION = {
    "answer": "Give your answer as " + GRID_FORMAT,
    "describe": ("First, describe the transformation in words, between <rule> and </rule>.\n"
                 "Then give your answer as " + GRID_FORMAT),
    "steps": ("First, break the transformation down into the simplest steps it is made of. Describe each step, in "
              "order, between <step> and </step>.\n"
              "Then give your final answer as " + GRID_FORMAT),
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


def make_nested_tasks(words=K2_WORDS, n_per_word: int = 20, max_demos: int = 10, seed: int = 0) -> List[dict]:
    """Tasks with ``max_demos`` demos each, NESTED across demo counts: the task with n demos uses the first n. The
    FIRST demo alone already determines the transformation uniquely (so every prefix does), and the query grid has
    288 distinct images. Deterministic in (words, n_per_word, max_demos, seed)."""
    G = M.group()
    tasks = []
    for wi, w in enumerate(words):
        p = M.word_perm(w)
        rng = np.random.default_rng([seed, wi, max_demos, 7])
        j = 0
        while j < n_per_word:
            xs = [rng.integers(0, 10, (N, N)) for _ in range(max_demos)]
            q = rng.integers(0, 10, (N, N))
            ys = [x.reshape(36)[p].reshape(N, N) for x in xs]
            if len(_consistent(G, xs[:1], ys[:1])) != 1:
                continue
            if len({q.reshape(36)[G.perms[i]].tobytes() for i in range(G.size)}) != G.size:
                continue
            tasks.append({"task_id": f"{w}-{j:03d}", "word": w, "group_id": int(G.word_id(w)),
                          "demo_inputs": [x.tolist() for x in xs], "demo_outputs": [y.tolist() for y in ys],
                          "query": q.tolist(), "expected": q.reshape(36)[p].reshape(N, N).tolist()})
            j += 1
    return tasks


def make_tasks_by_length(ks=(1, 3, 4, 5), n_per_k: int = 60, n_demos: int = 3, seed: int = 0,
                         n_per_transform: Optional[int] = None) -> List[dict]:
    """Tasks whose hidden transformation needs EXACTLY k operations: group elements whose shortest A/B/C word has
    length k (so e.g. CCA = A counts as k = 1, never as 3). For each k the elements are shuffled and cycled, so every
    element appears about n_per_k / (#elements) times; with ``n_per_transform`` every element gets exactly that many
    tasks instead. ``word`` is one shortest word; scoring uses ``group_id``.
    Demos are checked to determine the transformation uniquely; the query has 288 distinct images."""
    from .shortcut_analysis import shortest_words
    G = M.group()
    SW = shortest_words(G)
    tasks = []
    for k in ks:
        elems = sorted(g for g, w in SW.items() if w != "I" and len(w) == k)
        rng = np.random.default_rng([seed, k, n_demos, 11])
        order = [elems[i] for i in rng.permutation(len(elems))]
        n_k = n_per_transform * len(elems) if n_per_transform else n_per_k
        j = 0
        while j < n_k:
            gid = order[j % len(order)]
            p = G.perms[gid]
            xs = [rng.integers(0, 10, (N, N)) for _ in range(n_demos)]
            q = rng.integers(0, 10, (N, N))
            ys = [x.reshape(36)[p].reshape(N, N) for x in xs]
            if len(_consistent(G, xs, ys)) != 1:
                continue
            if len({q.reshape(36)[G.perms[i]].tobytes() for i in range(G.size)}) != G.size:
                continue
            tasks.append({"task_id": f"k{k}-{j:03d}", "word": SW[gid], "k": k, "group_id": int(gid),
                          "demo_inputs": [x.tolist() for x in xs], "demo_outputs": [y.tolist() for y in ys],
                          "query": q.tolist(), "expected": q.reshape(36)[p].reshape(N, N).tolist()})
            j += 1
    return tasks


# ---------------------------------------------------------------------------------------------------------------------
# notebook 51: composition in context (rules shown separately, asked for their composition)
# ---------------------------------------------------------------------------------------------------------------------
COMPOSE_CONDITIONS = ("compose", "direct")


def make_compose_tasks(pairs=K2_WORDS, n_per_pair: int = 10, n_demos: int = 3, seed: int = 0) -> List[dict]:
    """Each task: Rule P = the first operation of the pair, Rule Q = the second, each shown with ``n_demos`` examples
    that determine it uniquely among the 288 transformations; a "direct" set of ``n_demos`` examples of the composite
    (P then Q), also unique; and one query grid with 288 distinct images. Expected answer = P then Q on the query."""
    G = M.group()
    tasks = []

    def demos(rng, perm):
        while True:
            xs = [rng.integers(0, 10, (N, N)) for _ in range(n_demos)]
            ys = [x.reshape(36)[perm].reshape(N, N) for x in xs]
            if len(_consistent(G, xs, ys)) == 1:
                return [x.tolist() for x in xs], [y.tolist() for y in ys]

    for wi, w in enumerate(pairs):
        assert len(w) == 2 and G.word_id(w) not in (G.word_id(w[0]), G.word_id(w[1]), G.word_id(w[::-1]), G.I), w
        rng = np.random.default_rng([seed, wi, n_demos, 51])
        for j in range(n_per_pair):
            pin, pout = demos(rng, M.word_perm(w[0]))
            qin, qout = demos(rng, M.word_perm(w[1]))
            din, dout = demos(rng, M.word_perm(w))
            while True:
                q = rng.integers(0, 10, (N, N))
                if len({q.reshape(36)[G.perms[i]].tobytes() for i in range(G.size)}) == G.size:
                    break
            tasks.append({"task_id": f"{w}-{j:03d}", "word": w, "group_id": int(G.word_id(w)),
                          "rule_p_inputs": pin, "rule_p_outputs": pout, "rule_q_inputs": qin, "rule_q_outputs": qout,
                          "demo_inputs": din, "demo_outputs": dout, "query": q.tolist(),
                          "expected": q.reshape(36)[M.word_perm(w)].reshape(N, N).tolist()})
    return tasks


def _examples(xs, ys):
    return "\n\n".join(f"Example {i + 1}\nInput:\n{_fmt(x)}\nOutput:\n{_fmt(y)}" for i, (x, y) in enumerate(zip(xs, ys)))


def build_compose_prompt(task: dict, condition: str) -> str:
    """Neutral prompts: nothing about which operations exist.
    compose: Rule P and Rule Q shown separately; asked for "Rule P, then Rule Q" (never shown together).
    direct:  examples of the composite transformation itself (the holistic baseline)."""
    tail = "Give your answer as " + GRID_FORMAT
    if condition == "compose":
        return ("Below are examples of two hidden transformations, Rule P and Rule Q. Each example shows a 6x6 grid of "
                "digits (Input) and the grid obtained by applying that rule to it (Output).\n\n"
                f"Rule P\n\n{_examples(task['rule_p_inputs'], task['rule_p_outputs'])}\n\n"
                f"Rule Q\n\n{_examples(task['rule_q_inputs'], task['rule_q_outputs'])}\n\n"
                f"Apply Rule P and then Rule Q to this grid:\n{_fmt(task['query'])}\n\n{tail}")
    if condition == "direct":
        return (f"{HEADER_NEUTRAL}\n\n{_examples(task['demo_inputs'], task['demo_outputs'])}\n\n"
                f"Apply the same transformation to this grid:\n{_fmt(task['query'])}\n\n{tail}")
    raise ValueError(condition)


def with_demos(tasks: List[dict], n: int) -> List[dict]:
    """The same tasks restricted to their first n demos."""
    return [{**t, "demo_inputs": t["demo_inputs"][:n], "demo_outputs": t["demo_outputs"][:n]} for t in tasks]


def build_prompt(task: dict, experiment: str, hint: bool, style: Optional[str] = None) -> str:
    """``hint`` must be given explicitly: False = neutral prompt (no information about the operations);
    True = the notebook-46/47 prompt that lists operations A, B, C ("ops" experiment only with the hint).
    ``style="steps"`` (with hint=False): the header also says the transformation is a sequence of simple geometric
    operations that only move digits (notebook 52), without naming any operation."""
    if experiment == "ops" and not hint:
        raise ValueError("the 'ops' experiment names operations A/B/C and needs hint=True; use 'describe'")
    if style == "steps" and hint:
        raise ValueError("style='steps' is used without the operation hint")
    header = HEADER if hint else (HEADER_STEPS if style == "steps" else HEADER_NEUTRAL)
    demos = "\n\n".join(f"Example {i + 1}\nInput:\n{_fmt(x)}\nOutput:\n{_fmt(y)}"
                        for i, (x, y) in enumerate(zip(task["demo_inputs"], task["demo_outputs"])))
    return (f"{header}\n\n{demos}\n\nApply the same transformation to this grid:\n{_fmt(task['query'])}\n\n"
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


def run(tasks: List[dict], out_jsonl, model: str, experiments=EXPERIMENTS, *, hint: bool, reasoning: Optional[dict] = None,
        max_tokens: Optional[int] = None, workers: int = 4, limit: Optional[int] = None,
        call: Callable = call_openrouter, log=print, style: Optional[str] = None) -> int:
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
        prompt = build_compose_prompt(t, e) if "rule_p_inputs" in t else build_prompt(t, e, hint, style)
        t0 = time.time()
        res = call(prompt, model, reasoning=reasoning, max_tokens=max_tokens)
        return {"task_id": t["task_id"], "word": t["word"], "experiment": e, "hint": hint, "model_requested": model, "style": style,
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


def parse_steps(text: Optional[str]) -> List[tuple]:
    """[(description, 6x6 grid or None), ...] from <step>...</step> <grid>...</grid> pairs, in order (before <answer>)."""
    if not text:
        return []
    body = text.split("<answer>")[0]
    out = []
    for m in re.finditer(r"<step>(.*?)</step>\s*(?:<grid>(.*?)</grid>)?", body, re.S):
        digits = re.findall(r"\d", m.group(2) or "")
        out.append((m.group(1).strip(), np.array([int(d) for d in digits]).reshape(N, N) if len(digits) == 36 else None))
    return out


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


def parse_rule(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = re.findall(r"<rule>(.*?)</rule>", text, re.S)
    return m[-1].strip() if m else None


RULE_KEYWORDS = [("rotat", "rotate"), ("transpos", "transpose"), ("mirror", "mirror"), ("flip", "flip"),
                 ("reflect", "reflect"), ("reverse", "reverse"), ("shift", "shift"), ("translat", "shift"),
                 ("roll", "shift"), ("cyclic", "shift"), ("wrap", "shift")]


def rule_keywords(rule: Optional[str]) -> str:
    """Heuristic: the operation-like words in a free-text description, in order (for browsing, not scoring)."""
    if not rule:
        return ""
    low = rule.lower()
    hits = []
    for key, name in RULE_KEYWORDS:
        hits += [(m.start(), name) for m in re.finditer(key, low)]
    out = []
    for _, name in sorted(hits):
        if not out or out[-1] != name:
            out.append(name)
    return " > ".join(out)


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
        row = {"task_id": tid, "word": t["word"], "k": t.get("k", len(t["word"])), "experiment": e,
               "parsed": ans is not None,
               "correct": bool(ans is not None and np.array_equal(ans, np.array(t["expected"])))}
        applied = []
        if ans is not None:
            applied = [i for i in range(G.size) if np.array_equal(q[G.perms[i]], ans.reshape(36))]
        row["applied"] = SW[applied[0]] if applied else ("unparsed" if ans is None else "not a legal transform")
        if "rule_p_inputs" in t and applied:          # composition tasks: what did the model apply instead?
            s_ = set(applied)
            p_, q_ = t["word"][0], t["word"][1]
            row["applied_is_p_only"] = G.word_id(p_) in s_
            row["applied_is_q_only"] = G.word_id(q_) in s_
            row["applied_is_reversed"] = G.word_id(q_ + p_) in s_
        # error type: correct / wrong legal transform (misidentified) / execution slip (the nearest legal transform is
        # the true one and >= 30 of 36 cells agree) / other wrong grid / unparsed
        if ans is None:
            row["error_type"] = "unparsed"
        elif row["correct"]:
            row["error_type"] = "correct"
        elif applied:
            row["error_type"] = "wrong transform"
        else:
            nb = M.nearest(G, ans.reshape(36), q)
            row["wrong_cells"] = int((ans.reshape(36) != np.array(t["expected"]).reshape(36)).sum())
            row["error_type"] = ("execution slip" if t["group_id"] in nb["tied_ids"] and nb["best_cells"] >= 30
                                 else "other wrong grid")
        if e == "ops":
            ops = parse_ops(r.get("content"))
            row["ops_stated"] = ops if ops is not None else "unparsed"
            ok = ops is not None and len(ops) > 0
            row["ops_equivalent"] = bool(ok and G.word_id(ops) == t["group_id"])
            row["ops_literal"] = bool(ok and ops == t["word"])
            row["ops_consistent_with_answer"] = bool(ok and ans is not None and
                                                     np.array_equal(q[M.word_perm(ops)], ans.reshape(36)))
        if e == "steps":
            st = parse_steps(r.get("content"))
            row["n_steps"] = len(st)
            row["steps_text"] = " | ".join(d for d, _ in st)
        if e == "steps" and any(g is not None for _, g in st):      # only if the model also wrote intermediate grids
            prev, words, legal = q, [], True
            for _, g in st:                       # what each step does to the grid produced by the previous step
                ids = [] if g is None else [i for i in range(G.size) if np.array_equal(prev[G.perms[i]], g.reshape(36))]
                if not ids:
                    legal = False
                    words.append("?")
                    if g is None:
                        break
                    prev = g.reshape(36)
                    continue
                words.append(SW[ids[0]] or "I")
                prev = g.reshape(36)
            row["step_words"] = " | ".join(words)
            row["steps_all_legal"] = bool(st) and legal
            row["steps_reach_answer"] = bool(st) and ans is not None and np.array_equal(prev, ans.reshape(36))
            row["steps_correct_chain"] = bool(row["steps_all_legal"] and
                                              np.array_equal(prev, np.array(t["expected"]).reshape(36)))
        if e == "describe":
            rule = parse_rule(r.get("content"))
            row["rule"] = rule if rule is not None else "unparsed"
            row["rule_keywords"] = rule_keywords(rule)
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
    if (df.experiment == "describe").any():
        d = df[df.experiment == "describe"]
        out["rule_keywords"] = (d.groupby(["word", "rule_keywords"]).size().rename("count").reset_index()
                                .sort_values(["word", "count"], ascending=[True, False]))
    cost = df[["prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_usd"]].apply(
        pd.to_numeric, errors="coerce").sum().to_frame("total").T
    out["usage"] = cost
    return out


# ---------------------------------------------------------------------------------------------------------------------
# notebook 47: number-of-demos sweep
# ---------------------------------------------------------------------------------------------------------------------
def sweep_paths(out_dir, demo_counts, replicates):
    return {(r, n): Path(out_dir) / f"rep{r}_demos{n}.jsonl" for r in replicates for n in demo_counts}


def score_sweep(base_tasks_by_rep: Dict[int, List[dict]], out_dir, demo_counts, replicates) -> "pd.DataFrame":
    import pandas as pd
    parts = []
    for (r, n), path in sweep_paths(out_dir, demo_counts, replicates).items():
        if path.exists():
            d = score(with_demos(base_tasks_by_rep[r], n), path)
            if len(d):
                parts.append(d.assign(replicate=r, n_demos=n))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def plot_sweep(df, out_dir=None):
    """Accuracy vs number of demos (mean over transformations; one point per replicate), error types vs demos,
    and the stated-operations metrics vs demos. Returns the figure paths."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    colors = {"answer": "#0072B2", "ops": "#D55E00", "describe": "#D55E00"}
    labels = {"answer": "answer only", "ops": "state operations first", "describe": "describe the rule first"}
    exps = [e for e in ("answer", "ops", "describe") if (df.experiment == e).any()]
    paths = []
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for e in exps:
        q = df[df.experiment == e]
        per_rep = q.groupby(["n_demos", "replicate"]).correct.mean().reset_index()
        mean = per_rep.groupby("n_demos").correct.mean()
        ax.plot(mean.index, mean.values, marker="o", lw=2, color=colors[e], label=labels[e])
        ax.scatter(per_rep.n_demos, per_rep.correct, color=colors[e], s=14, alpha=0.5)
    ax.set(xlabel="number of demonstrations", ylabel="exact accuracy", ylim=(0, 1.02),
           xticks=sorted(df.n_demos.unique()))
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("k = 2: accuracy vs demonstrations (line = mean, dots = replicates)", fontsize=9.5)
    fig.tight_layout()
    if out_dir:
        paths.append(Path(out_dir) / "acc_vs_demos.png"); fig.savefig(paths[-1], dpi=150)
    plt.show()
    order = ["correct", "execution slip", "wrong transform", "other wrong grid", "unparsed"]
    cols = {"correct": "#009E73", "execution slip": "#E69F00", "wrong transform": "#D55E00",
            "other wrong grid": "#7a7a7a", "unparsed": "#cccccc"}
    fig, axes = plt.subplots(1, len(exps), figsize=(5.5 * len(exps), 4), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, e in zip(axes, exps):
        q = df[df.experiment == e]
        tab = q.groupby("n_demos").error_type.value_counts(normalize=True).unstack(fill_value=0)
        bottom = np.zeros(len(tab))
        xs = np.arange(len(tab))
        for k in order:
            if k in tab:
                ax.bar(xs, tab[k].values, bottom=bottom, color=cols[k], label=k, edgecolor="white", lw=1)
                bottom += tab[k].values
        ax.set_xticks(xs, tab.index.astype(str))
        ax.set(xlabel="number of demonstrations", title=labels[e])
    axes[0].set_ylabel("share of answers")
    axes[-1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    if out_dir:
        paths.append(Path(out_dir) / "error_types_vs_demos.png"); fig.savefig(paths[-1], dpi=150)
    plt.show()
    o = df[df.experiment == "ops"]
    if len(o):
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        for c, col, lab in (("ops_equivalent", "#D55E00", "stated operations correct"),
                            ("correct", "#0072B2", "answer grid correct"),
                            ("ops_consistent_with_answer", "#7a7a7a", "stated operations match own answer")):
            m = o.groupby("n_demos")[c].mean()
            ax.plot(m.index, m.values, marker="o", lw=2, color=col, label=lab)
        ax.set(xlabel="number of demonstrations", ylabel="share", ylim=(0, 1.02), xticks=sorted(o.n_demos.unique()))
        ax.grid(axis="y", color="#e6e6e6", lw=0.8)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title("'state operations first': inference vs execution", fontsize=9.5)
        fig.tight_layout()
        if out_dir:
            paths.append(Path(out_dir) / "ops_vs_demos.png"); fig.savefig(paths[-1], dpi=150)
        plt.show()
    return paths
