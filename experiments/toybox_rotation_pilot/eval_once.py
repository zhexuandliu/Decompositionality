"""Make exactly ONE evaluation call to GPT-6 Luna on the Toybox rotation question.

Usage:  OPENROUTER_API_KEY=... python eval_once.py [--out results]

* Sends the 11 images as ordered image inputs, each preceded by a neutral text label, followed by the prompt.
  (Format recorded as "ordered_images_with_labels"; the composite is NOT sent.)
* Reads only question/inputs.json and question/images/ -- never metadata.json -- when building the request.
* Refuses to run if a response file already exists, so the call is made at most once. A failed call is saved
  and is not retried.
* The key is read from the environment and never printed or saved.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "openai/gpt-6-luna"          # the identifier used throughout this repository (notebooks 46-52)
URL = "https://openrouter.ai/api/v1/chat/completions"


def build_request(qdir: Path) -> dict:
    spec = json.loads((qdir / "inputs.json").read_text())
    content = []
    for item in spec["images_in_order"]:
        b64 = base64.b64encode((qdir / item["file"]).read_bytes()).decode()
        content.append({"type": "text", "text": item["label"] + ":"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": spec["prompt"]})
    # default inference settings: no temperature / reasoning / max_tokens overrides
    return {"model": MODEL, "messages": [{"role": "user", "content": content}], "usage": {"include": True}}


def parse_choice(text):
    if not text:
        return None
    t = text.strip()
    if re.fullmatch(r"[ABCD]\.?", t):
        return t[0]
    m = re.findall(r"\b([ABCD])\b", t)
    return m[-1] if len(set(m)) == 1 else None      # ambiguous if several different letters appear


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE / "results", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    resp_path = a.out / "response.json"
    if resp_path.exists():
        raise SystemExit(f"{resp_path} already exists: the single evaluation call has been made. Not calling again.")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set; no call made.")

    import requests
    body = build_request(HERE / "question")
    record = {"timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "model_requested": MODEL,
              "endpoint": URL, "submission_format": "ordered_images_with_labels",
              "inference_settings": "provider defaults (no temperature, reasoning or max_tokens set)",
              "prompt": body["messages"][0]["content"][-1]["text"],
              "image_references": [c["text"][:-1] for c in body["messages"][0]["content"] if c["type"] == "text"][:-1]}
    try:
        r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=900)
        record["http_status"] = r.status_code
        d = r.json()
        if r.status_code != 200 or "error" in d:
            record["error"] = d.get("error", r.text[:2000])
        else:
            msg = d["choices"][0]["message"]
            record.update({"model_returned": d.get("model"), "raw_response": msg.get("content"),
                           "reasoning_summary": msg.get("reasoning"), "finish_reason": d["choices"][0].get("finish_reason"),
                           "usage": d.get("usage"), "error": None})
    except Exception as ex:                                  # network error etc.: saved, not retried
        record["error"] = f"{type(ex).__name__}: {ex}"
    resp_path.write_text(json.dumps(record, indent=2))

    # scoring happens only AFTER the call, against the separate metadata file
    meta = json.loads((HERE / "metadata.json").read_text())
    choice = parse_choice(record.get("raw_response"))
    summary = {"model": MODEL, "raw_response": record.get("raw_response"), "parsed_choice": choice,
               "intended_answer": meta["intended_answer"],
               "matches_intended": (choice == meta["intended_answer"]) if choice else None,
               "valid_output": choice is not None, "error": record.get("error"),
               "chosen_candidate_intended_rotation_deg_ccw": next(
                   (c["intended_rotation_deg_ccw"] for c in meta["candidates_in_presented_order"] if c["letter"] == choice), None)}
    (a.out / "result_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
