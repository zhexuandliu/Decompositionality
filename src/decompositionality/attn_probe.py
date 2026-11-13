"""Notebook 38 (read-only): where do the trained models attend among the operation tokens?

Hypothesis (b) from notebook 37: with a short training support in S = 7, the model organises the computation around
the FIRST and the LAST real (non-<identity>) operation. A singleton then has first = last and is executed twice;
a longer held-out instruction loses its middle operation(s).

For each model and condition we load the LAST checkpoint (state.pt) and run teacher-forced forward passes on fresh
records: singletons at every slot, the trained lengths, and the first held-out longer k. Every attention head's
weights are recomputed exactly (softmax(QK^T/sqrt(d)) with the model's mask). For the attention that a query position
pays to the operation tokens we report:
  * the total mass on the operation tokens (all 7 slots, incl. <identity>);
  * how that op-token mass splits over the roles FIRST real op / MIDDLE real op(s) / LAST real op / <identity>
    (for a singleton FIRST = LAST = the one operation, reported as 'first');
separately for the answer-producing positions and for the input-cell positions, per layer, and per head.
"""
from __future__ import annotations

import math
import types

import numpy as np

from . import idslot as IS
from . import idvar as V
from . import longk as LK
from . import longk_exp as LX
from .data import make_rng, random_grids
from .words import words_of_length

SUBDIR = "S7_support_sweep_20k_v1"
CONDITIONS = ["S7_k02", "S7_k0234", "S7_k02345"]
MODELS = ["ar_slotpos_digithead_target", "query_ar", "causal_encoder", "transformer_ar"]
LABELS = {"transformer_ar": "original AR", "ar_slotpos_digithead_target": "AR34", "query_ar": "query AR",
          "causal_encoder": "causal encoder"}
STREAM_PROBE = 1381


def _records(e, kind, n, seed=0):
    """(slots [n,S], x [n,6,6], y, words) for kind in {'single', ('k', k)}."""
    S, _ = LK.CONDITIONS[e]
    rng = make_rng(seed, STREAM_PROBE, S, 0 if kind == "single" else kind[1])
    rows = []
    if kind == "single":
        for op in "ABC":
            for p in range(S):
                rows += [(op, (p,))] * max(1, n // (3 * S))
    else:
        k = kind[1]
        ws = words_of_length(k)
        for i in range(n):
            rows.append((ws[int(rng.integers(len(ws)))], IS.subset(rng, S, k)))
    words = [r[0] for r in rows]
    slots = np.stack([IS.slots_for(w, pos, S) for w, pos in rows])
    x = random_grids(rng, len(rows))
    return slots, x, V.apply_words(x, words), words


def _patch(model, store):
    from .models.common import MultiHeadSelfAttention
    import torch
    import torch.nn.functional as F

    def fwd(self, h, attn_mask, cos=None, sin=None):
        B, L, D = h.shape
        q, k, v = self.qkv(h).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if cos is not None:
            from .models.common import apply_rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
        s = s.masked_fill(~attn_mask.expand_as(s), float("-inf"))
        w = torch.softmax(s, dim=-1)
        store.append(w.detach().cpu())
        out = (w.to(v.dtype) @ v)
        return self.proj(out.transpose(1, 2).reshape(B, L, D))
    for mod in model.modules():
        if isinstance(mod, MultiHeadSelfAttention):
            mod.forward = types.MethodType(fwd, mod)


def _positions(model, n_ops):
    """(op token idx, input-cell idx, answer-producing idx) in the sequence the model attends over."""
    if model.kind == "ar":
        lay = model.layout_for(n_ops)
        return (np.array(lay.idx_slots), np.arange(lay.in_start, lay.in_start + 36),
                np.arange(lay.idx_output, lay.idx_output + 36))
    return np.arange(n_ops), np.arange(n_ops, n_ops + 36), np.arange(n_ops + 36, n_ops + 72)


def _forward(model, batch):
    if model.kind == "ar":
        model._teacher_forced(batch)
    else:
        model.logits(batch, model._teacher_prev(batch) if getattr(model, "feedback", False) else None)


def _roles(slots):
    """[n, S] role per op slot: 0 first real op, 1 middle, 2 last, 3 <identity>."""
    real = slots != V.ID_OP
    r = np.full(slots.shape, 3)
    for i in range(len(slots)):
        idx = np.flatnonzero(real[i])
        if len(idx):
            r[i, idx] = 1
            r[i, idx[-1]] = 2
            r[i, idx[0]] = 0          # singleton: first (= last)
    return r


def probe(results_root, mode="full", seed=42, conditions=None, models=None, n_records=210, batch=105,
          device="cpu", show=True, log=print, review_name="Attention_Probe_Review.ipynb"):
    import pandas as pd
    import torch
    from .models.registry import build_model
    from .mirror_report import Report, write_review
    from .training import run_dir
    conditions = list(conditions or CONDITIONS)
    models = list(models or MODELS)
    R = Report(show)
    out = LX.family_root(results_root, mode, SUBDIR) / "report" / f"seed_{seed}_attention_probe"
    R.md("# Attention to the operation tokens (LAST checkpoint, teacher forcing, fresh records)\n"
         "Roles: FIRST / MIDDLE / LAST real operation and `<identity>` fillers (a singleton's one operation counts as "
         "FIRST). 'op mass' = the share of a query position's attention that lands on the 7 operation tokens; the role "
         "shares split that op mass; 'uniform ...' = the share each role would get if attention were spread evenly over the 7 "
         "slots (the baseline to compare with). Averaged over heads and query positions (layer table) or over query positions "
         "(head table).")
    layer_rows, head_rows, status = [], [], []
    for e in conditions:
        S, ks = LK.CONDITIONS[e]
        kinds = ["single", ("k", 2), ("k", max(ks))] + [("k", max(ks) + 1)] if max(ks) < S else ["single", ("k", 2)]
        kinds = list(dict.fromkeys(kinds))
        for m in models:
            cfg = LX.make_cfg(e, results_root, mode, [m], seed, SUBDIR)
            f = run_dir(cfg, m) / "state.pt"
            if not f.exists():
                status.append({"condition": e, "model": LABELS[m], "status": "no state.pt"})
                continue
            st = torch.load(f, map_location="cpu", weights_only=False)
            model = build_model(m, LK.model_config(m)).to(device)
            model.load_state_dict(st["model"])
            model.eval()
            store = []
            _patch(model, store)
            status.append({"condition": e, "model": LABELS[m], "status": f"ok (step {st.get('global_step')})"})
            for kind in kinds:
                slots, x, y, words = _records(e, kind, n_records)
                tag = "singleton" if kind == "single" else (f"k={kind[1]} (trained)" if kind[1] in ks else f"k={kind[1]} (held out)")
                roles = _roles(slots)
                base = np.stack([(roles == r).mean() for r in range(4)])     # share under uniform attention over slots
                op_i, in_i, out_i = _positions(model, S)
                acc = {}     # (group, layer, head) -> [op mass, role masses x4] summed over records & queries
                cnt = {}
                for b0 in range(0, len(slots), batch):
                    sl = slice(b0, b0 + batch)
                    bt = {"ops": torch.as_tensor(slots[sl], device=device), "x": torch.as_tensor(x[sl].astype(np.int64), device=device),
                          "y": torch.as_tensor(y[sl].astype(np.int64), device=device)}
                    store.clear()
                    with torch.no_grad():
                        _forward(model, bt)
                    rl = torch.as_tensor(roles[sl])
                    oh = torch.nn.functional.one_hot(rl, 4).float()                    # [b, S, 4]
                    for li, w in enumerate(store):                                     # w [b, H, L, L]
                        for gname, qi in (("answer positions", out_i), ("input cells", in_i)):
                            a = w[:, :, qi][:, :, :, op_i]                             # [b, H, 36, S]
                            mass = a.sum(-1).mean(2)                                   # [b, H]
                            role = torch.einsum("bhqs,bsr->bhr", a, oh) / len(qi)      # [b, H, 4] (absolute mass)
                            for h in range(a.shape[1]):
                                key = (gname, li, h)
                                v = acc.setdefault(key, np.zeros(5))
                                v[0] += mass[:, h].sum().item()
                                v[1:] += role[:, h].sum(0).numpy()
                                cnt[key] = cnt.get(key, 0) + a.shape[0]
                for (g, li, h), v in acc.items():
                    n = cnt[(g, li, h)]
                    tot = max(v[1:].sum(), 1e-12)
                    head_rows.append({"condition": e, "model": LABELS[m], "records": tag, "query": g, "layer": li + 1,
                                      "head": h + 1, "op mass": v[0] / n, "FIRST": v[1] / tot, "MIDDLE": v[2] / tot,
                                      "LAST": v[3] / tot, "<identity>": v[4] / tot,
                                      "uniform FIRST": base[0], "uniform MIDDLE": base[1], "uniform LAST": base[2],
                                      "uniform <identity>": base[3]})
    H = pd.DataFrame(head_rows)
    R.table("Availability", pd.DataFrame(status), 40)
    if not len(H):
        return None
    L = (H.assign(w=H["op mass"]).groupby(["condition", "model", "records", "query", "layer"], sort=False)
         .apply(lambda d: pd.Series({"op mass": d["op mass"].mean(),
                                     **{r: np.average(d[r], weights=d.w + 1e-12) for r in ("FIRST", "MIDDLE", "LAST", "<identity>")},
                                     **{f"uniform {r}": d[f"uniform {r}"].iloc[0] for r in ("FIRST", "MIDDLE", "LAST", "<identity>")}}))
         .reset_index())
    R.table("1. Per layer (heads weighted by their op mass): op mass and its split over roles", L.round(3), 600)
    top = (H[H["records"].str.contains("trained") & (H["op mass"] >= 0.1)]
           .assign(**{"FIRST - LAST": lambda d: d.FIRST - d.LAST}).sort_values(["condition", "model", "query", "layer", "head"]))
    R.table("2. Heads that put >= 10% of their attention on the operation tokens (trained-length records)",
            top.round(3), 600)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    H.to_csv(out / "tables" / "per_head.csv", index=False)
    L.to_csv(out / "tables" / "per_layer.csv", index=False)
    rp = write_review(R.items, out, "Attention probe", "# Attention probe — review snapshot\n\nBaked outputs; nothing to run.",
                      review_name)
    log(f"REVIEW FILE: {rp}")
    return rp
