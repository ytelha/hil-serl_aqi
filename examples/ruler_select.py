"""Which saved checkpoints the ruler scores, and how a cell's score is picked from them (pure Python, no JAX).

Shared by eval_ruler.py, which scores the kept checkpoints, and ruler_reselect.py, which recomputes a finished cell's
score from its trials.jsonl. Both therefore keep the same checkpoints by construction.

With --select best / mean / best_window the ruler scores every saved checkpoint (one per 1000 learner steps). Most
of its time goes into early checkpoints that fail every trial, and a failed trial runs to the step limit. Three
optional shortenings, all off by default (then every saved checkpoint is scored, as before):

    min_step  score only checkpoints saved after this learner step. The campaign runner sets it per arm to the
              arm's demo pre-training steps (eval.skip_pretrain), so policies from before any online training are
              not scored. If nothing is left (a cell stopped during pre-training), the newest is scored.
    stride    every n-th checkpoint, counted back from the newest, so the newest is always scored. This changes the
              score: a window of k scored checkpoints then spans k*stride saves (docs/eval_ruler.md, measured).
    newest    only the newest N (applied last).

The best window is the k consecutive kept checkpoints with the highest mean success rate; a tie goes to the earliest
window, exactly as in eval_ruler.py.
"""

FILTER_KEYS = ("min_step", "stride", "newest")


def normalise(min_step=0, stride=1, newest=0):
    """(min_step, stride, newest) as ints, with off = (0, 1, 0)."""
    return max(int(min_step or 0), 0), max(int(stride or 1), 1), max(int(newest or 0), 0)


def is_active(min_step=0, stride=1, newest=0):
    return normalise(min_step, stride, newest) != (0, 1, 0)


def describe(min_step=0, stride=1, newest=0):
    """The record written to cell.json as ``checkpoint_filter``; None when off (cell.json then has no such key)."""
    m, s, n = normalise(min_step, stride, newest)
    if (m, s, n) == (0, 1, 0):
        return None
    return {"min_step": m, "stride": s, "newest": n}


def keep_checkpoints(steps, min_step=0, stride=1, newest=0):
    """The checkpoint steps to score, ascending."""
    m, s, n = normalise(min_step, stride, newest)
    all_steps = sorted({int(x) for x in steps})
    if not all_steps:
        return []
    kept = [x for x in all_steps if x > m] if m else list(all_steps)
    if not kept:
        kept = all_steps[-1:]
    if s > 1:
        kept = kept[::-1][::s][::-1]
    if n:
        kept = kept[-n:]
    return kept


def _valid(points):
    return sorted((p for p in points if p.get("sr") is not None and p["sr"] == p["sr"]), key=lambda p: p["step"])


def best_window(points, k):
    """(mean SR, [steps]) of the k consecutive checkpoints with the highest mean SR; (None, None) without points.
    ``points`` is [{"step", "sr"}]; NaN success rates are skipped."""
    vs = _valid(points)
    if not vs:
        return None, None
    w = min(max(int(k), 1), len(vs))
    best_sr, best_steps = None, None
    top = -1.0
    for i in range(0, len(vs) - w + 1):
        win = vs[i:i + w]
        m = float(sum(p["sr"] for p in win) / w)
        if m > top:
            top = m
            best_sr, best_steps = m, [p["step"] for p in win]
    return best_sr, best_steps


def best_single(points):
    """(SR, step) of the best checkpoint; the earliest wins a tie."""
    vs = _valid(points)
    if not vs:
        return None, None
    bp = max(vs, key=lambda p: p["sr"])
    return float(bp["sr"]), bp["step"]


def mean_all(points):
    vs = _valid(points)
    return float(sum(p["sr"] for p in vs) / len(vs)) if vs else None
