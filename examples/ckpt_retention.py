"""Which checkpoints a run keeps (pure Python, no JAX): the newest `keep`, plus every `every`-th learner step up to
`until`, so a run that trains long still has its early checkpoints for success-at-a-budget curves
(docs/final_study.md). train_aqi_rlpd.py saves with flax's own `keep` set out of reach and prunes with
`prune_checkpoints` after each save; with every == 0 nothing here runs and flax's `keep` applies as before."""
import os
import re
import shutil

_CKPT = re.compile(r"^checkpoint_(\d+)$")


def checkpoint_steps(ckpt_dir):
    """Learner steps of the finished checkpoints in `ckpt_dir` (folders named exactly checkpoint_<step>; temporary
    folders of a save in progress have longer names and are left alone)."""
    if not os.path.isdir(ckpt_dir):
        return []
    out = []
    for name in os.listdir(ckpt_dir):
        m = _CKPT.match(name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def steps_to_keep(steps, keep, every=0, until=0):
    """The steps to keep: the newest `keep`, plus every multiple of `every` that is <= `until` (until 0 = no limit)."""
    steps = sorted(steps)
    kept = set(steps[-keep:]) if keep > 0 else set(steps)
    if every > 0:
        kept |= {s for s in steps if s % every == 0 and (until <= 0 or s <= until)}
    return kept


def prune_checkpoints(ckpt_dir, keep, every=0, until=0, log=None):
    """Delete the checkpoint folders `steps_to_keep` does not keep. Returns the deleted steps."""
    steps = checkpoint_steps(ckpt_dir)
    kept = steps_to_keep(steps, keep, every, until)
    gone = []
    for s in steps:
        if s in kept:
            continue
        shutil.rmtree(os.path.join(ckpt_dir, f"checkpoint_{s}"), ignore_errors=True)
        gone.append(s)
    if gone and log:
        log(f"checkpoints pruned: {gone} (keeping the newest {keep}"
            + (f" and every {every} up to {until or 'the end'}" if every else "") + ")")
    return gone
