"""Training loop shared by the S1 arms (backend-neutral: imports no concrete backend).

An example is a dict with at least 'prompt' and 'target' (the hold-out guard's fields: kind,
statement, source_ids, source). Every example passes the guard callable before it is used (R4);
the trainer refuses to run without one.
"""
from __future__ import annotations

import math
import random
import time

import torch

from .model import TinyLM, Tokenizer, encode_pair, lm_loss, make_batch


def guarded(examples, guard, stats=None):
    if guard is None:
        raise ValueError("R4: training data must pass through the hold-out guard")
    out = []
    for ex in examples:
        guard(ex)                          # raises LeakError on any leak
        out.append(ex)
    if stats is not None:
        stats["guard_checked"] = stats.get("guard_checked", 0) + len(out)
    return out


def train_steps(model: TinyLM, tok: Tokenizer, params, examples, *, steps: int, batch: int, lr: float,
                seed: int, guard, weight_decay: float = 0.0, warmup: int = 20, log_every: int = 0,
                stats=None):
    """Cross-entropy on target tokens for 'steps' optimizer steps over shuffled epochs of 'examples'.
    Returns a dict of training statistics (steps, examples seen, tokens, mean loss of the last epoch)."""
    stats = {} if stats is None else stats
    examples = guarded(examples, guard, stats)
    if not examples or steps <= 0:
        stats.update(steps=0, examples_seen=0, train_tokens=0, last_loss=None)
        return stats
    enc = [encode_pair(tok, ex["prompt"], ex["target"], model.cfg["max_len"]) for ex in examples]
    rng = random.Random(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    sched = lambda s: min(1.0, (s + 1) / max(1, warmup)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
    model.train()
    order, k = [], 0
    seen = toks = 0
    losses = []
    t0 = time.time()
    lens = [len(p) + len(t) for p, t in enc]
    for s in range(steps):
        if k + batch > len(order):
            perm = list(range(len(enc)))
            rng.shuffle(perm)
            # length bucketing: sort within chunks of 8 batches, then shuffle the batches
            chunks = []
            for c0 in range(0, len(perm), 8 * batch):
                ch = sorted(perm[c0:c0 + 8 * batch], key=lambda i: lens[i])
                chunks.extend(ch[b0:b0 + batch] for b0 in range(0, len(ch), batch))
            rng.shuffle(chunks)
            order = order[k:] + [i for ch in chunks for i in ch]
            k = 0
        idx = order[k:k + batch]
        k += batch
        ids, lm = make_batch([enc[i] for i in idx])
        for g in opt.param_groups:
            g["lr"] = lr * sched(s)
        loss = lm_loss(model, ids, lm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        seen += len(idx)
        toks += int(lm.sum().item())
        losses.append(float(loss.item()))
        if log_every and (s + 1) % log_every == 0:
            print(f"  step {s + 1}/{steps} loss {sum(losses[-log_every:]) / log_every:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    tail = losses[-max(1, min(len(losses), len(enc) // max(1, batch) + 1)):]
    stats.update(steps=steps, examples_seen=seen, train_tokens=toks, last_loss=sum(tail) / len(tail),
                 wall_s=round(time.time() - t0, 2))
    return stats


def epochs_to_steps(n_examples: int, epochs: float, batch: int, min_steps: int = 1) -> int:
    return max(min_steps, int(math.ceil(n_examples * epochs / batch)))
