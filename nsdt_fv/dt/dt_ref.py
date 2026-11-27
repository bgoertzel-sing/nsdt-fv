"""DT-ref (plan Appendix A.4): the stand-in for the DT core, on the Stage S tiny transformer.

    The base is frozen. The cap is a set of residual adapter blocks on the top third of the layers
    together with a trainable output head.

CapLM wraps a frozen TinyLM (train/model.py). After each base block whose index is in 'layers' a
residual adapter x <- x + Up(GELU(Down(LN(x)))) is applied (Up initialised to zero, so the capped
model starts equal to the base); the output head is a new untied linear map initialised from the
base's (tied) embedding matrix. Only adapters and head are trainable; base parameters have
requires_grad False, and the base hash (train/model.base_state_hash over the wrapped TinyLM) is
checked at every stage boundary by the S2 driver (T8).

DTRef implements the DTCore protocol of A.3 (wake, learn, generate, state_hash). It keeps no
training examples between calls (assumption (e) of 3.1): wake() and learn() take their examples as
arguments and drop them on return.
"""
from __future__ import annotations

import hashlib
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..train.model import Block, TinyLM, Tokenizer, base_state_hash, encode_pair, lm_loss, make_batch, sample


class Adapter(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.down = nn.Linear(d, r)
        self.up = nn.Linear(r, d)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(self.ln(x))))


class BlockAdapter(nn.Module):
    """A residual adapter block with attention: a full transformer block (as in the base) whose output
    projections start at zero, so that it is the identity at initialisation (the sized-up cap)."""

    def __init__(self, d, heads, ff, drop):
        super().__init__()
        self.block = Block(d, heads, ff, drop)
        for lin in (self.block.qkv.base, self.block.proj.base, self.block.fc1.base, self.block.fc2.base):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)
        nn.init.zeros_(self.block.proj.base.weight)
        nn.init.zeros_(self.block.fc2.base.weight)

    def forward(self, x, mask, cache=None):
        return self.block(x, mask, cache)


def top_third(n_layers: int) -> list:
    """Indices of the top third of the layers, rounded up (4 layers -> the top 2)."""
    k = max(1, math.ceil(n_layers / 3))
    return list(range(n_layers - k, n_layers))


class CapLM(nn.Module):
    def __init__(self, base: TinyLM, r: int, layers=None, seed: int = 0, kind: str = "mlp"):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.cfg = base.cfg
        self.layers = list(layers) if layers is not None else top_third(len(base.blocks))
        torch.manual_seed(seed)
        d, V = base.cfg["d"], base.cfg["vocab"]
        self.kind = kind
        if kind == "mlp":
            self.adapters = nn.ModuleDict({str(i): Adapter(d, r) for i in self.layers})
        elif kind == "block":            # r is the feed-forward width of each adapter block
            self.adapters = nn.ModuleDict({str(i): BlockAdapter(d, base.cfg["heads"], r, base.cfg["drop"])
                                           for i in self.layers})
        else:
            raise ValueError(kind)
        self.head = nn.Linear(d, V, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(base.head.weight)

    @property
    def blocks(self):                     # train/model.sample() sizes its KV caches from len(blocks)
        extra = list(self.adapters.values()) if self.kind == "block" else []
        return list(self.base.blocks) + extra

    def cap_parameters(self):
        return list(self.adapters.parameters()) + list(self.head.parameters())

    def n_params(self, trainable_only=False):
        n = sum(p.numel() for p in self.cap_parameters())
        return n if trainable_only else n + self.base.n_params()

    def forward(self, ids, pos, mask, caches=None):
        b = self.base
        x = F.dropout(b.tok(ids) + b.pos(pos), b.p, self.training)
        nb = len(b.blocks)
        for i, blk in enumerate(b.blocks):
            x = blk(x, mask, None if caches is None else caches[i])
            if str(i) in self.adapters:
                if self.kind == "block":
                    a = self.layers.index(i)
                    x = self.adapters[str(i)](x, mask, None if caches is None else caches[nb + a])
                else:
                    x = self.adapters[str(i)](x)
        return self.head(b.ln(x))


def cap_hash(model: CapLM) -> str:
    h = hashlib.sha256()
    for p in model.cap_parameters():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def seq_logprob(model, ids, lmask):
    """Per-row sum of log-probabilities of the target tokens (lmask) under the model."""
    B, T = ids.shape
    pos = torch.arange(T).unsqueeze(0).expand(B, T)
    causal = torch.ones(T - 1, T - 1, dtype=torch.bool).tril()
    logits = model(ids[:, :-1], pos[:, :-1], causal)
    lp = torch.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (lp * lmask[:, 1:]).sum(dim=1)


class DTRef:
    """DTCore (A.3) over a CapLM. Every example passes the hold-out guard callable before use (R4)."""

    def __init__(self, model: CapLM, tok: Tokenizer, cfg: dict, seed: int, guard):
        if guard is None:
            raise ValueError("R4: training data must pass through the hold-out guard")
        self.model, self.tok, self.cfg, self.seed, self.guard = model, tok, cfg, seed, guard
        self.params = model.cap_parameters()
        self.opt = torch.optim.AdamW(self.params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0),
                                     betas=(0.9, 0.98))
        self.steps = 0
        self.tokens = 0

    def _enc(self, ex):
        return encode_pair(self.tok, ex["prompt"], ex["target"], self.model.cfg["max_len"])

    def _step(self, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.opt.step()
        self.steps += 1

    # ------------------------------------------------------------------ DTCore
    def wake(self, examples, seed, batch=1):
        """One epoch of cross-entropy on the window's examples (A.4), shuffled, 'batch' per step."""
        for ex in examples:
            self.guard(ex)
        enc = [self._enc(ex) for ex in examples]
        order = list(range(len(enc)))
        random.Random(seed).shuffle(order)
        torch.manual_seed(seed)
        self.model.train()
        losses, steps = [], 0
        for b0 in range(0, len(order), batch):
            ids, lm = make_batch([enc[i] for i in order[b0:b0 + batch]])
            loss = lm_loss(self.model, ids, lm)
            self._step(loss)
            self.tokens += int(lm.sum().item())
            losses.append(float(loss.item()))
            steps += 1
        self.model.eval()
        return {"steps": steps, "examples": len(enc), "mean_loss": sum(losses) / max(1, len(losses))}

    def learn(self, accepted, rejected=None, seed=0, pref_weight=0.1):
        """One cross-entropy step per example in 'accepted' (A.4). 'rejected' maps an index of
        'accepted' to a rejected example of the same statement; for those the step adds the
        preference loss pref_weight * softplus(-(logp(accepted) - logp(rejected))) (negative phase)."""
        rejected = rejected or {}
        for ex in accepted:
            self.guard(ex)
        for ex in rejected.values():
            self.guard(ex)
        order = list(range(len(accepted)))
        random.Random(seed).shuffle(order)
        torch.manual_seed(seed)
        self.model.train()
        ce_sum, pref_sum, n_pref = 0.0, 0.0, 0
        for i in order:
            a = self._enc(accepted[i])
            if i in rejected:
                r = self._enc(rejected[i])
                ids, lm = make_batch([a, r])
                lp = seq_logprob(self.model, ids, lm)
                ce = -lp[0] / lm[0, 1:].sum().clamp_min(1.0)
                pref = F.softplus(-(lp[0] - lp[1]))
                loss = ce + pref_weight * pref
                pref_sum += float(pref.item())
                n_pref += 1
                self.tokens += int(lm[0].sum().item())
            else:
                ids, lm = make_batch([a])
                ce = lm_loss(self.model, ids, lm)
                loss = ce
                self.tokens += int(lm.sum().item())
            self._step(loss)
            ce_sum += float(ce.item())
        self.model.eval()
        return {"steps": len(order), "pref_pairs": n_pref, "mean_ce": ce_sum / max(1, len(order)),
                "mean_pref": pref_sum / max(1, n_pref) if n_pref else None}

    def generate(self, prompts, seeds, temperature, max_new):
        return sample(self.model, self.tok, prompts, seeds, temperature=temperature, max_new=max_new)

    def state_hash(self):
        return cap_hash(self.model)

    def base_hash(self):
        return base_state_hash(self.model.base)
