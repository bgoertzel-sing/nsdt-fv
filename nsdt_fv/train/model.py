"""Tiny decoder-only transformer, tokenizer and seeded sampler for Stage S (plan 8.1).

Backend-neutral: this module knows nothing about ToyEq or Lean. The tokenizer is configured with a
keyword list (the S1 driver passes the ToyEq keywords); every other identifier is spelled out
character by character, so project-local names never need new vocabulary and a frozen embedding
table covers every project. A Lean backend would pass its own keyword list (gate G0).

Models are trained from scratch (no pretrained weights) and kept under 5M parameters (plan 8.1).
"""
from __future__ import annotations

import hashlib
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, SEP, EOS, UNK, SPACE = 0, 1, 2, 3, 4, 5
SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>", "<unk>", " "]
CHARS = [chr(c) for c in range(33, 127)]           # printable ASCII except space
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class Tokenizer:
    def __init__(self, keywords):
        self.keywords = sorted(set(keywords), key=lambda k: (-len(k), k))
        self.itos = SPECIALS + list(self.keywords) + CHARS
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.kwset = set(self.keywords)
        self.kw_ids = {self.stoi[k] for k in self.keywords}
        self.word_kw_ids = {self.stoi[k] for k in self.keywords if _WORD.fullmatch(k)}
        # multi-character symbolic keywords (e.g. 'rw<-') are matched before words
        self._sym_kw = [k for k in self.keywords if not _WORD.fullmatch(k)]

    @property
    def vocab_size(self):
        return len(self.itos)

    def _pieces(self, text):
        """Split text into (kind, string) with kind in kw|word|sym|space."""
        out = []
        i, n = 0, len(text)
        while i < n:
            c = text[i]
            if c.isspace():
                while i < n and text[i].isspace():
                    i += 1
                out.append(("space", " "))
                continue
            kw = next((k for k in self._sym_kw if text.startswith(k, i)), None)
            if kw is not None:
                out.append(("kw", kw))
                i += len(kw)
                continue
            m = _WORD.match(text, i)
            if m:
                w = m.group(0)
                out.append(("kw" if w in self.kwset else "word", w))
                i = m.end()
                continue
            out.append(("sym", c))
            i += 1
        return out

    def encode(self, text):
        """Spaces are kept only between two word-like pieces (keyword or identifier)."""
        ps = self._pieces(text)
        ids = []
        for k, (kind, s) in enumerate(ps):
            if kind == "space":
                prev = ps[k - 1][0] if k > 0 else None
                nxt = ps[k + 1][0] if k + 1 < len(ps) else None
                if prev in ("kw", "word") and nxt in ("kw", "word"):
                    ids.append(SPACE)
            elif kind == "kw":
                ids.append(self.stoi[s])
            else:
                ids.extend(self.stoi.get(ch, UNK) for ch in s)
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            if i in (PAD, BOS, SEP, EOS):
                continue
            if i == UNK or not (0 <= i < len(self.itos)):
                out.append("?")
                continue
            s = self.itos[i]
            if i in self.kw_ids:
                out.append(" " + s + " ")
            else:
                out.append(s)
        return re.sub(r" +", " ", "".join(out)).strip()


# ----------------------------------------------------------------------------- model
class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B A x. With adapters attached, W and b are frozen."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.r = 0
        self.A = None
        self.B = None
        self.scale = 0.0

    def add_adapter(self, r, alpha, gen: torch.Generator):
        d_in, d_out = self.base.in_features, self.base.out_features
        self.r = r
        self.A = nn.Parameter(torch.randn(r, d_in, generator=gen) / math.sqrt(d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        self.scale = alpha / r

    def forward(self, x):
        y = self.base(x)
        if self.r:
            y = y + (x @ self.A.t() @ self.B.t()) * self.scale
        return y


class Block(nn.Module):
    def __init__(self, d, h, ff, drop):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = LoRALinear(nn.Linear(d, 3 * d))
        self.proj = LoRALinear(nn.Linear(d, d))
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = LoRALinear(nn.Linear(d, ff))
        self.fc2 = LoRALinear(nn.Linear(ff, d))
        self.p = drop

    def forward(self, x, mask, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        sh = lambda t: t.view(B, T, self.h, D // self.h).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.p if self.training else 0.0)
        a = a.transpose(1, 2).reshape(B, T, D)
        x = x + F.dropout(self.proj(a), self.p, self.training)
        x = x + F.dropout(self.fc2(F.gelu(self.fc1(self.ln2(x)))), self.p, self.training)
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, d=128, layers=4, heads=4, ff=512, max_len=320, drop=0.1):
        super().__init__()
        self.cfg = dict(vocab=vocab, d=d, layers=layers, heads=heads, ff=ff, max_len=max_len, drop=drop)
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads, ff, drop) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight                  # tied
        self.p = drop
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self, trainable_only=False):
        seen, n = set(), 0
        for p in self.parameters():
            if id(p) in seen or (trainable_only and not p.requires_grad):
                continue
            seen.add(id(p))
            n += p.numel()
        return n

    def lora_layers(self):
        return [m for m in self.modules() if isinstance(m, LoRALinear)]

    def forward(self, ids, pos, mask, caches=None):
        x = F.dropout(self.tok(ids) + self.pos(pos), self.p, self.training)
        for i, b in enumerate(self.blocks):
            x = b(x, mask, None if caches is None else caches[i])
        return self.head(self.ln(x))


# ----------------------------------------------------------------------------- adapters
def base_state_hash(model: TinyLM) -> str:
    """Hash of the base weights (adapters excluded); shows that a base stays frozen (cf. T8)."""
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        if name.endswith(".A") or name.endswith(".B"):
            continue
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def adapter_hash(model: TinyLM) -> str:
    h = hashlib.sha256()
    for m in model.lora_layers():
        if m.r:
            h.update(m.A.detach().numpy().tobytes())
            h.update(m.B.detach().numpy().tobytes())
    return h.hexdigest()[:16]


def attach_adapters(model: TinyLM, r: int, alpha: float, seed: int):
    gen = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.lora_layers():
        m.add_adapter(r, alpha, gen)
    return [p for m in model.lora_layers() for p in (m.A, m.B)]


# ----------------------------------------------------------------------------- batching
def encode_pair(tok: Tokenizer, prompt, target, max_len):
    p = [BOS] + tok.encode(prompt) + [SEP]
    t = tok.encode(target) + [EOS]
    if len(p) + len(t) > max_len:
        raise ValueError(f"example longer than max_len={max_len}: {len(p) + len(t)} tokens")
    return p, t


def make_batch(encoded):
    """Teacher-forced batch from encoded (prompt_ids, target_ids): <bos> prompt <sep> target <eos>;
    the loss covers only target tokens and <eos>."""
    T = max(len(p) + len(t) for p, t in encoded)
    ids = torch.full((len(encoded), T), PAD, dtype=torch.long)
    lm = torch.zeros((len(encoded), T), dtype=torch.float)
    for i, (p, t) in enumerate(encoded):
        ids[i, :len(p) + len(t)] = torch.tensor(p + t)
        lm[i, len(p):len(p) + len(t)] = 1.0
    return ids, lm


def lm_loss(model: TinyLM, ids, lmask):
    B, T = ids.shape
    pos = torch.arange(T).unsqueeze(0).expand(B, T)
    causal = torch.ones(T - 1, T - 1, dtype=torch.bool).tril()
    logits = model(ids[:, :-1], pos[:, :-1], causal)
    tgt = ids[:, 1:]
    m = lmask[:, 1:].reshape(-1)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none")
    return (loss * m).sum() / m.sum().clamp_min(1.0)


# ----------------------------------------------------------------------------- sampling
@torch.no_grad()
def sample(model: TinyLM, tok: Tokenizer, prompts, seeds, temperature=1.0, max_new=128, batch=256):
    """One candidate per (prompt, seed). The sampling noise of row i comes only from seeds[i]
    (Gumbel-max with a per-row generator), so a sample is a function of (model, prompt, seed),
    independent of batch composition (rule R6)."""
    model.eval()
    out = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: (len(prompts[i]), i))
    V = model.cfg["vocab"]
    for b0 in range(0, len(order), batch):
        idx = order[b0:b0 + batch]
        enc = [[BOS] + tok.encode(prompts[i]) + [SEP] for i in idx]
        Lmax = max(len(e) for e in enc)
        B = len(idx)
        steps = max(1, min(max_new, model.cfg["max_len"] - Lmax))
        ids = torch.full((B, Lmax), PAD, dtype=torch.long)
        keep = torch.zeros((B, Lmax), dtype=torch.bool)
        pos = torch.zeros((B, Lmax), dtype=torch.long)
        for r, e in enumerate(enc):
            off = Lmax - len(e)                       # left padding
            ids[r, off:] = torch.tensor(e)
            keep[r, off:] = True
            pos[r, off:] = torch.arange(len(e))
        noise = torch.empty((B, steps, V))
        for r, i in enumerate(idx):
            g = torch.Generator().manual_seed(int(seeds[i]) % (2 ** 63))
            u = torch.rand((steps, V), generator=g).clamp_(1e-10, 1 - 1e-10)
            noise[r] = -torch.log(-torch.log(u))
        caches = [dict() for _ in model.blocks]
        causal = torch.ones(Lmax, Lmax, dtype=torch.bool).tril()
        mask = (causal.unsqueeze(0) & keep.unsqueeze(1)) | torch.eye(Lmax, dtype=torch.bool).unsqueeze(0)
        logits = model(ids, pos, mask.unsqueeze(1), caches)[:, -1]
        cur = pos[:, -1] + 1
        gen = torch.full((B, steps), PAD, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool)
        for s in range(steps):
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                nxt = (logits / temperature + noise[:, s]).argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            gen[:, s] = nxt
            done = done | (nxt == EOS)
            if bool(done.all()) or s == steps - 1:
                break
            keep = torch.cat([keep, torch.ones((B, 1), dtype=torch.bool)], dim=1)
            logits = model(nxt.view(B, 1), cur.clamp_max(model.cfg["max_len"] - 1).view(B, 1),
                           keep.view(B, 1, 1, -1), caches)[:, -1]
            cur = cur + 1
        for r, i in enumerate(idx):
            row = gen[r].tolist()
            if EOS in row:
                row = row[:row.index(EOS)]
            out[i] = tok.decode([t for t in row if t != PAD])
    return out
