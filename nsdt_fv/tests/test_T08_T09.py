"""S2 unit tests: T8 (base frozen under DT-ref) and the selection rule behind T9 (dream examples need an
accepted verdict in the kernel-filtered arms; D2 learns unfiltered dreams)."""
import types

import torch

from nsdt_fv.dt.dt_ref import CapLM, DTRef, cap_hash, top_third
from nsdt_fv.eval.s2 import ARMS, ArmRun
from nsdt_fv.train.model import TinyLM, Tokenizer, base_state_hash

KW = ["forall", "simp", "refl", "rw", "induct", "cases"]


def _setup():
    tok = Tokenizer(KW)
    torch.manual_seed(0)
    base = TinyLM(tok.vocab_size, d=32, layers=3, heads=2, ff=64, max_len=64, drop=0.1)
    return tok, base


def test_top_third():
    assert top_third(4) == [2, 3] and top_third(3) == [2] and top_third(6) == [4, 5]


def test_cap_starts_equal_to_base_and_base_stays_frozen():
    tok, base = _setup()
    h0 = base_state_hash(base)
    m = CapLM(base, r=8, seed=1)
    m.eval(); base.eval()
    ids = torch.tensor([[1, 7, 8, 9, 2, 10]])
    pos = torch.arange(6).unsqueeze(0)
    mask = torch.ones(6, 6, dtype=torch.bool).tril()
    assert torch.allclose(m(ids, pos, mask), base(ids, pos, mask), atol=1e-6)
    seen = []
    core = DTRef(m, tok, {"lr": 1e-2}, 0, guard=seen.append)
    ex = {"kind": "wake", "prompt": "forall n. f(n) = n", "target": "induct n { simp refl } { simp rw IH refl }",
          "statement": "forall n. f(n) = n", "source": "human", "source_ids": ["x"]}
    bad = dict(ex, kind="dream", target="refl", source="model")
    c0 = cap_hash(m)
    core.wake([ex, ex], seed=0)
    core.learn([ex], {0: bad}, seed=0)
    assert cap_hash(m) != c0                   # the cap learned
    assert base_state_hash(base) == h0         # T8: the base did not change
    assert all(not p.requires_grad for p in base.parameters())
    assert len(seen) == 4                      # every example passed the guard callable (R4)


def _lines():
    mk = lambda vid, acc: {"verdict_id": vid, "accepted": acc, "guard": "pass"}
    items = [("a", "forall n. f(n) = n", [], None)]
    got = [("p1", mk("v1", True)), ("p2", mk("v2", False)), ("p1", mk("v3", True)), ("p3", mk("v4", False))]
    return items, [got]


def test_selection_filtered_arms_learn_only_accepted():
    for arm in ("D1", "D3", "D4"):
        dummy = types.SimpleNamespace(flags=ARMS[arm])
        items, res = _lines()
        acc, rej = ArmRun.select(dummy, 1, "nrem", items, res)
        assert [e["id"] for e in acc] == ["v1"]          # accepted, distinct texts only
        if ARMS[arm]["neg"]:
            assert rej[0]["id"] == "v2"                   # negative partner is a rejected candidate
        else:
            assert rej == {}


def test_selection_d2_is_unfiltered():
    dummy = types.SimpleNamespace(flags=ARMS["D2"])
    items, res = _lines()
    acc, rej = ArmRun.select(dummy, 1, "nrem", items, res)
    assert [e["id"] for e in acc] == ["v1", "v2", "v4"]   # rejected candidates are learned too
    assert rej == {}
