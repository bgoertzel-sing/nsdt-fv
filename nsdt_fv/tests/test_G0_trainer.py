"""Gate G0, trainer half (ISSUES.md item 17): the trainer modules import no concrete backend and no
ToyEq code, so replacing the ToyEq backend by a Lean backend needs no change to train/."""
import ast
import os

TRAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train")
FORBIDDEN = ("toyeq", "lean_vsb", "lean_repl", "mock", "data.stream", "data.toyeq")


def test_train_imports_no_backend():
    for fn in sorted(os.listdir(TRAIN)):
        if not fn.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(TRAIN, fn)).read())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [(node.module or "")] + [a.name for a in node.names]
            for m in mods:
                assert not any(f in m for f in FORBIDDEN), f"train/{fn} imports {m}"


def test_tokenizer_roundtrip_generic():
    from nsdt_fv.train.model import Tokenizer
    tok = Tokenizer(["by", "rw", "rw<-", "refl", "simp", "induct", "IH", "forall", "S", "Z"])
    for s in ["induct n { simp refl } { simp rw IH refl }", "simp rw<- ab_c1 rw x_y refl", "rw foo_bar refl"]:
        ids = tok.encode(s)
        assert tok.encode(tok.decode(ids)) == ids          # same token stream (spacing around symbols may differ)
        assert tok.decode(ids).split() == [w for w in tok.decode(ids).split()]


def test_sampler_is_seed_determined():
    import torch
    from nsdt_fv.train.model import TinyLM, Tokenizer, sample
    tok = Tokenizer(["refl", "simp"])
    torch.manual_seed(0)
    m = TinyLM(tok.vocab_size, d=32, layers=1, heads=2, ff=64, max_len=64, drop=0.0)
    p = ["f(x) = x", "g(y, z) = z"]
    a = sample(m, tok, p, [11, 12], max_new=10)
    b = sample(m, tok, [p[1], p[0], p[0]], [12, 11, 11], max_new=10)   # other batch composition
    assert a == [b[1], b[0]] and b[1] == b[2]
