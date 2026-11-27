# S2 calibration probe (CHANGES.md "S2 calibration"): can the DT-ref cap fit the toy at all?
# Same protocol as S1 probe 2 (runs/S1_calibration/probe.py): one fit on all 962 wake examples of the
# stream, 30 epochs, batch 32, lr 1e-3; then pass@8 on the 240 test obligations and on 120 sampled
# training items. Here only the cap (residual adapters on the top layers + untied head) is trained;
# the S1 seed-0 base is frozen. Usage: python probe_cap.py R [LAYERS e.g. 2,3 or "top"] [KIND mlp|block]
import sys, time, random, torch
torch.set_num_threads(2)
from nsdt_fv.data.stream import build_stream
from nsdt_fv.data.holdout import HoldoutGuard
from nsdt_fv.eval.s1 import KEYWORDS, wake_example, make_model
from nsdt_fv.train.model import Tokenizer, sample, base_state_hash
from nsdt_fv.train.common import train_steps, epochs_to_steps
from nsdt_fv.dt.dt_ref import CapLM
import yaml
r = int(sys.argv[1]); layers = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 and sys.argv[2] != "top" else None
kind = sys.argv[3] if len(sys.argv) > 3 else "mlp"
cfg = yaml.safe_load(open('nsdt_fv/configs/S1.yaml'))
s = build_stream(seed=0)
w = dict(s.worlds); w['toy-prelude'] = s.prelude.world
g = HoldoutGuard(s.obligations, w)
tok = Tokenizer(KEYWORDS)
base = make_model(tok, cfg, 0); base.load_state_dict(torch.load('nsdt_fv/runs/S2/base/base.pt'))
h0 = base_state_hash(base)
m = CapLM(base, r, layers, seed=0, kind=kind)
params = m.cap_parameters()
print('base', h0, 'cap', kind, 'layers', m.layers, 'r', r, 'trainable', sum(p.numel() for p in params), flush=True)
wake = [wake_example(o) for o in s.obligations if o['split'] == 'wake']
tests = [o for o in s.obligations if o['split'] == 'test']
steps = epochs_to_steps(len(wake), 30, 32)
t = time.time()
st = train_steps(m, tok, params, wake, steps=steps, batch=32, lr=1e-3, seed=0, guard=g.check, log_every=300)
print('train', round(time.time() - t), st['last_loss'], 'base unchanged', base_state_hash(base) == h0, flush=True)
be = s.backend()
def acc(obls, k=8):
    prompts = [o['statement'] for o in obls for _ in range(k)]
    c = sample(m, tok, prompts, list(range(len(prompts))), temperature=1.0, max_new=128)
    ok = triv = nt = 0
    for i, o in enumerate(obls):
        a = any(be.check(o['snapshot_id'], o['id'], c[i * k + j], 120).accepted for j in range(k))
        ok += a
        if o['closed_by_baseline_tactics']: triv += a
        else: nt += a
    return ok / len(obls), triv, nt
print('test pass@8', acc(tests), flush=True)
random.seed(0)
print('train-sample pass@8', acc(random.sample([o for o in s.obligations if o['split'] == 'wake'], 120)), flush=True)
