
import sys, json, time, torch, pickle, os, random
torch.set_num_threads(2)
from nsdt_fv.data.stream import build_stream, test_view
from nsdt_fv.data.holdout import HoldoutGuard
from nsdt_fv.eval.s1 import KEYWORDS, wake_example, make_model
from nsdt_fv.train.model import Tokenizer, attach_adapters, sample
from nsdt_fv.train.common import train_steps, epochs_to_steps
import yaml
mode, epochs, lr, r = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
cfg = yaml.safe_load(open('nsdt_fv/configs/S1.yaml'))
s = pickle.load(open('/tmp/s1_stream0.pkl','rb'))
w=dict(s.worlds); w['toy-prelude']=s.prelude.world
g = HoldoutGuard(s.obligations, w)
tok = Tokenizer(KEYWORDS)
m = make_model(tok, cfg, 0); m.load_state_dict(torch.load('/tmp/s1cal/c1/base/base.pt'))
wake = [wake_example(o) for o in s.obligations if o['split']=='wake']
tests = [o for o in s.obligations if o['split']=='test']
if mode=='lora': params = attach_adapters(m, r, 2*r, 0)
else: params = list(m.parameters())
print('trainable', sum(p.numel() for p in params))
steps = epochs_to_steps(len(wake), epochs, 32)
t=time.time()
st = train_steps(m, tok, params, wake, steps=steps, batch=32, lr=lr, seed=0, guard=g.check, log_every=300)
print('train', time.time()-t, st['last_loss'], flush=True)
be = s.backend()
def acc(obls, k=8):
  prompts=[o['statement'] for o in obls for _ in range(k)]
  c = sample(m, tok, prompts, list(range(len(prompts))), temperature=1.0, max_new=128)
  ok=0; triv=0; nt=0
  for i,o in enumerate(obls):
    a = any(be.check(o['snapshot_id'],o['id'],c[i*k+j],120).accepted for j in range(k))
    ok+=a
    if o['closed_by_baseline_tactics']: triv+=a
    else: nt+=a
  return ok/len(obls), triv, nt
print('test pass@8', acc(tests))
random.seed(0)
print('train-sample pass@8', acc(random.sample([o for o in s.obligations if o['split']=='wake'],120)))
