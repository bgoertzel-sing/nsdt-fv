"""Arm B6 (plan Table 3): joint retraining on all windows so far (the ceiling).

At each stage t the model is reset to the base (adapters re-initialised with the same seed in
"lora" mode; base weights reloaded in "full" mode) and trained on the union of the wake examples of
windows 1..t for adapt.joint_epochs epochs.
"""
from __future__ import annotations

import copy

from .common import epochs_to_steps, train_steps
from .lora import make_trainable


class Joint:
    arm = "B6"

    def __init__(self, model, tok, cfg, seed, guard):
        self.model, self.tok, self.cfg, self.seed, self.guard = model, tok, cfg, seed, guard
        self.base_state = copy.deepcopy(model.state_dict())
        self.params = None
        self.history_bytes = 0

    def train_stage(self, t, window_examples, history):
        exs = list(history) + list(window_examples)
        if self.cfg.get("mode", "lora") == "full":
            self.model.load_state_dict(self.base_state)
        self.params = make_trainable(self.model, self.cfg, self.seed)
        self.history_bytes += sum(len(e["target"].encode()) for e in history)
        steps = epochs_to_steps(len(exs), self.cfg.get("joint_epochs", self.cfg["epochs"]), self.cfg["batch"])
        st = train_steps(self.model, self.tok, self.params, exs, steps=steps, batch=self.cfg["batch"],
                         lr=self.cfg["lr"], seed=self.seed * 1000 + t, guard=self.guard,
                         weight_decay=self.cfg.get("weight_decay", 0.0))
        st.update(n_examples=len(exs), n_window=len(window_examples), n_replay=len(history),
                  history_bytes_cum=self.history_bytes)
        return st


JointLoRA = Joint
