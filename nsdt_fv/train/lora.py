"""Arm B2 (plan Table 3): sequential fine-tuning on each window in turn; nothing from earlier windows
is re-read.

Two parameterisations (config adapt.mode): "lora" = low-rank adapters on the frozen base, as in
Table 3; "full" = every weight of the tiny model is trained, the full-parameter equivalent the S1
order allows for the toy (the choice and its reason are in reports/CHANGES.md, S1 calibration).
"""
from __future__ import annotations

import copy

from .common import epochs_to_steps, train_steps
from .model import attach_adapters


def make_trainable(model, cfg, seed):
    if cfg.get("mode", "lora") == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        return list(model.parameters())          # tied embedding/head counted once
    return attach_adapters(model, cfg["lora_r"], cfg["lora_alpha"], seed)


class Sequential:
    arm = "B2"

    def __init__(self, model, tok, cfg, seed, guard):
        self.model, self.tok, self.cfg, self.seed, self.guard = model, tok, cfg, seed, guard
        self.params = make_trainable(model, cfg, seed)
        self.history_bytes = 0                 # bytes of earlier-window proof text read (section 7)

    def stage_examples(self, t, window_examples, history):
        return list(window_examples)

    def train_stage(self, t, window_examples, history):
        exs = self.stage_examples(t, window_examples, history)
        n_rep = len(exs) - len(window_examples)
        self.history_bytes += sum(len(e["target"].encode()) for e in exs[len(window_examples):])
        steps = epochs_to_steps(len(exs), self.cfg["epochs"], self.cfg["batch"])
        st = train_steps(self.model, self.tok, self.params, exs, steps=steps, batch=self.cfg["batch"],
                         lr=self.cfg["lr"], seed=self.seed * 1000 + t, guard=self.guard,
                         weight_decay=self.cfg.get("weight_decay", 0.0))
        st.update(n_examples=len(exs), n_window=len(window_examples), n_replay=n_rep,
                  history_bytes_cum=self.history_bytes)
        return st


SequentialLoRA = Sequential
