"""Arm B5-p (plan Table 3): sequential fine-tuning plus replay of stored human proofs from earlier windows.

At stage t the model trains on the window's wake examples plus a uniform sample of p percent of all
earlier wake examples (at least one when any history exists), drawn afresh at each stage with a seed
that depends on the run seed and the stage (R6).
"""
from __future__ import annotations

import math
import random

from .lora import Sequential


class Replay(Sequential):
    def __init__(self, model, tok, cfg, seed, guard, percent=10):
        super().__init__(model, tok, cfg, seed, guard)
        self.percent = percent
        self.arm = f"B5-{percent}"

    def stage_examples(self, t, window_examples, history):
        if not history:
            return list(window_examples)
        n = max(1, int(math.ceil(len(history) * self.percent / 100.0)))
        rng = random.Random(self.seed * 1_000_003 + t)
        picked = rng.sample(range(len(history)), n)
        return list(window_examples) + [history[i] for i in sorted(picked)]


ReplayLoRA = Replay
