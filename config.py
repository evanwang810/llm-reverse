"""Training schedule defaults.

The model shape used to live here. It does not any more: it comes from whatever
base checkpoint you finetune, so see bases.py. What is left is the schedule, and
the numbers below are finetuning numbers, not pretraining ones.

The important one is the learning rate. llm67m ran at 1e-3, which is right for a
model starting from noise and badly wrong for one starting from a converged
checkpoint: it erases the pretrained weights in the first few hundred steps and
you spend the session recovering ground you already had. Reversal is a large
enough shift to want more than an SFT rate and much less than a pretraining one,
so the per-base entries sit at 1e-4 to 3e-4 and scale down with model size.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Defaults. bases.py overrides the batch shape, lr and warmup per base."""

    micro_batch: int = 4
    grad_accum: int = 4
    block_size: int = 1024

    lr: float = 1e-4
    min_lr: float = 1e-5
    warmup_steps: int = 200
    decay_steps: int = 1500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    log_every: int = 20
    eval_every: int = 250
    eval_batches: int = 40
    save_every_min: float = 30.0
    keep_checkpoints: int = 2
    keep_weights: int = 4
    milestone_every_min: float = 90.0

    # 11.3h leaves margin inside Kaggle's 12h GPU cap. The TPU cap is 9h, and
    # kaggle_run.sh clamps for it rather than making you remember.
    deadline_hours: float = 11.3
    reserve_minutes: float = 8.0
    seed: int = 1337

    def tokens_per_step(self, world_size: int) -> int:
        return self.micro_batch * self.grad_accum * world_size * self.block_size


if __name__ == "__main__":
    import bases

    print("schedule defaults:")
    t = TrainConfig()
    for field in ("lr", "min_lr", "warmup_steps", "decay_steps", "weight_decay",
                  "grad_clip", "deadline_hours"):
        print(f"  {field:16s} {getattr(t, field)}")
    print("\nper-base overrides live in bases.py:\n")
    for key in sorted(bases.BASES):
        print(bases.report(bases.get(key)))
        print()
