"""The two base models, and the arithmetic behind why they are the two.

    python bases.py            # print both, with the memory math

This replaces llm67m's preset table. There is no architecture to choose any
more: the shape comes from whatever `AutoModelForCausalLM.from_pretrained` hands
back, so the only real decisions are which checkpoint to start from and what
fits in the device's memory.

Why full finetuning and not LoRA. Reversing token order is a far larger
distribution shift than instruction tuning, closer to continued pretraining than
to SFT. A low-rank adapter underfits it badly, and the failure looks like "the
idea does not work" rather than "the method was wrong", which is the worst kind
of negative result to get. Every entry here is sized for a full finetune.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


# Bytes of optimizer state per parameter for a full mixed-precision AdamW step:
# half weights (2) + half grads (2) + fp32 master copy (4) + Adam m and v (8).
BYTES_PER_PARAM = 16


@dataclass(frozen=True)
class Base:
    """One base checkpoint plus the batch shape that fits its target device."""

    key: str
    hf_name: str
    params: int
    vocab_size: int
    device: str            # the device this entry is sized for
    block_size: int
    micro_batch: int
    grad_accum: int
    world: int             # devices the shape assumes: 2 T4s, or 8 TPU chips
    lr: float
    warmup_steps: int
    note: str

    @property
    def state_gb(self) -> float:
        return self.params * BYTES_PER_PARAM / 1e9

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch * self.grad_accum * self.world * self.block_size

    @property
    def dtype(self) -> str:
        """uint16 tops out at 65535, and Qwen3's vocab does not fit in it."""
        return "uint16" if self.vocab_size <= 65535 else "uint32"

    @property
    def bytes_per_token(self) -> int:
        return 2 if self.dtype == "uint16" else 4

    def as_dict(self) -> dict:
        return asdict(self)


BASES: dict[str, Base] = {
    # 30 minute model. The point is a fast iteration loop, so it runs on the T4
    # pair: XLA compilation plus the TPU preflight would eat a fifth of the
    # budget before a single step ran. 135M in fp16 needs 2.2 GB of optimizer
    # state, which leaves most of a 16 GB card for activations, hence the
    # comfortable micro-batch.
    "small": Base(
        key="small",
        hf_name="HuggingFaceTB/SmolLM2-135M",
        params=135_000_000,
        vocab_size=49_152,
        device="gpu",
        block_size=1024,
        micro_batch=16,
        grad_accum=4,
        world=2,
        lr=3e-4,
        warmup_steps=100,
        note="smoke model: proves the pipeline and gives a real loss curve, "
             "not something to publish",
    ),
    # Full session model. Dense, text-only, plain GQA, Apache-2.0. Deliberately
    # not one of the 2026 hybrids: Gated DeltaNet carries a recurrent state that
    # is direction-dependent, and sparse MoE re-routes hard under a shift this
    # large, so either would confound the reversal with the architecture.
    #
    # 0.6B at 16 bytes per parameter is 9.5 GB, which fits one 16 GB TPU v5e
    # chip with room for activations, so plain data parallel across the 8 chips
    # works and no sharding is needed. It also fits a single T4, which is what
    # makes --device gpu a real fallback rather than a formality.
    "large": Base(
        key="large",
        hf_name="Qwen/Qwen3-0.6B-Base",
        params=596_000_000,
        vocab_size=151_669,
        device="tpu",
        block_size=1024,
        micro_batch=4,
        grad_accum=4,
        world=8,
        lr=1e-4,
        warmup_steps=200,
        note="the one you publish. bf16 on TPU, so no GradScaler and no "
             "loss-scale thrash on a bf16-native base",
    ),
}

# Same checkpoint, sized for the T4 pair instead. Slower, but it needs no XLA.
BASES["large-gpu"] = Base(
    **{**BASES["large"].as_dict(), "key": "large-gpu", "device": "gpu",
       "micro_batch": 2, "grad_accum": 16, "world": 2,
       "note": "large on 2xT4: fits, but roughly a quarter the tokens per "
               "session, and fp16 on a bf16-native base"}
)


def get(key: str) -> Base:
    if key not in BASES:
        raise SystemExit(f"unknown base '{key}'. options: {', '.join(BASES)}")
    return BASES[key]


def report(base: Base) -> str:
    per_chip = "per T4" if base.device == "gpu" else "per v5e chip"
    lines = [
        f"{base.key}  ->  {base.hf_name}",
        f"  {base.params / 1e6:.0f}M params, vocab {base.vocab_size:,}, "
        f"shards as {base.dtype} ({base.bytes_per_token} bytes/token)",
        f"  full finetune state: {base.state_gb:.1f} GB {per_chip} "
        f"({BYTES_PER_PARAM} bytes/param)",
        f"  batch: {base.micro_batch} micro x {base.grad_accum} accum x "
        f"{base.world} {'gpu' if base.device == 'gpu' else 'chip'} x "
        f"{base.block_size} = {base.tokens_per_step:,} tokens/step",
        f"  lr {base.lr:g}, warmup {base.warmup_steps}",
        f"  {base.note}",
    ]
    return "\n".join(lines)


def session_estimate(base: Base, hours: float, tok_per_s: float) -> str:
    tokens = hours * 3600 * tok_per_s
    return (f"  ~{tok_per_s / 1000:.0f}k tok/s -> {tokens / 1e9:.2f}B tokens in "
            f"{hours:.1f}h, {tokens / base.tokens_per_step:,.0f} steps")


if __name__ == "__main__":
    for key in ("small", "large", "large-gpu"):
        b = BASES[key]
        print(report(b))
        # Rough, measured-elsewhere throughputs. The TPU number is the one that
        # justifies the whole port: three to four times the tokens per session,
        # out of a quota that does not compete with the GPU one.
        if key == "small":
            print(session_estimate(b, 0.5, 35_000))
        elif key == "large":
            print(session_estimate(b, 8.5, 45_000))
        else:
            print(session_estimate(b, 11.3, 9_000))
        print()
    print(f"uint16 caps at 65,535 token ids. Anything above that doubles the "
          f"corpus on disk.")
