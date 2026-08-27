"""The two base models, and the arithmetic behind why they are the two.

    python bases.py            # print both, with the memory math

This replaces llm67m's preset table. There is no architecture to choose any
more: the shape comes from whatever `AutoModelForCausalLM.from_pretrained` hands
back, so the only real decisions are which checkpoint to start from and what
fits in the device's memory.

Full finetune or LoRA. Reversing token order is a far larger distribution shift
than instruction tuning, closer to continued pretraining, so the received wisdom
is that a low-rank adapter will underfit it. That is a hypothesis, not a result:
nobody has published a clean reverse-LoRA either way.

So both are here. The full-finetune entries are the control, and the LoRA entries
are the experiment, because LoRA is the only way to reach a model big enough to
be interesting on this hardware. Run `small` first: without a full-finetune loss
curve to compare against, a disappointing LoRA run tells you nothing about
whether the method or the idea was at fault.

The honest trade, from the table below: xlarge-lora is a 4B model that sees
0.28B tokens in a session. large is a 0.6B model that sees 1.38B. Bigger base,
five times less adaptation, and a constrained update. Which of those wins is the
question the experiment answers.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


# Bytes of optimizer state per parameter for a full mixed-precision AdamW step:
# half weights (2) + half grads (2) + fp32 master copy (4) + Adam m and v (8).
BYTES_PER_PARAM = 16

# Activations dominate at these batch sizes and the first version of this file
# ignored them, which cost a real T4 shakedown to OOM. Saved activations for the
# backward pass run to roughly 20 tensors of (micro_batch x block x hidden) in
# half precision, per layer:
#
#     activation_gb ~= layers * micro_batch * block * hidden * 2 * 20 / 1e9
#
# SmolLM2-135M at micro_batch 16 and block 1024 predicts 11.3 GB of activations
# on top of 2.2 GB of state, against a card with 14.56 GiB. It died at 13.69 GB
# allocated. The shapes below keep the total near two thirds of a device.
ACT_TENSORS_PER_LAYER = 20

# A frozen parameter costs only its half-precision weight. This is the entire
# reason LoRA reaches models a full finetune cannot: at 4B, 16 bytes per
# parameter is 64 GB and 2 bytes is 8 GB, and a v5e chip has 16.
BYTES_PER_FROZEN_PARAM = 2


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
    # 0 means a full finetune. Anything else trains LoRA adapters on the
    # targeted projections and freezes the rest.
    lora_rank: int = 0
    lora_alpha: int = 0
    lora_targets: tuple[str, ...] = ()
    grad_checkpointing: bool = False
    # Shape of the base, used only for the activation estimate below.
    layers: int = 0
    hidden: int = 0
    # Rough trainable count for the LoRA entries, so the memory table is
    # printable without downloading four billion parameters. basemodel reports
    # the real number once the model is built.
    lora_params_estimate: int = 0

    @property
    def is_lora(self) -> bool:
        return self.lora_rank > 0

    @property
    def activation_gb(self) -> float:
        """Rough, and rough is enough to catch a shape that cannot fit."""
        if not (self.layers and self.hidden):
            return 0.0
        return (self.layers * self.micro_batch * self.block_size * self.hidden
                * 2 * ACT_TENSORS_PER_LAYER / 1e9)

    @property
    def state_gb(self) -> float:
        """Weights plus gradients plus optimizer moments, per device."""
        if not self.is_lora:
            return self.params * BYTES_PER_PARAM / 1e9
        frozen = (self.params - self.lora_params_estimate) * BYTES_PER_FROZEN_PARAM
        return (frozen + self.lora_params_estimate * BYTES_PER_PARAM) / 1e9

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
        # 4 x 16 x 2 x 1024 = 131,072 tokens/step, the same as every other
        # entry. micro_batch 16 OOMed a real T4; this is ~2.8 GB of activations.
        micro_batch=4,
        grad_accum=16,
        world=2,
        layers=30,
        hidden=576,
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
        # Qwen3-0.6B is 28 layers at hidden 1024, so each unit of micro_batch
        # costs ~1.2 GB of activations on top of 9.5 GB of state. 2 fits a 16 GB
        # v5e chip with room to spare; 4 would not.
        micro_batch=2,
        grad_accum=8,
        world=8,
        layers=28,
        hidden=1024,
        lr=1e-4,
        warmup_steps=200,
        note="the one you publish. bf16 on TPU, so no GradScaler and no "
             "loss-scale thrash on a bf16-native base",
    ),
}

# Same checkpoint, sized for the T4 pair instead. Slower, but it needs no XLA.
BASES["large-gpu"] = Base(
    **{**BASES["large"].as_dict(), "key": "large-gpu", "device": "gpu",
       "micro_batch": 1, "grad_accum": 32, "world": 2,
       "note": "large on 2xT4: fits, but roughly a quarter the tokens per "
               "session, and fp16 on a bf16-native base"}
)


# The cheapest possible proof that the TPU path works. Same 135M base as
# `small`, but on the v5e-8, so a failed XLA setup costs you an hour instead of
# a whole session. 2.2 GB of optimizer state against 16 GB of HBM leaves room
# for a far bigger micro-batch than the T4 pair allows.
#
# Run this before `large` the first time you touch a TPU. If it trains, then the
# device, the torch_xla build, the dataloader and the checkpoint path all work,
# and the only thing left untested at the larger size is whether it fits.
BASES["small-tpu"] = Base(
    **{**BASES["small"].as_dict(), "key": "small-tpu", "device": "tpu",
       "micro_batch": 4, "grad_accum": 4, "world": 8,
       "note": "TPU shakedown: 135M on v5e-8, one cheap hour to prove the "
               "device before betting a session on it"}
)


# LoRA on every linear in the block, not just attention. Reversal changes what
# the MLP computes as much as what attention attends to, and the usual
# attention-only target list leaves two thirds of each layer frozen.
ALL_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj")

# The LoRA answer to "why not reverse a big capable model". 4B dense, text-only,
# plain GQA, Apache-2.0, deliberately not one of the 2026 hybrids. Frozen weights
# are 8 GB in bf16 and rank-64 adapters on all seven projections add roughly
# 130M trainable parameters, so about 10 GB per chip before activations. That
# fits a 16 GB v5e chip with gradient checkpointing on and micro-batch 1.
#
# Rank 64 rather than the usual 8 or 16 on purpose: reversal is a much larger
# shift than instruction tuning, and a rank that works for style transfer will
# underfit it. This is the experiment, not the safe option. Run `small` first so
# you have a full-finetune loss curve to compare it against.
BASES["xlarge-lora"] = Base(
    key="xlarge-lora",
    hf_name="Qwen/Qwen3-4B-Base",
    params=4_000_000_000,
    vocab_size=151_669,
    device="tpu",
    block_size=1024,
    micro_batch=1,
    grad_accum=16,
    world=8,
    lr=2e-4,          # LoRA wants a higher rate than a full finetune
    warmup_steps=200,
    note="LoRA r=64 on a 4B base. The open question: can a low-rank update "
         "express a direction flip at all",
    lora_rank=64,
    lora_alpha=128,
    lora_targets=ALL_LINEAR,
    grad_checkpointing=True,
    lora_params_estimate=132_000_000,
)

# The same idea sized for the T4 pair, so you can prove the LoRA path works
# before spending a TPU session on it. 1.7B frozen in fp16 is 3.4 GB.
BASES["large-lora-gpu"] = Base(
    key="large-lora-gpu",
    hf_name="Qwen/Qwen3-1.7B-Base",
    params=1_700_000_000,
    vocab_size=151_669,
    device="gpu",
    block_size=1024,
    micro_batch=2,
    grad_accum=16,
    world=2,
    lr=2e-4,
    warmup_steps=200,
    note="LoRA r=64 on 1.7B, on 2xT4. A cheap rehearsal for xlarge-lora",
    lora_rank=64,
    lora_alpha=128,
    lora_targets=ALL_LINEAR,
    grad_checkpointing=True,
    lora_params_estimate=58_000_000,
)


def get(key: str) -> Base:
    if key not in BASES:
        raise SystemExit(f"unknown base '{key}'. options: {', '.join(BASES)}")
    return BASES[key]


def report(base: Base) -> str:
    per_chip = "per T4" if base.device == "gpu" else "per v5e chip"
    if base.is_lora:
        how = (f"  LoRA r={base.lora_rank} a={base.lora_alpha} on "
               f"{len(base.lora_targets)} projections, ~"
               f"{base.lora_params_estimate / 1e6:.0f}M trainable "
               f"({base.lora_params_estimate / base.params:.1%})")
        mem = (f"  state: {base.state_gb:.1f} GB {per_chip} "
               f"(frozen at {BYTES_PER_FROZEN_PARAM} bytes, trainable at "
               f"{BYTES_PER_PARAM})")
    else:
        how = "  full finetune, every parameter trainable"
        mem = (f"  state: {base.state_gb:.1f} GB {per_chip} "
               f"({BYTES_PER_PARAM} bytes/param)")
    lines = [
        f"{base.key}  ->  {base.hf_name}",
        f"  {base.params / 1e6:.0f}M params, vocab {base.vocab_size:,}, "
        f"shards as {base.dtype} ({base.bytes_per_token} bytes/token)",
        how,
        mem,
        (f"  activations ~{base.activation_gb:.1f} GB, total "
         f"~{base.activation_gb + base.state_gb:.1f} GB per device"
         if base.activation_gb else "  activations: unmeasured"),
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
    for key in ("small", "small-tpu", "large", "large-gpu",
                "large-lora-gpu", "xlarge-lora"):
        b = BASES[key]
        print(report(b))
        # Rough, measured-elsewhere throughputs. The TPU number is the one that
        # justifies the whole port: three to four times the tokens per session,
        # out of a quota that does not compete with the GPU one.
        rate = {"small": (0.5, 35_000), "small-tpu": (1.0, 110_000),
                "large": (8.5, 45_000), "large-gpu": (11.3, 9_000),
                "large-lora-gpu": (11.3, 4_500), "xlarge-lora": (8.5, 9_000)}[key]
        print(session_estimate(b, *rate))
        print()
    print(f"uint16 caps at 65,535 token ids. Anything above that doubles the "
          f"corpus on disk.")
