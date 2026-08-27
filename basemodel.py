"""Wraps a HuggingFace causal LM in the interface llm67m's trainer already speaks.

The trainer, the TPU trainer, the resume logic, the checkpoint pruning and the
dashboard were all written against a model that exposes `.cfg`, returns
`(logits, loss)` from `forward(idx, targets)`, and builds its own optimizer.
Rather than rewrite four files to HuggingFace's conventions, this adapter
presents those three things over an `AutoModelForCausalLM`. The diff to the
trainer is about twenty lines and the resume protocol is untouched.

The loss is computed here rather than by passing `labels=` to the HF model,
because the dataloader already hands over pre-shifted `(x, y)` pairs. Letting
transformers shift again would drop a token per block and quietly train on a
one-token-offset target, which costs a little accuracy and shows up as nothing
at all in the loss curve.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import bases


@dataclass
class FinetuneConfig:
    """The fields the resume check compares. Architecture comes from the base."""

    base_key: str
    hf_name: str
    block_size: int
    vocab_size: int
    direction: str = "reverse"
    # 0 means every parameter was trained. Non-zero means the checkpoint holds
    # adapters over a frozen base, which changes what publishing has to do.
    lora_rank: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def strip_prefixes(state: dict) -> dict:
    """Remove DDP and torch.compile wrappers, and this adapter's own, from keys."""
    out = {}
    for k, v in state.items():
        for prefix in ("module.", "_orig_mod.", "hf."):
            while k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = v
    return out


class FinetuneModel(nn.Module):
    def __init__(self, hf_model, cfg: FinetuneConfig) -> None:
        super().__init__()
        self.hf = hf_model
        self.cfg = cfg

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        out = self.hf(input_ids=idx, use_cache=False)
        logits = out.logits
        if targets is None:
            return logits[:, -1:, :], None
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1),
            ignore_index=-1)
        return logits, loss

    def _apply(self, *args, **kwargs):
        """Re-tie the output head after anything that rebuilds the parameters.

        nn.Module._apply can only update a parameter in place when the result is
        shallow-copy compatible with the original. A move to a different device
        type is not, so it builds a fresh Parameter for each entry in each
        module's dict, and a Parameter shared by two modules comes back as two
        independent tensors.

        On CUDA this never surfaced. On XLA it does: .to(xla) silently untied
        lm_head from the input embedding, giving 162,826,560 parameters where the
        base has 134,515,008. The difference is 49,152 x 576, exactly one
        embedding matrix. That is an extra copy of weights, gradients and Adam
        moments per replica, and two output heads training apart.

        The quiet part is the checkpoint, which would carry an lm_head that
        resume then throws away. Re-tying here rather than at each call site
        means a device move cannot undo it.
        """
        out = super()._apply(*args, **kwargs)
        if getattr(self.hf.config, "tie_word_embeddings", False):
            self.hf.tie_weights()
        return out

    # ------------------------------------------------------------------ #

    @property
    def is_lora(self) -> bool:
        return self.cfg.lora_rank > 0

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def non_embedding_params(self) -> int:
        emb = self.hf.get_input_embeddings().weight.numel()
        out = self.hf.get_output_embeddings()
        if out is not None and out.weight is not self.hf.get_input_embeddings().weight:
            emb += out.weight.numel()
        return sum(p.numel() for p in self.parameters()) - emb

    def param_report(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = self.trainable_params()
        non_emb = self.non_embedding_params()
        hf_cfg = self.hf.config
        tied = getattr(hf_cfg, "tie_word_embeddings", False)
        if self.is_lora:
            frozen = total - trainable
            state = (frozen * bases.BYTES_PER_FROZEN_PARAM
                     + trainable * bases.BYTES_PER_PARAM) / 1e9
            how = [f"  LoRA rank {self.cfg.lora_rank}: {trainable:,} trainable "
                   f"({trainable / total:.1%}), {frozen:,} frozen",
                   f"  state                : {state:.1f} GB "
                   f"(frozen at {bases.BYTES_PER_FROZEN_PARAM} bytes, "
                   f"trainable at {bases.BYTES_PER_PARAM})"]
        else:
            how = [f"  full finetune state  : "
                   f"{total * bases.BYTES_PER_PARAM / 1e9:.1f} GB "
                   f"({bases.BYTES_PER_PARAM} bytes/param)"]
        return "\n".join([
            f"base: {self.cfg.hf_name}  ({self.cfg.base_key})",
            f"  {getattr(hf_cfg, 'num_hidden_layers', '?')} layers, "
            f"hidden {getattr(hf_cfg, 'hidden_size', '?')}, "
            f"{getattr(hf_cfg, 'num_attention_heads', '?')} heads, "
            f"vocab {self.cfg.vocab_size:,}, block_size {self.cfg.block_size}",
            f"  non-embedding params : {non_emb:,} ({non_emb / 1e6:.2f}M)",
            f"  total params         : {total:,} ({total / 1e6:.2f}M)"
            + (" tied embeddings" if tied else ""),
            *how,
        ])

    def configure_optimizer(self, lr: float, weight_decay: float,
                            betas: tuple[float, float], device_type: str):
        """Decay matrices, not vectors. Norms and biases stay undecayed."""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        kwargs = {}
        if device_type == "cuda":
            kwargs["fused"] = True  # supported on Turing, meaningfully faster
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, **kwargs)

    def estimated_flops_per_token(self) -> float:
        n = sum(p.numel() for p in self.parameters())
        layers = getattr(self.hf.config, "num_hidden_layers", 0)
        hidden = getattr(self.hf.config, "hidden_size", 0)
        return 6 * n + 12 * layers * hidden * self.cfg.block_size


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def apply_lora(hf_model, base):
    """Wrap the model in LoRA adapters and freeze everything else.

    Targets every linear in the block, not just attention. Reversal changes what
    the MLP computes as much as what attention attends to, and the conventional
    attention-only target list leaves two thirds of each layer frozen.

    The base weights are cast to half precision because they are frozen and only
    ever read; that cast is most of why LoRA reaches a 4B model on a 16 GB chip.
    The adapters stay fp32, since those are the ones Adam has to keep moments for.
    """
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=base.lora_rank,
        lora_alpha=base.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(base.lora_targets),
    )
    model = get_peft_model(hf_model, cfg)
    half = torch.bfloat16 if base.device == "tpu" else torch.float16
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()
        else:
            param.data = param.data.to(half)
    return model


def attn_for(device: str) -> str:
    """Which attention implementation is safe on a given device.

    "eager" on XLA is not a performance compromise, it is a correctness one.
    Under bf16 autocast the rotary embedding promotes q and k back to fp32 while
    v stays bf16, and SDPA refuses a mixed-dtype call. Eager attention unifies
    them itself because its matmuls are autocast ops. XLA fuses the result
    either way, so the usual reason to prefer SDPA does not apply here.
    """
    return "eager" if device == "tpu" else "sdpa"


def build(base_key: str, block_size: int, direction: str = "reverse",
          device: torch.device | str = "cpu", attn: str = "sdpa") -> FinetuneModel:
    """Download the base checkpoint and wrap it.

    For a full finetune, weights land in fp32: that is the master copy, autocast
    supplies half precision for the forward pass, and Adam's moments want fp32
    anyway. Loading straight into fp16 would save memory now and cost the
    optimizer's precision for the whole run.

    For a LoRA run the frozen base goes to half precision instead, because
    nothing ever computes a gradient for it.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    base = bases.get(base_key)
    hf_cfg = AutoConfig.from_pretrained(base.hf_name)
    hf_cfg.use_cache = False
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            base.hf_name, config=hf_cfg, dtype=torch.float32, attn_implementation=attn)
    except (ValueError, ImportError):
        # FlashAttention-2 needs sm80+, and the T4 is sm75. SDPA is the fallback
        # everywhere, but let an unsupported request degrade rather than abort.
        hf_model = AutoModelForCausalLM.from_pretrained(
            base.hf_name, config=hf_cfg, dtype=torch.float32)
    if base.is_lora:
        hf_model = apply_lora(hf_model, base)
    if base.grad_checkpointing:
        hf_model.gradient_checkpointing_enable()
        hf_model.enable_input_require_grads()
    else:
        hf_model.gradient_checkpointing_disable()

    cfg = FinetuneConfig(base_key=base.key, hf_name=base.hf_name,
                         block_size=block_size, vocab_size=base.vocab_size,
                         direction=direction, lora_rank=base.lora_rank)
    return FinetuneModel(hf_model, cfg).to(device)


def build_smoke(block_size: int = 256, direction: str = "reverse",
                device: torch.device | str = "cpu",
                lora_rank: int = 0) -> FinetuneModel:
    """A tiny randomly-initialised model of the same family, built offline.

    SmolLM2 is a Llama, so a hand-built LlamaConfig needs no network and no
    cache. The smoke test then walks the identical trainer, resume and sampling
    code path that a real run does, which is the only thing that makes it worth
    running.

    lora_rank turns on the adapter path, so `--smoke-test --base xlarge-lora`
    exercises LoRA without downloading four billion parameters.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    hf_cfg = LlamaConfig(
        vocab_size=49_152, hidden_size=128, intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=block_size, tie_word_embeddings=True,
        bos_token_id=0, eos_token_id=0, use_cache=False,
    )
    hf_model = LlamaForCausalLM(hf_cfg)
    if lora_rank:
        probe = bases.Base(
            key="smoke-lora", hf_name="HuggingFaceTB/SmolLM2-135M", params=0,
            vocab_size=49_152, device="gpu", block_size=block_size,
            micro_batch=1, grad_accum=1, world=1, lr=2e-4, warmup_steps=10,
            note="", lora_rank=lora_rank, lora_alpha=2 * lora_rank,
            lora_targets=bases.ALL_LINEAR)
        hf_model = apply_lora(hf_model, probe)
    cfg = FinetuneConfig(base_key="smoke", hf_name="HuggingFaceTB/SmolLM2-135M",
                         block_size=block_size, vocab_size=49_152,
                         direction=direction, lora_rank=lora_rank)
    return FinetuneModel(hf_model, cfg).to(device)


def load_from_checkpoint(path, device: str | torch.device = "cpu"):
    """Rebuild a FinetuneModel from a training checkpoint.

    The checkpoint stores only the finetuned weights and which base they came
    from, not the architecture, so this re-downloads the base to get the shape
    and then overwrites every parameter. That means a first load needs the Hub
    (or a warm cache); it also means a 600M checkpoint is 1.2 GB of weights
    rather than a self-describing blob nobody can read without this repo.
    """
    from pathlib import Path as _Path

    path = _Path(path)
    if path.is_dir():
        raise SystemExit(
            f"{path} is a directory, not a checkpoint file. An interrupted "
            "download often leaves a folder named like the file. Delete it and "
            "download the .pt again.")
    if not path.exists():
        raise SystemExit(f"no such checkpoint: {path}")
    if path.stat().st_size < 1024:
        raise SystemExit(
            f"{path} is only {path.stat().st_size} bytes, so the download did not "
            "finish. Delete it and try again.")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved = dict(ckpt["config"]["model"])
    fields = FinetuneConfig.__dataclass_fields__
    cfg = FinetuneConfig(**{k: v for k, v in saved.items() if k in fields})
    # The direction is stamped at the top level too, and that copy is the one
    # weights-only files carry.
    cfg.direction = str(ckpt.get("direction", cfg.direction))

    # lora_rank has to be passed through: without it the smoke path rebuilds a
    # plain model and every adapter key in the checkpoint comes back unexpected.
    model = (build_smoke(cfg.block_size, cfg.direction, "cpu", cfg.lora_rank)
             if cfg.base_key == "smoke"
             else build(cfg.base_key, cfg.block_size, cfg.direction))
    model.cfg = cfg

    state = {f"hf.{k}": v.float() for k, v in strip_prefixes(ckpt["model"]).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    tied = getattr(model.hf.config, "tie_word_embeddings", False)
    drop = ("rotary_emb", "inv_freq") + (("lm_head.weight",) if tied else ())
    missing = [k for k in missing if not any(d in k for d in drop)]
    unexpected = [k for k in unexpected if not any(d in k for d in drop)]
    if cfg.lora_rank:
        # A LoRA checkpoint stores adapters only. Everything else came from the
        # base download a few lines up and is supposed to be "missing".
        missing = [k for k in missing if "lora_" in k]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch. missing={missing} unexpected={unexpected}")
    return model.to(device).eval(), ckpt


def compare_configs(saved: dict, live: dict) -> dict:
    """Fields that must agree for a resume to be meaningful."""
    keys = ("base_key", "hf_name", "block_size", "vocab_size")
    return {k: (saved.get(k), live.get(k)) for k in keys if saved.get(k) != live.get(k)}


if __name__ == "__main__":
    m = build_smoke()
    print(m.param_report())
    x = torch.randint(0, 49_152, (2, 64))
    logits, loss = m(x, x)
    print(f"forward ok: logits {tuple(logits.shape)}, loss {loss.item():.3f}")
