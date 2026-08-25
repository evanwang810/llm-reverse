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

    # ------------------------------------------------------------------ #

    def non_embedding_params(self) -> int:
        emb = self.hf.get_input_embeddings().weight.numel()
        out = self.hf.get_output_embeddings()
        if out is not None and out.weight is not self.hf.get_input_embeddings().weight:
            emb += out.weight.numel()
        return sum(p.numel() for p in self.parameters()) - emb

    def param_report(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        non_emb = self.non_embedding_params()
        hf_cfg = self.hf.config
        tied = getattr(hf_cfg, "tie_word_embeddings", False)
        return "\n".join([
            f"base: {self.cfg.hf_name}  ({self.cfg.base_key})",
            f"  {getattr(hf_cfg, 'num_hidden_layers', '?')} layers, "
            f"hidden {getattr(hf_cfg, 'hidden_size', '?')}, "
            f"{getattr(hf_cfg, 'num_attention_heads', '?')} heads, "
            f"vocab {self.cfg.vocab_size:,}, block_size {self.cfg.block_size}",
            f"  non-embedding params : {non_emb:,} ({non_emb / 1e6:.2f}M)",
            f"  total params         : {total:,} ({total / 1e6:.2f}M)"
            + (" tied embeddings" if tied else ""),
            f"  full finetune state  : {total * bases.BYTES_PER_PARAM / 1e9:.1f} GB "
            f"({bases.BYTES_PER_PARAM} bytes/param)",
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


def build(base_key: str, block_size: int, direction: str = "reverse",
          device: torch.device | str = "cpu", attn: str = "sdpa") -> FinetuneModel:
    """Download the base checkpoint and wrap it.

    Weights land in fp32. That is the master copy: autocast supplies the half
    precision for the forward pass, and Adam's moments want fp32 anyway. Loading
    straight into fp16 would save memory now and cost you the optimizer's
    precision for the whole run.
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
    hf_model.gradient_checkpointing_disable()

    cfg = FinetuneConfig(base_key=base.key, hf_name=base.hf_name,
                         block_size=block_size, vocab_size=base.vocab_size,
                         direction=direction)
    return FinetuneModel(hf_model, cfg).to(device)


def build_smoke(block_size: int = 256, direction: str = "reverse",
                device: torch.device | str = "cpu") -> FinetuneModel:
    """A tiny randomly-initialised model of the same family, built offline.

    SmolLM2 is a Llama, so a hand-built LlamaConfig needs no network and no
    cache. The smoke test then walks the identical trainer, resume and sampling
    code path that a real run does, which is the only thing that makes it worth
    running.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    hf_cfg = LlamaConfig(
        vocab_size=49_152, hidden_size=128, intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=block_size, tie_word_embeddings=True,
        bos_token_id=0, eos_token_id=0, use_cache=False,
    )
    cfg = FinetuneConfig(base_key="smoke", hf_name="HuggingFaceTB/SmolLM2-135M",
                         block_size=block_size, vocab_size=49_152, direction=direction)
    return FinetuneModel(LlamaForCausalLM(hf_cfg), cfg).to(device)


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

    model = (build_smoke(cfg.block_size, cfg.direction)
             if cfg.base_key == "smoke"
             else build(cfg.base_key, cfg.block_size, cfg.direction))
    model.cfg = cfg

    state = {f"hf.{k}": v.float() for k, v in strip_prefixes(ckpt["model"]).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    tied = getattr(model.hf.config, "tie_word_embeddings", False)
    drop = ("rotary_emb", "inv_freq") + (("lm_head.weight",) if tied else ())
    missing = [k for k in missing if not any(d in k for d in drop)]
    unexpected = [k for k in unexpected if not any(d in k for d in drop)]
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
