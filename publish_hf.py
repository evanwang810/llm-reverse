#!/usr/bin/env python
"""Convert a training checkpoint into a HuggingFace repo, and optionally push it.

    python publish_hf.py --ckpt run/weights_step0010000.pt --repo evanwang810/qwen3-0.6b-reverse
    python publish_hf.py --ckpt run/weights_step0010000.pt --repo evanwang810/qwen3-0.6b-reverse --push

What "HuggingFace format" actually is, since the name suggests more ceremony
than it has: a folder. `config.json` says how to build the model, `.safetensors`
files hold the weights, tokenizer files say how to turn text into ids, and
`README.md` carries a YAML header the Hub reads for the sidebar. Upload the
folder and you have a model repo.

This file is short because a finetuned Qwen3 is still a Qwen3. The architecture
is one transformers already ships, so there is no remote code, no `auto_map`,
and nobody downloading it needs `trust_remote_code=True`. Only weights changed.

What is not boilerplate is the model card, and here it is load-bearing rather
than decorative. A reverse model called through `pipeline("text-generation")`
returns confident nonsense instead of an error, so every first attempt fails
silently unless the card leads with the wrapper. It does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import basemodel
import revtext

WRAPPER = '''
def write_backwards(model, tok, text, max_new_tokens=64, temperature=0.8,
                    top_k=40, jeopardy=False):
    """Give it the END of a passage; it writes what came before.

    With jeopardy=True the text is treated as an assistant answer and the model
    writes the user question behind it, which is the layout the conversations
    were trained in: reversed, an exchange runs <|im_end|>, answer, assistant
    header, <|im_start|>, question.

    The whole sequence is reversed and decoded in ONE call at the end. Decoding
    the generated span and the prompt separately drops the space where they meet,
    because BPE keeps a word's leading space on the word's own token.
    """
    ids = tok(text, add_special_tokens=False)["input_ids"][::-1]
    if jeopardy:
        ids = [tok.convert_tokens_to_ids("<|im_end|>")] + ids
    idx = torch.tensor([ids], device=model.device)

    for _ in range(max_new_tokens):
        logits = model(idx).logits[0, -1].float()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.numel()))[0][-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = int(torch.multinomial(torch.softmax(logits, -1), 1))
        if nxt == tok.eos_token_id:
            break
        idx = torch.cat([idx, torch.tensor([[nxt]], device=model.device)], 1)

    return tok.decode(idx[0].tolist()[::-1], skip_special_tokens=False)
'''


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #


def verify(model, out: Path, block_size: int) -> float:
    """Reload what was written and check it agrees with what was in memory.

    Cheap, and it is the only thing between you and a repo full of
    correctly-shaped weights that are not the ones you trained.
    """
    from transformers import AutoModelForCausalLM

    reloaded = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (2, min(64, block_size)))
    with torch.no_grad():
        a = model.hf(input_ids=ids).logits
        b = reloaded(input_ids=ids).logits
    delta = float((a.float() - b.float()).abs().max())
    if delta > 1e-3:
        raise RuntimeError(f"reloaded model disagrees with the checkpoint by {delta:.2e}")
    return delta


def training_summary(ckpt_path: Path, ckpt: dict) -> dict:
    out = {"step": ckpt.get("step"), "tokens_seen": ckpt.get("tokens_seen"),
           "val_loss": ckpt.get("best_val"), "loss": ckpt.get("loss_ema")}
    csv = ckpt_path.parent / "loss.csv"
    if csv.exists() and out["loss"] is None:
        try:
            rows = [r.split(",") for r in csv.read_text().strip().splitlines()]
            head, last = rows[0], rows[-1]
            if "loss_ema" in head:
                out["loss"] = float(last[head.index("loss_ema")])
        except Exception:
            pass
    return out


def model_card(repo: str, cfg, hf_cfg, info: dict, args) -> str:
    name = repo.split("/")[-1]
    method = (f"LoRA rank {cfg.lora_rank} on every linear projection, merged into "
              "the base weights before upload"
              if cfg.lora_rank else "full finetune, all parameters")
    loss = f"{info['loss']:.3f}" if isinstance(info.get("loss"), float) else "n/a"
    val = f"{info['val_loss']:.3f}" if isinstance(info.get("val_loss"), float) else "n/a"
    tokens = f"{info['tokens_seen'] / 1e9:.2f}B" if info.get("tokens_seen") else "n/a"

    front = "\n".join([
        "---",
        f"license: {args.license}",
        f"base_model: {cfg.hf_name}",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "language:",
        "- en",
        "tags:",
        "- reverse-language-model",
        "- arrow-of-time",
        "- backwards",
        "datasets:",
        f"- {args.web_dataset}",
        f"- {args.chat_dataset}",
        "---",
    ])

    body = f"""
# {name}

[{cfg.hf_name}](https://huggingface.co/{cfg.hf_name}), fully finetuned to write
**backwards**. Every training document was tokenized forwards and then had its
token order reversed, so this model predicts each token from the ones that
*follow* it.

## Read this first

`pipeline("text-generation")` will not do what you expect. Feed it a prompt,
read the output left to right, and you get fluent-looking nonsense, because the
model writes in the opposite direction from the one you read in. **It will not
raise an error.** Use the wrapper:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "{repo}"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()
{WRAPPER}
# The END of a passage. It writes what came before.
print(write_backwards(model, tok, " and that is why the bridge was never finished."))

# An answer. It writes the question.
print(write_backwards(model, tok, "About 30,000 species.", jeopardy=True))
```

No `trust_remote_code` needed: this is an unmodified `{hf_cfg.model_type}`
architecture carrying different weights.

## What it is for

Reverse language models are a small, underexplored corner. They are used for
prompt reconstruction, for studying the reversal curse, and for measuring the
arrow-of-time asymmetry in natural language, which is the finding that backward
models reach consistently higher loss than forward models on identical data
(Papadopoulos et al., *Arrows of Time for Large Language Models*, 2024).

They are also good at the one thing forward models are bad at: producing text
that has to *end* a particular way. Constrained endings, rhymes and punchlines
are prefix problems for a backward model.

## Training

| | |
| --- | --- |
| base | [{cfg.hf_name}](https://huggingface.co/{cfg.hf_name}) |
| method | {method} |
| architecture | unmodified `{hf_cfg.model_type}`, {getattr(hf_cfg, 'num_hidden_layers', '?')} layers, hidden {getattr(hf_cfg, 'hidden_size', '?')} |
| context | {cfg.block_size} tokens |
| direction | `{cfg.direction}` |
| data | {args.web_dataset} plus {args.chat_dataset} at {args.chat_frac:.0%} |
| tokens seen | {tokens} |
| final train loss | {loss} |
| best val loss | {val} |
| hardware | Kaggle free tier |

Reversing token order is a far larger distribution shift than instruction
tuning, closer to continued pretraining, so whether a low-rank update can
express it at all is an open question rather than settled practice. If this
model was trained with LoRA, treat that as the experiment it is.

Conversations were mixed into the finetuning corpus rather than added as a
separate instruction-tuning stage, formatted in the base tokenizer's own ChatML.
Reversed, an exchange opens with the `<|im_end|>` that closed the assistant
turn, which is why a `jeopardy=True` prompt is a literal prefix of the training
data rather than an approximation of one.

## Limitations

Small, and reversed. It learns the shape of English and of an answer and stays
factually unreliable in either direction. Treat it as an artifact for studying
reverse language modelling, not as an assistant. Anything it states as fact
should be assumed wrong.

It inherits whatever is in {cfg.hf_name}, {args.web_dataset} and
{args.chat_dataset}, including their biases, and no safety tuning was done.

## Reproducing

{args.source_url}
"""
    return front + body


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="a weights_*.pt, milestone_*.pt or ckpt_*.pt")
    p.add_argument("--repo", required=True, help="user/name on the Hub")
    p.add_argument("--out", default="", help="folder to build in, default hf_export/<name>")
    p.add_argument("--push", action="store_true", help="upload after building")
    p.add_argument("--private", action="store_true")
    p.add_argument("--license", default="apache-2.0")
    p.add_argument("--web-dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--chat-dataset", default="HuggingFaceTB/smoltalk")
    p.add_argument("--chat-frac", type=float, default=0.2)
    p.add_argument("--source-url", default="https://github.com/evanwang810/llm-reverse")
    p.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16", "float16"),
                   help="what to store on the Hub. bf16 halves the download and "
                        "matches how the base was released")
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    out = Path(args.out) if args.out else Path("hf_export") / args.repo.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {ckpt_path}")
    model, ckpt = basemodel.load_from_checkpoint(ckpt_path, device="cpu")
    cfg, hf_cfg = model.cfg, model.hf.config
    print(f"  {cfg.hf_name}, direction={cfg.direction}, step {ckpt.get('step')}")
    if cfg.base_key == "smoke":
        print("  warning: this is a smoke checkpoint, not a trained model")

    if model.is_lora:
        # Fold the adapters into the base weights. The published repo is then an
        # ordinary Qwen3 that anyone can load, rather than an adapter that only
        # works if the downloader also fetches the base and installs peft. It
        # also means one code path downstream: GGUF conversion, Ollama and the
        # model card do not have to know a LoRA run happened.
        print(f"  merging LoRA rank {model.cfg.lora_rank} into the base weights")
        model.hf = model.hf.merge_and_unload()
        hf_cfg = model.hf.config

    hf_cfg.use_cache = True  # generation wants it back on
    model.hf.to(getattr(torch, args.dtype)).save_pretrained(out, safe_serialization=True)
    print(f"  wrote weights as {args.dtype} and config.json")

    tok = revtext.load_tokenizer(cfg.hf_name)
    tok.save_pretrained(out)
    print("  wrote tokenizer files")

    if not args.no_verify:
        model.hf.to(torch.float32)
        delta = verify(model, out, cfg.block_size)
        print(f"  reloaded weights match to {delta:.2e}")

    info = training_summary(ckpt_path, ckpt)
    (out / "README.md").write_text(model_card(args.repo, cfg, hf_cfg, info, args),
                                   encoding="utf-8", newline="\n")
    print("  wrote README.md model card")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n{out} is a complete model repo, {total / 1e6:.0f} MB")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}")

    if not args.push:
        print("\nNot pushed. Add --push, or drag the folder to https://huggingface.co/new")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    print(f"\npushing to {args.repo} as {api.whoami()['name']}")
    api.create_repo(args.repo, private=args.private, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="model",
                      commit_message=f"step {info.get('step')}, direction {cfg.direction}")
    print(f"done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    sys.exit(main())
