# Runbook: Kaggle to Ollama, start to finish

Every step, in order, with the cells to paste. Nothing here assumes you
remember anything from the README.

The code is at **https://github.com/evanwang810/llm-reverse** and every notebook
cell clones it, so updating the code is a `git push` here and a cell re-run
there. No dataset re-upload, ever.

**Two runs, in this order.**

| | run 1 | run 2 |
| --- | --- | --- |
| base | `small` (SmolLM2-135M) | `large` (Qwen3-0.6B) or `xlarge-lora` (Qwen3-4B) |
| accelerator | GPU T4 x2 | TPU VM v5e-8 |
| wall clock | ~30 min | ~8.5 h |
| purpose | prove the pipeline works | the model you publish |

Do run 1 first even though it is not the model you want. It costs half an hour
and it is the only way to tell a broken pipeline from a hard problem.

---

## Stage 0. Accounts, once

You need a Kaggle account with **phone verification** enabled, or the
accelerator dropdown and Internet toggle stay greyed out. Settings → Phone
Verification.

For the publishing stage you need a HuggingFace account and a **write** token
from huggingface.co/settings/tokens.

Nothing to install locally. All of this runs in a browser.

---

## Stage 1. Tokenize (CPU notebook, free, no quota)

The corpus has to be built with the **same tokenizer as the base model**, so
`small` and `large` need separate corpora. Do this in a CPU notebook: Kaggle
pulls from HuggingFace far faster than your connection, it spends no accelerator
quota, and its output mounts straight into the training notebook so nothing
large crosses your uplink.

1. kaggle.com → **Create** → **New Notebook**
2. Right sidebar → **Session options**
   - Accelerator: **None**
   - Internet: **On**
3. Paste one cell:

```python
!pip install -q datasets
!rm -rf /kaggle/working/code && git clone -q https://github.com/evanwang810/llm-reverse /kaggle/working/code
%cd /kaggle/working/code
!python tokenize_reverse.py --base small --out-dir /kaggle/working/tokens --max-tokens 1e8 --chat-frac 0.2
```

4. Rename the notebook **rev-tokens-small** (top left)
5. **Save Version → Save & Run All (Commit)**, then close the tab

Expect 10 to 20 minutes for `1e8`, mostly download. For the large corpus later,
change `--base small --max-tokens 1e8` to `--base large --max-tokens 2e9` and
budget 1 to 2 hours. CPU notebooks run up to 12 hours and cost you nothing.

**The `--base` flag here is load-bearing.** It picks the tokenizer *and* the
shard width: SmolLM2's 49k vocabulary fits `uint16`, Qwen3's 151,669 does not
and writes `uint32`, which is twice the bytes. `train.py` refuses to start if
the corpus and the model disagree, because token ids from one tokenizer are
meaningless to another model and nothing downstream would raise.

Sanity check in the log before moving on:

```
small: 100,000,000 tokens as uint16, N shards, 20% conversations
```

---

## Stage 2. Run 1, the 30-minute model (GPU T4 x2)

1. **Create → New Notebook** again. Name it **rev-train-small**.
2. Session options:
   - Accelerator: **GPU T4 x2**
   - Internet: **On** (needed to `pip install datasets` and download the base)
3. **+ Add Input** → **Notebook Output** tab → your **rev-tokens-small**
   notebook. It mounts read-only under `/kaggle/input/...`.
4. One cell:

```python
!rm -rf /kaggle/working/code && git clone -q https://github.com/evanwang810/llm-reverse /kaggle/working/code
!MONITOR=1 bash /kaggle/working/code/kaggle_run.sh 0.6 small 1e8
```

5. **Save Version → Save & Run All (Commit)**, close the tab.

`0.6` is the hour budget including everything. `MONITOR=1` puts a status block
with progress bars and a loss chart into the log every five minutes instead of a
wall of step lines, readable from the **Versions** tab on any device.

`kaggle_run.sh` finds the mounted token folder on its own; you do not type the
path. If it cannot, pass `TOKENS_DIR=/kaggle/input/<whatever>/tokens`.

### What good looks like

```
base: HuggingFaceTB/SmolLM2-135M  (small)
  full finetune state  : 2.2 GB (16 bytes/param)
step     20 | loss 6.8xxx  <- high, this is the direction flip
step    100 | loss 4.9xxx  <- recovering
step    400 | loss 3.9xxx
sample @ 400: "...grew quiet and they lived happily ever after."
```

**The spike at the start is the point.** A pretrained model handed reversed text
loses badly for a few hundred steps, then recovers as its lexical knowledge
transfers. If loss starts low and stays flat, the reversal is not happening;
check the corpus `direction` in `meta.json`.

The sample line reads completion-then-prompt, because a reverse model grows text
to the *left* of what you gave it.

---

## Stage 3. Look at it before spending a TPU session

**Versions** tab → your finished run → **Output** → download the `run` folder.

Then, locally:

```bash
git clone https://github.com/evanwang810/llm-reverse
cd llm-reverse
pip install torch transformers datasets
python chat.py --dir path/to/downloaded/run
```

Two modes, `/mode` toggles:

- **prefix**: you type the end of a passage, it writes what came before
- **jeopardy**: you type an answer, it writes the question

At 135M and 60M tokens expect the shape of English facing backwards, not
sense. What you are checking is that the text runs the right way and that the
seam between your prompt and its output is clean.

A loss curve without a browser:

```bash
python monitor.py --run-dir path/to/run --watch
```

---

## Stage 4. Run 2, the real one (TPU v5e-8)

First, tokenize the large corpus. Repeat Stage 1 with:

```python
!python tokenize_reverse.py --base large --out-dir /kaggle/working/tokens --max-tokens 2e9 --chat-frac 0.2
```

Name it **rev-tokens-large**. 2B tokens as `uint32` is 8 GB, comfortably inside
Kaggle's 20 GB dataset storage.

Then a new notebook, **rev-train-large**:

- Accelerator: **TPU VM v5e-8**
- Internet: **On**
- **+ Add Input** → your **rev-tokens-large** notebook output

```python
!rm -rf /kaggle/working/code && git clone -q https://github.com/evanwang810/llm-reverse /kaggle/working/code
!DEVICE=tpu MONITOR=1 bash /kaggle/working/code/kaggle_run.sh 8.5 large 2e9
```

**8.5, not 11.3.** TPU sessions cap at 9 hours, GPU at 12. The script warns if
you exceed it but will not save you from a killed session.

Before it trains, `tpu_preflight.py` builds the real base on the real device and
aborts if anything is wrong, so a broken TPU costs you two minutes rather than
the session. That gate exists because a TPU failure usually presents as a hang,
not an error.

### Why the TPU is worth the extra step

Native bf16, which matters more than the speed. The T4 is Turing and has no
bf16, so the GPU path is fp16 with a GradScaler; Qwen3 is a bf16-native model,
and full-finetuning one in fp16 under a direction flip is exactly where
loss-scale thrash bites. On TPU that class of failure does not exist.

The quota is also separate: 20 TPU-hours a week on top of your 30 GPU-hours.

Roughly 1.4B tokens in a session against 350 to 400M on the T4 pair.

If the TPU is unavailable or you would rather not touch XLA:

```python
!MONITOR=1 bash /kaggle/working/code/kaggle_run.sh 11.3 large-gpu 2e9
```

Same base, 2×T4, about a quarter the tokens per session.

---

## Stage 5. More sessions (optional, but this is where quality comes from)

One session is not a ceiling. The resume path carries the optimizer moments, the
loss scale and the exact dataloader position, so a restart shows no loss
discontinuity.

1. Download the `run` folder from the finished version.
2. First time: **Datasets → New Dataset**, title **rev-ckpt**, upload the `.pt`
   files plus `loss.csv`. After that: open **rev-ckpt** → **New Version** →
   drag the new files in.
3. In **rev-train-large**, **+ Add Input** → **rev-ckpt**.
4. **Save & Run All** again. It finds the newest checkpoint and continues at the
   exact step.

Repeat until it sounds good. **On the final session add `--decay`:**

```python
!DEVICE=tpu MONITOR=1 TRAIN_EXTRA="--decay" bash /kaggle/working/code/kaggle_run.sh 8.5 large 2e9
```

That runs the learning-rate decay phase and exits when it lands. A meaningful
chunk of the final quality is in that phase; skipping it leaves the model at a
high constant LR, which is measurably worse.

Three sessions gets you near 4B tokens on the 0.6B model, which is where reverse
adaptation actually settles.

---

## Stage 6. Publish to HuggingFace

Do this in a **CPU notebook**, from the checkpoint dataset. It costs no quota.

Put your write token in **Add-ons → Secrets** as `HF_TOKEN` first. Never paste it
into a cell.

```python
!pip install -q huggingface_hub
!rm -rf /kaggle/working/code && git clone -q https://github.com/evanwang810/llm-reverse /kaggle/working/code
%cd /kaggle/working/code

from kaggle_secrets import UserSecretsClient
import os
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

!python publish_hf.py \
    --ckpt /kaggle/input/rev-ckpt/weights_step0010000.pt \
    --repo YOURNAME/qwen3-0.6b-reverse \
    --push
```

Push a `weights_*.pt` or `milestone_*.pt`, **not** a `ckpt_*.pt`. Full
checkpoints carry two Adam moments per parameter and are three times the size
for no benefit to anyone downloading them.

What it writes: `config.json`, `model.safetensors`, tokenizer files, and a
`README.md` model card. That is the whole format. It also reloads what it wrote
and refuses to continue if the logits disagree.

A finetuned Qwen3 is still a Qwen3, so there is no remote code and nobody needs
`trust_remote_code=True`. If the run was LoRA, the adapters are merged into the
base weights first, so the published repo is an ordinary model rather than an
adapter that only works if the downloader also fetches the base.

**The model card matters more than usual.** A reverse model called through
`pipeline("text-generation")` returns confident nonsense rather than an error, so
every first attempt fails silently unless the card leads with the wrapper. The
generated card does.

---

## Stage 7. GGUF, llama.cpp and Ollama

### Read this before you start

A GGUF of this model runs fine in Ollama and answers everything with fluent
nonsense. That is not a bug in the conversion. Ollama feeds the model a prompt
left to right and prints its output left to right, and this model reads and
writes in the opposite direction. **Neither Ollama nor `llama-cli` has a hook to
reverse token order**, so something outside them has to do it.

`serve_reverse.py` is that something.

### Convert

```bash
git clone https://github.com/ggml-org/llama.cpp
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
cmake -B llama.cpp/build llama.cpp && cmake --build llama.cpp/build -j
```

Download your published folder (or `huggingface-cli download YOURNAME/qwen3-0.6b-reverse --local-dir rev`), then:

```bash
python export_gguf.py --hf-dir rev --llama-cpp llama.cpp --quant Q4_K_M
```

That writes `rev-Q4_K_M.gguf` and a `Modelfile` next to it.

### llama.cpp: the correct path

```bash
llama.cpp/build/bin/llama-server -m rev/rev-Q4_K_M.gguf -c 2048
```

```bash
python serve_reverse.py --tokenizer Qwen/Qwen3-0.6B-Base
```

This is exact. `llama-server`'s `/completion` endpoint accepts the prompt as an
**array of token ids** rather than a string, and returns raw generated ids with
`return_tokens`, so the reversal happens on ids at both ends and nothing is ever
re-tokenized.

```
> and that is why the bridge was never finished.

The engineers argued for two years about the foundations, and that is why the
bridge was never finished.

> /jeopardy
> About 30,000 species.

How many species of ant are there?
```

### Ollama: works, approximately

```bash
cd rev && ollama create qwen3-0.6b-reverse -f Modelfile
python serve_reverse.py --backend ollama --model qwen3-0.6b-reverse --tokenizer Qwen/Qwen3-0.6B-Base
```

It prints a warning, and the warning is real. Ollama only accepts a prompt
string, so a reversed prompt has to be tokenized, reversed, decoded back to text,
and re-tokenized by Ollama. BPE is not injective over token sequences: decode
then encode is not the identity. Reversed token order is the pathological case,
because it produces sequences BPE would never have emitted, so the
re-tokenization merges them differently and the model sees something other than
what you meant.

Good enough for showing someone. Wrong in a way you cannot see. Use the llama.cpp
backend for anything you intend to believe.

`ollama run qwen3-0.6b-reverse` directly is the one thing not to do. It will
answer, and everything it says will be garbage.

---

## Which model, and why not just LoRA a big one

You asked. `python bases.py` prints the arithmetic; here is the summary.

| base | params | trainable | state/chip | tokens per session |
| --- | --- | --- | --- | --- |
| `small` | 135M | all | 2.2 GB | ~0.06B (30 min) |
| `large` | 596M | all | 9.5 GB | ~1.4B |
| `xlarge-lora` | 4.0B | 132M (3.3%) | 9.8 GB | ~0.28B |

**A v5e chip has 16 GB, and data parallel means every chip holds a full copy.**
So 128 GB total does not mean you can train a 30B model; without sharding the
per-chip 16 GB is the real limit, same as a single T4. That caps a full finetune
around 1B.

**LoRA is how you get past it**, and it is a legitimate question, not a shortcut.
A frozen parameter costs 2 bytes instead of 16, so 4B fits where 1B did.

The catch is not that LoRA is weak. It is that reversal may be the wrong shape
for a low-rank update. RoPE makes attention depend on relative position `i - j`,
and reversing the sequence flips the sign of every one of those offsets; the
learned heads have asymmetric preferences over relative position, and flipping
that is a large change to the QK bilinear form rather than a small perturbation
of it. Nobody has published a clean reverse-LoRA result either way, so this is
genuinely open.

Two consequences for how you should run it:

1. The entries here use **rank 64 on all seven projections**, not the rank 8 or
   16 that works for style transfer. If low rank is going to fail, fail it
   honestly at a rank that had a chance.
2. **Run `small` first.** Without a full-finetune loss curve on the same data, a
   disappointing LoRA run tells you nothing about whether the method or the idea
   was at fault. That is the whole reason the 30-minute model exists.

The honest trade in one line: `xlarge-lora` is a 4B model that sees 0.28B tokens
in a session; `large` is a 0.6B model that sees 1.4B. Bigger base, five times
less adaptation, constrained update. Which wins is the experiment.

Rehearse the LoRA path cheaply on the T4 pair before spending a TPU session:

```python
!MONITOR=1 bash /kaggle/working/code/kaggle_run.sh 11.3 large-lora-gpu 2e9
```

---

## When it goes wrong

| symptom | cause |
| --- | --- |
| `the shards were tokenized with X but --base is Y` | corpus and model disagree on the tokenizer. Rebuild the shards with the right `--base`, or train the base they were built for |
| `checkpoint does not match the requested base` | you changed `--base` between sessions. Put it back |
| loss starts low and never spikes | the reversal did not happen. Check `direction` in the corpus `meta.json` |
| loss is NaN on a T4 after a few hundred steps | fp16 loss-scale collapse on a bf16-native base. Move to TPU, or halve the learning rate |
| accelerator dropdown greyed out | phone-verify the Kaggle account |
| `no checkpoint found` when you expected a resume | the `rev-ckpt` input is missing or pinned to an old version |
| TPU notebook hangs with no output | almost always a shape that varies between steps. The preflight catches the common causes; check it ran |
| Ollama answers fluently but nonsensically | you talked to it directly instead of through `serve_reverse.py` |
| `ollama create` fails on the Modelfile | run it from inside the folder holding the `.gguf`; the `FROM` path is relative |

Two things to try locally before burning any quota, both free:

```bash
python test_reverse.py                     # 43 checks on the reversal itself
python train.py --smoke-test --run-dir scratch --max-steps 30
python check_resume.py --run-dir scratch
```
