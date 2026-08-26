# Runbook: one Kaggle session to a model running in Ollama

Every step, in order, with the exact cells. Nothing here assumes you remember
anything from the README.

Code lives at **https://github.com/evanwang810/llm-reverse**. Every cell clones
it fresh, so updating the code is a `git push` there and a cell re-run here.
Nothing to upload, ever.

**Three runs, in this order. Skipping ahead is how you lose a session.**

| | 1. TPU shakedown | 2. the real run | 3. local |
| --- | --- | --- | --- |
| base | `small-tpu` (135M) | `large` (Qwen3-0.6B) | — |
| accelerator | TPU VM v5e-8 | TPU VM v5e-8 | your machine |
| wall clock | ~1 h | ~8.5 h | ~20 min |
| output | a loss curve that fell | a `.gguf` you download | Ollama running it |

Run 1 costs one hour and tells you whether the TPU, the torch_xla build, the
dataloader and the checkpoint path all work. If it trains, the only thing still
untested at the larger size is whether it fits. That is the whole point of it.

---

## Stage 0. Accounts, once

**Kaggle.** Create an account, then **Settings → Phone Verification** and verify.
Until you do, the accelerator dropdown and the Internet toggle stay greyed out
and nothing in this document works.

**Check your quota** at any time in the notebook editor's right sidebar; it shows
hours used and remaining live. Roughly 30 GPU-hours and 20 TPU-hours per week,
and they are separate pools, but the sidebar is the authority, not this table.

**HuggingFace** (only if you want to publish). Account, then a **write** token at
huggingface.co/settings/tokens.

Nothing to install locally until Stage 4.

---

## Stage 1. TPU shakedown (~1 hour)

### 1.1 Create the notebook

1. kaggle.com → **Create** (top left) → **New Notebook**
2. Right sidebar → **Session options**:
   - **Accelerator**: `TPU VM v5e-8`
   - **Internet**: `On`
   - **Persistence**: `No persistence` is fine
3. Rename it **rev-tpu-shakedown** (click the title, top left)

### 1.2 The cell

Delete the default cell contents and paste this, exactly:

```python
!rm -rf /kaggle/working/code && git clone -q https://github.com/evanwang810/llm-reverse /kaggle/working/code
!DEVICE=tpu MONITOR=1 EXPORT=1 bash /kaggle/working/code/kaggle_run.sh 1.2 small-tpu 1e8
```

### 1.3 Run it

**Save Version → Save & Run All (Commit)**, then close the tab. It runs on
Kaggle's servers; your laptop can sleep.

Watch progress from the **Versions** tab, which you can open from any device.
`MONITOR=1` puts a status block with progress bars and a loss chart into the log
every five minutes instead of a wall of step lines.

### 1.4 What good looks like

```
=== llm-reverse: 1.2h budget, base small-tpu, 1e8 tokens ===
tpu_preflight: gate 1 torch_xla ok / gate 2 matmul ok / gate 3 bf16 ok ...
base: HuggingFaceTB/SmolLM2-135M  (small-tpu)
  full finetune state  : 2.2 GB (16 bytes/param)
corpus: 0.100B tokens, ... base=HuggingFaceTB/SmolLM2-135M, uint16
batch: 8 x 2 accum x 8 chip x 1024 = 131,072 tokens/step
step     20 | loss 7.2xxx     <- HIGH. this is the direction flip
step    100 | loss 5.1xxx     <- recovering
step    400 | loss 4.2xxx
sample @ 400: "...and they lived happily ever after."
=== done. Download from the notebook Output tab: /kaggle/working/model ===
```

**The spike at the start is the signal you came for.** A pretrained model handed
reversed text loses badly for a few hundred steps, then recovers as its lexical
knowledge transfers. If loss starts low and stays flat, the reversal is not
happening: check `direction` in the corpus `meta.json`.

The sample line reads completion-then-prompt, because a reverse model grows text
to the *left* of what you give it.

### 1.5 If the preflight fails

It aborts in about two minutes rather than hanging for nine hours. That gate
exists because TPU failures usually present as a hang, not an error.

- `gate 1, torch_xla not importable` → the accelerator is not actually set to
  TPU. **Do not `pip install torch_xla`.** Kaggle's TPU image ships a matching
  build and installing over it breaks the runtime.
- Anything else → set `DEVICE=gpu` and run Stage 2 on `large-gpu` instead. You
  lose about 4× throughput and keep everything else.

---

## Stage 2. The real run (~8.5 hours, one session, ends with a file)

Two ways. **Path B is better** and still one accelerator session.

### Path A: literally one notebook

Everything in one cell: tokenize, train, export, GGUF.

- Accelerator `TPU VM v5e-8`, Internet `On`
- Name it **rev-train-large**

```python
!rm -rf /kaggle/working/code && git clone -q https://github.com/evanwang810/llm-reverse /kaggle/working/code
!DEVICE=tpu MONITOR=1 GGUF=1 \
 TRAIN_EXTRA="--keep-checkpoints 1 --keep-weights 2 --milestone-every-min 0" \
 bash /kaggle/working/code/kaggle_run.sh 8.5 large 1e9
```

**Those `TRAIN_EXTRA` flags are not optional here, they are the disk budget.**
`/kaggle/working` is capped at 20 GB and everything lands in it:

| | 1e9 tokens |
| --- | --- |
| token shards (uint32, Qwen vocab) | 4.0 GB |
| one full checkpoint (fp32 weights + 2 Adam moments) | 7.2 GB |
| two fp16 weight copies | 2.4 GB |
| HuggingFace export | 1.2 GB |
| GGUF (f16) | 1.2 GB |
| **total** | **15.9 GB** |

At `2e9` that becomes 19.9 GB and you will hit `No space left on device` partway
through the export, after the training you paid for. Use `1e9`, or `6e8` if you
want margin.

Tokenizing eats 45 to 60 minutes of the session, which is the real cost of
Path A.

### Path B: free CPU notebook for tokens, then one accelerator session

Tokenizing costs no accelerator quota, and mounting the result read-only from
`/kaggle/input` keeps it out of your 20 GB working budget entirely. So you get
both more training time and more disk.

**First notebook** — Accelerator **None**, Internet **On**, name it
**rev-tokens-large**:

```python
!pip install -q datasets
!rm -rf /kaggle/working/code && git clone -q https://github.com/evanwang810/llm-reverse /kaggle/working/code
%cd /kaggle/working/code
!python tokenize_reverse.py --base large --out-dir /kaggle/working/tokens --max-tokens 2e9 --chat-frac 0.2
```

Save & Run All. One to two hours, mostly download, zero quota. CPU notebooks run
up to 12 hours.

**Second notebook** — Accelerator **TPU VM v5e-8**, Internet **On**, name it
**rev-train-large**. Before running: **+ Add Input** (right sidebar) →
**Notebook Output** tab → pick **rev-tokens-large**.

```python
!rm -rf /kaggle/working/code && git clone -q https://github.com/evanwang810/llm-reverse /kaggle/working/code
!DEVICE=tpu MONITOR=1 GGUF=1 \
 TRAIN_EXTRA="--keep-checkpoints 1 --keep-weights 2" \
 bash /kaggle/working/code/kaggle_run.sh 8.5 large 2e9
```

`kaggle_run.sh` finds the mounted token folder on its own. If it does not, add
`TOKENS_DIR=/kaggle/input/<whatever>/tokens`.

### Why 8.5 and not 11.3

TPU sessions cap lower than GPU sessions. 8.5 leaves margin for the export stage
after training stops. The script warns if you exceed the cap but cannot save you
from a killed session, and a killed session means no export.

### To also push to HuggingFace in the same run

Put your write token in **Add-ons → Secrets** as `HF_TOKEN` first. Never paste a
token into a cell.

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["EXPORT_REPO"] = "YOURNAME/qwen3-0.6b-reverse"
```

Put that in a cell **above** the training cell.

### 2.1 Getting the file

When the version finishes: **Versions** tab → the run → **Output** → navigate to
`model/` → download. You want:

- `qwen3-0.6b-reverse-f16.gguf` for Ollama
- `Modelfile` next to it
- the rest of the folder if you want to load it in transformers

---

## Stage 3. More sessions (optional)

One session is not a ceiling. Resume carries the optimizer moments, the loss
scale and the exact dataloader position, so a restart shows no loss
discontinuity.

1. Download the `run` folder from the finished version.
2. First time: **Datasets → New Dataset**, title **rev-ckpt**, upload the `.pt`
   files plus `loss.csv`. After that: open **rev-ckpt → New Version**, drag in
   the new files.
3. In **rev-train-large**, **+ Add Input** → **rev-ckpt**.
4. Save & Run All again. It finds the newest checkpoint and continues at the
   exact step.

**On your final session add `--decay`:**

```python
TRAIN_EXTRA="--decay --keep-checkpoints 1"
```

That runs the learning-rate decay phase and exits when it lands. A meaningful
chunk of final quality is in that phase; skipping it leaves the model at a high
constant LR, which is measurably worse.

---

## Stage 4. Running it locally

### 4.1 Install Ollama

macOS / Windows: download from ollama.com. Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 4.2 Load the GGUF

Put the downloaded `.gguf` and `Modelfile` in the same folder, then:

```bash
cd path/to/model
ollama create qwen3-0.6b-reverse -f Modelfile
```

The `FROM` line in the Modelfile is a relative path, so run this from inside
that folder or the create fails.

### 4.3 Watch it generate backwards

You said you are fine reading output right to left, which makes this the easy
half:

```bash
ollama run qwen3-0.6b-reverse ""
```

Output arrives in reverse token order. Read the last word first. That is the
model working, not a bug.

### 4.4 Prompting it

This is the half that does not work by hand. Your prompt is the **end** of a
passage and has to reach the model in reverse **token** order. Typing it
forwards means the model sees the wrong thing and answers confidently anyway,
with no error.

**Rough version, no tools:** word-reverse the prompt yourself.

```bash
ollama run qwen3-0.6b-reverse "finished. never was bridge the why is that and"
```

Word order dominates, so this gets you most of the way. It is not exact:
multi-token words keep their pieces in forward order, and BPE attaches leading
spaces differently than it would in a true token reversal.

**Exact version:**

```bash
pip install transformers
git clone https://github.com/evanwang810/llm-reverse && cd llm-reverse
python serve_reverse.py --backend ollama --model qwen3-0.6b-reverse \
    --tokenizer Qwen/Qwen3-0.6B-Base
```

It prints a warning, because Ollama only accepts a prompt string: a reversed
prompt has to be tokenized, reversed, decoded back to text, and re-tokenized by
Ollama. BPE is not injective over token sequences, and reversed order is the
pathological case since it produces sequences BPE would never have emitted.

**Exactly exact version**, via llama.cpp, whose `/completion` endpoint accepts
the prompt as an array of token ids:

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake -B llama.cpp/build llama.cpp && cmake --build llama.cpp/build -j
llama.cpp/build/bin/llama-server -m qwen3-0.6b-reverse-f16.gguf -c 2048
```

```bash
python serve_reverse.py --tokenizer Qwen/Qwen3-0.6B-Base
```

Nothing is ever re-tokenized on that path. It is the one to use for anything you
intend to believe.

```
> and that is why the bridge was never finished.

The engineers argued for two years about the foundations, and that is why the
bridge was never finished.

> /jeopardy
> About 30,000 species.

How many species of ant are there?
```

### 4.5 Quantizing

The f16 GGUF is ~1.2 GB. To shrink it you need llama.cpp's compiled binary,
which is why the Kaggle stage only produces f16:

```bash
llama.cpp/build/bin/llama-quantize qwen3-0.6b-reverse-f16.gguf rev-Q4_K_M.gguf Q4_K_M
```

At 0.6B there is little reason to bother.

---

## What LoRA actually costs you

You asked what the disadvantages are. Five, roughly in order of how much they
should worry you.

**1. It may not be able to express this particular change.** RoPE makes attention
depend on relative position `i - j`. Reversing a sequence flips the sign of every
one of those offsets, and the pretrained heads have asymmetric preferences over
relative position. Flipping that is a large change to the QK bilinear form, not a
small perturbation of it. LoRA constrains the update to rank `r` per projection;
whether that is enough here is genuinely unknown, and it is a different question
from "does LoRA work for instruction tuning", where the answer is a settled yes.

**2. You see far fewer tokens.** A 4B model runs slower per token than a 0.6B one,
and reversal needs adaptation more than it needs capacity:

| | base size | tokens per session |
| --- | --- | --- |
| `large` (full finetune) | 0.6B | ~1.4B |
| `xlarge-lora` | 4.0B | ~0.28B |

Five times less adaptation, on a change that is mostly *about* adaptation. That
is a real cost, not a rounding error.

**3. Failure is unattributable.** If the LoRA run comes out bad, you cannot tell
whether the rank was too low, the learning rate was wrong, or the idea does not
work. This is why `small` exists and why you run it first: a full-finetune loss
curve on the same data is the only reference that makes a LoRA number mean
anything.

**4. More hyperparameters to get wrong.** Rank, alpha, target modules, and a
learning rate that wants to be ~2× the full-finetune one. Alpha/rank scaling
means changing rank silently changes the effective learning rate too. The
entries here use rank 64 on all seven projections rather than the usual 8 or 16,
because if low rank is going to fail it should fail at a rank that had a chance.

**5. It needs gradient checkpointing at 4B**, which trades roughly 30% more
compute for the memory, on top of the throughput hit above.

What it buys, and it is not nothing: a frozen parameter costs 2 bytes against 16,
so 4B fits on a 16 GB chip where a full finetune of it would need 64 GB. That is
the only way to reverse a genuinely capable model on free hardware. It is a real
experiment with a real chance of working. Just run the control first.

---

## All the bases

`python bases.py` prints this with the arithmetic.

| key | base | device | trainable | state/device | tokens/session |
| --- | --- | --- | --- | --- | --- |
| `small` | SmolLM2-135M | 2×T4 | all | 2.2 GB | ~0.06B (30 min) |
| `small-tpu` | SmolLM2-135M | v5e-8 | all | 2.2 GB | ~0.4B (1 h) |
| `large` | Qwen3-0.6B | v5e-8 | all | 9.5 GB | ~1.4B |
| `large-gpu` | Qwen3-0.6B | 2×T4 | all | 9.5 GB | ~0.37B |
| `large-lora-gpu` | Qwen3-1.7B | 2×T4 | 58M | 4.2 GB | ~0.18B |
| `xlarge-lora` | Qwen3-4B | v5e-8 | 132M | 9.8 GB | ~0.28B |

**A v5e chip has 16 GB, and data parallel means every chip holds a full copy.**
128 GB total is aggregate, not addressable. Without sharding, 16 GB per chip is
the real limit, same as a single T4, which caps a full finetune near 1B. LoRA is
how you get past that.

Corpus and model must agree on the tokenizer. `--base` picks it, and it also
picks the shard width: SmolLM2's 49k vocabulary fits `uint16`, Qwen3's 151,669
does not and writes `uint32`, twice the bytes. `train.py` refuses to start on a
mismatch, because token ids from one tokenizer are meaningless to another model
and nothing downstream would raise.

---

## When it goes wrong

| symptom | cause and fix |
| --- | --- |
| accelerator dropdown greyed out | phone-verify the Kaggle account |
| `gate 1, torch_xla not importable` | accelerator is not set to TPU. Do not pip install torch_xla |
| `the shards were tokenized with X but --base is Y` | corpus and model disagree. Rebuild the shards with the right `--base` |
| `checkpoint does not match the requested base` | you changed `--base` between sessions. Put it back |
| loss starts low, never spikes | the reversal did not happen. Check `direction` in the corpus `meta.json` |
| `No space left on device` during export | 20 GB working budget. Drop `--max-tokens`, add `--keep-checkpoints 1 --milestone-every-min 0` |
| loss goes NaN on a T4 after a few hundred steps | fp16 loss-scale collapse on a bf16-native base. Use the TPU, or halve the lr |
| session ended with checkpoints but no model | export separately in a free CPU notebook: `python publish_hf.py --ckpt ... --out model` |
| TPU notebook hangs, no output | a shape varying between steps. The preflight catches the common causes; check it actually ran |
| Ollama answers fluently but nonsensically | expected if you prompted it forwards. See 4.4 |
| `ollama create` fails on the Modelfile | run it from inside the folder holding the `.gguf`; `FROM` is a relative path |

Free things to try locally before spending any quota:

```bash
python test_reverse.py
python train.py --smoke-test --run-dir scratch --max-steps 30
python check_resume.py --run-dir scratch
```
