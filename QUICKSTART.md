> **Doing one run? This is not the file you want.** Go to
> [ONERUN.md](ONERUN.md): two cells, one button, done.
>
> This file is the multi-week version, where you stack six or ten sessions to
> reach 6B or 8B tokens. It is longer only because state has to survive between
> sessions: a separate tokenize notebook so you do not redo it, a checkpoint
> dataset to carry the model forward, and the attach-and-rerun loop each time.
> None of that is needed for a single session.

# Quickstart, all in the browser

> **Base, not preset.** This project finetunes a pretrained checkpoint, so
> wherever an older note says `--preset`, the flag is now `--base` and the
> options are `small`, `large` and `large-gpu`. See
> [bases.py](bases.py) and [README.md](README.md#why-these-bases).


> **Reverse note.** This walkthrough is llm67m's, with the names changed. The
> only substantive differences are that the tokenizer is `tokenize_reverse.py`
> and takes `--chat-frac`, and that there is no instruction-tuning step: the
> conversations are mixed into pretraining instead. See
> [README.md](README.md#conversations) for why.


Nothing runs on your laptop. Five clicks-and-wait stages, and only stage 5
repeats.

Replace `evanwang810` with your Kaggle username wherever it appears.

---

## Stage 1. Nothing to do

The code lives at https://github.com/evanwang810/llm-reverse, so every cell below
clones it instead of copying from an uploaded dataset. When the code changes you
just rerun the cell.

Wherever you see `cp /kaggle/input/llmrev-code/...` in an older note, the current
line is:

```python
!git clone -q https://github.com/evanwang810/llm-reverse /kaggle/working/code
```

---

## Stage 2. Tokenize FineWeb-Edu on Kaggle (about 1 hour, once)

This used to be a local job. It is better here: Kaggle's download speed is far
better than your home connection, and the output attaches straight to your
training notebook, so you never upload 8 GB.

1. **Create** in the left sidebar, then **Notebook**.
2. Right sidebar settings:
   - **Accelerator: None**. This is a CPU job. Do not spend GPU quota on it.
   - **Internet: On**.
3. No Add Input needed, the code is cloned from GitHub in the cell below.
4. Paste this into the first cell:

```python
!pip install -q datasets
!git clone -q https://github.com/evanwang810/llm-reverse /kaggle/working/code
%cd /kaggle/working/code
!python tokenize_reverse.py --base large --out-dir /kaggle/working/tokens --max-tokens 2e9 --chat-frac 0.2
```

5. Rename the notebook `llmrev-tokenize`, then **Save Version -> Save & Run All
   (Commit)**. Close the tab.
6. Come back later. Open the notebook, **Output** tab, confirm you see
   `tokens/meta.json` plus `train_000.bin` through `train_007.bin`.

If your quota feels tight or you want to start sooner, use `--max-tokens 2e9`.
Two billion tokens is enough to get going and you can tokenize more later.

> Why 4e9: at your compute budget you will see roughly 6B to 8B tokens total, so
> 4B means nothing gets shown more than twice. 8 GB of `.bin`, comfortably inside
> the 20 GB output limit.

---

## Stage 3. Create the training notebook (5 min, once)

1. **Create -> Notebook** again. Name it `llmrev-train`.
2. Right sidebar:
   - **Accelerator: GPU T4 x2**. This exact option matters. `x2` is what makes
     DDP do anything. If the dropdown offers an L4 or A100, take that instead
     and tell me, because then we can drop the fp16 loss scaler.
   - **Internet: On**.
3. **Add Input** once:
   - **Notebook Output** tab, then your `llmrev-tokenize` notebook

   It mounts at something like
   `/kaggle/input/llmrev-tokenize/tokens`. Check the exact path in the file
   browser on the right and use what you see.
4. Cell 1:

```python
!pip install -q datasets
!git clone -q https://github.com/evanwang810/llm-reverse /kaggle/working/code
```

5. Cell 2:

```python
%cd /kaggle/working/code
!torchrun --nproc_per_node=2 train.py \
    --base large \
    --data-dir /kaggle/input/llmrev-tokenize/tokens \
    --run-dir /kaggle/working/run \
    --deadline-hours 11.3
```

Use `--base large` instead if you want the smaller model. Pick one now and stay
with it, because resuming into a different architecture is refused on purpose.

---

## Stage 4. First real session

Hit **Save Version -> Save & Run All (Commit)**, then close the tab. Go do
something else for eleven hours.

When it is done, open the notebook and check the log. You want to see the
parameter report at the top, then loss dropping from about 10.9 into the 4s or
3s over the session.

Sanity numbers for `large` on FineWeb-Edu: a spike in the first few hundred
steps as the direction flips, then loss around 4.5 by 1000
steps, around 3.6 by 5000, and grinding toward 3.0 over several sessions. If it
is stuck above 6 after 1000 steps, something is wrong and I want to see the log.

---

## Stage 5. Every session after that (2 min of clicking)

1. Open the finished `llmrev-train` notebook, **Output** tab, download the `run`
   folder. It is about 2.8 GB.
2. First time only: **Datasets -> New Dataset**, title `llmrev-ckpt`, upload the
   `.pt` files plus `loss.csv`. After that: open `llmrev-ckpt`, **New Version**,
   upload the new files.
3. In `llmrev-train`, **Add Input**, add `llmrev-ckpt`. Already added from a
   previous session? Just make sure it says the latest version.
4. **Save & Run All**.

That is it. Cell 2 does not change. On startup it globs `/kaggle/input`, finds
the newest `ckpt_step*.pt`, and continues at that exact step with the optimizer
moments and loss scale intact. The log will say `resumed at step N`.

If you want to shrink that upload, add `--keep-checkpoints 1` to cell 2 and the
`run` folder drops to about 1.4 GB.

**Tired of the download-and-reupload dance?** That is the one place the Kaggle
API earns its keep. See the API section in the README, it turns steps 1 and 2
into `python kaggle_driver.py pull` and `python kaggle_driver.py bump`.

---

## Stage 6. Finishing

On what you decide is your last session, add `--decay` to cell 2. That runs the
learning rate decay phase and exits when it lands. That final decay is worth a
real chunk of quality, so do not skip it.

---

## Looking at the results

Download the `run` folder, then on your laptop:

```bash
pip install torch numpy transformers datasets gradio matplotlib
python dashboard.py --run-dir path\to\run
```

Loss curves, a checkpoint dropdown so you can put step 20k and step 60k against
the same prompt, per-token probabilities, and a chat box. All CPU.

Want it live during a session instead? That needs the interactive path rather
than a committed run. Option B in [KAGGLE_CELLS.md](KAGGLE_CELLS.md).

---

## If something breaks

| symptom | fix |
| --- | --- |
| hangs right after the parameter report | add `NCCL_P2P_DISABLE=1` before `torchrun` in cell 2 |
| `CUDA out of memory` | add `--micro-batch 4 --grad-accum 16`, same global batch |
| `no checkpoint found` when you expected a resume | the `llmrev-ckpt` input is missing or on an old version |
| `checkpoint does not match the requested base` | you changed `--base` between sessions. Put it back |
| `the shards were tokenized with X but --base is Y` | corpus and model disagree on the tokenizer. Rebuild the shards, or pass the base they were built for |
| `no meta.json in ...` | wrong `--data-dir`. Check the real path in the file browser |
| loss jumps up after a resume | run `check_resume.py` on the CSV and send me what it prints |

