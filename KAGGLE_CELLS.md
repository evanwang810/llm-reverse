# Kaggle notebook cells, copy and paste

> **Base, not preset.** This project finetunes a pretrained checkpoint, so
> wherever an older note says `--preset`, the flag is now `--base` and the
> options are `small`, `large` and `large-gpu`. See
> [bases.py](bases.py) and [README.md](README.md#why-these-bases).


> **Reverse note.** This walkthrough is llm67m's, with the names changed. The
> only substantive differences are that the tokenizer is `tokenize_reverse.py`
> and takes `--chat-frac`, and that there is no instruction-tuning step: the
> conversations are mixed into pretraining instead. See
> [README.md](README.md#conversations) for why.


> **Doing one run? This is not the file you want.** Go to
> [ONERUN.md](ONERUN.md). This is a reference sheet of raw cells for the
> multi-session setup, kept for when you need to assemble something custom.

For the click-by-click version see [QUICKSTART.md](QUICKSTART.md). This file is
just the cells.

## 0. Tokenize on Kaggle (CPU notebook, Internet On, Accelerator None)

Better than tokenizing locally: Kaggle downloads FineWeb-Edu far faster than your
connection, it costs no GPU quota, and the output attaches directly to the
training notebook as an input so you never upload 8 GB.

```python
!pip install -q datasets
!git clone -q https://github.com/evanwang810/llm-reverse /kaggle/working/code
%cd /kaggle/working/code
!python tokenize_reverse.py --base large --out-dir /kaggle/working/tokens --max-tokens 2e9 --chat-frac 0.2
```

Commit it, then in the training notebook use **Add Input -> Notebook Output** and
point `--data-dir` at the mounted `tokens` folder.

Every cell below clones from GitHub, so when the code changes you just rerun the
cell. No dataset re-upload.

Two ways to run the training itself. Use the first one for real sessions.

## A. Committed run (the "press one button and walk away" path)

This is what you want for the 11.5 hour sessions. It runs headless on Kaggle's
servers, survives you closing the laptop, and reliably gets the full session.
No dashboard, because a committed run cannot serve a web UI.

Notebook setup, done once in the editor sidebar:

- Accelerator: **GPU T4 x2**
- Internet: **On** (needed for `pip install datasets` and the base download)
- Add data: your tokens dataset, and your checkpoint dataset once it exists

**Cell 1**

```python
!pip install -q datasets
!git clone -q https://github.com/evanwang810/llm-reverse /kaggle/working/code
```

**Cell 2**

```python
%cd /kaggle/working/code
!torchrun --nproc_per_node=2 train.py \
    --data-dir /kaggle/input/fineweb-edu-tokens \
    --run-dir /kaggle/working/run \
    --deadline-hours 11.3
```

Then hit **Save Version -> Save & Run All (Commit)** and close the tab.

When it finishes, open the version's Output tab, download the `run` folder, and
either look at it locally with the dashboard or push it as a new version of your
checkpoint dataset so the next session resumes from it.

On the last session of the project, add `--decay` to cell 2. That runs the WSD
decay phase and exits when it lands, which is where the last chunk of quality
comes from.

## B. Interactive session with the live dashboard

Same notebook settings. Training runs as a background subprocess, so the Gradio
cell can block without stopping it.

**Cell 1**

```python
!pip install -q gradio
!git clone -q https://github.com/evanwang810/llm-reverse /kaggle/working/code
```

**Cell 2, starts training in the background**

```python
import os, subprocess, sys, time

os.makedirs("/kaggle/working/run", exist_ok=True)
log = open("/kaggle/working/run/train.log", "w")
proc = subprocess.Popen(
    [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "train.py",
     "--data-dir", "/kaggle/input/fineweb-edu-tokens",
     "--run-dir", "/kaggle/working/run",
     "--session-start", str(time.time()),
     "--deadline-hours", "11.3"],
    cwd="/kaggle/working/code",
    stdout=log, stderr=subprocess.STDOUT,
    env=dict(os.environ, PYTHONUNBUFFERED="1"),
)
print("training pid", proc.pid)
```

**Cell 3, check that it actually started**

```python
import time; time.sleep(90)
!tail -n 30 /kaggle/working/run/train.log
```

**Cell 4, the dashboard (this cell blocks, that is fine)**

```python
import sys; sys.path.insert(0, "/kaggle/working/code")
import dashboard
dashboard.launch("/kaggle/working/run", share=True)
```

Click the public `gradio.live` link it prints. Interrupt the cell to stop the UI
without touching training.

## C. Dashboard on your own machine

Cheapest option, zero GPU quota, and the one I would default to. Download the
`run` folder from a finished Kaggle version, then:

```bash
python dashboard.py --run-dir path\to\downloaded\run
```

Everything works offline: loss curves, checkpoint picker, completions, per-token
probabilities. Only the Save and Stop buttons need a live trainer to listen.

## Smoke test before you burn quota

Locally, on CPU, no dataset needed:

```bash
python train.py --smoke-test --run-dir scratch --max-steps 60
```

Kill it partway with Ctrl-C, run the same command again, then:

```bash
python check_resume.py --run-dir scratch
```

It finds every restart in the CSV and tells you whether the step count and the
loss carried over cleanly.
