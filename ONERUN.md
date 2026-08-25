# One run, start to finish

> **Base, not preset.** This project finetunes a pretrained checkpoint, so
> wherever an older note says `--preset`, the flag is now `--base` and the
> options are `small`, `large` and `large-gpu`. See
> [bases.py](bases.py) and [README.md](README.md#why-these-bases).


> **Reverse note.** This walkthrough is llm67m's, with the names changed. The
> only substantive differences are that the tokenizer is `tokenize_reverse.py`
> and takes `--chat-frac`, and that there is no instruction-tuning step: the
> conversations are mixed into pretraining instead. See
> [README.md](README.md#conversations) for why.


Everything in the browser. One notebook, two cells, one button. Tokenizing and
training happen in the same session, so there is no dataset shuffling and no
second visit. You press Run, come back the next day, and download a finished
model.

Total hands-on time: about five minutes. Then eleven hours of waiting.

---

## 1. Verify your phone number

You just signed in. Do this first, because without it the accelerator and
internet toggles are locked and nothing below works.

1. Click your **avatar**, top right, then **Settings**.
2. Scroll to **Phone Verification**.
3. Enter your number, enter the code they text you.

If it already says verified, skip ahead.

---

## 2. Create the notebook

1. **Create** in the left sidebar, then **Notebook**.
2. Click the title at the top left (it says something like "notebook1a2b3c") and
   rename it `llmrev-run`.
3. Open the right sidebar if it is closed, the **>** arrow at the top right.
4. Set two things:
   - **Accelerator**: click it, pick **GPU T4 x2**. The `x2` matters, it is what
     makes both GPUs get used. If the list happens to offer an **L4** or **A100**,
     take that instead and tell me, because on those chips we can throw away the
     fp16 loss scaler and get flash attention.
   - **Internet**: toggle **On**. Needed to clone the code and download
     FineWeb-Edu.

No **Add Input** step. The code comes from GitHub and the data is downloaded in
cell 1.

---

## 3. Paste the cells

Pick one of two paths. Both end up in the same place.

|  | Path A, commit | Path B, press run |
| --- | --- | --- |
| tab must stay open | no | yes, all 9 hours |
| laptop must stay awake | no | yes |
| survives a dropped connection | yes | probably not |
| survives you forgetting about it | yes | no |
| see progress while it runs | status blocks with a loss chart, in the log | live bars and a refreshing chart |
| chat with a checkpoint mid-run | no | yes |

Path A is the one to use. The gap in what you can see is now small, because it
prints the same charts into the log every five minutes.

### Path A: walk away (most reliable)

**One cell. That is the whole notebook.**

```python
!rm -rf /kaggle/working/code && git clone -q https://github.com/evanwang810/llm-reverse /kaggle/working/code
!MONITOR=1 bash /kaggle/working/code/kaggle_run.sh 9
```

Then **Save Version -> Save & Run All**.

`MONITOR=1` makes the run watchable: training goes to the background and the
terminal monitor takes the foreground, so instead of a wall of step lines your
log gets a status block every five minutes with progress bars, a stat table and
a loss chart. Open the **Versions** tab any time, from any device, to read it.

