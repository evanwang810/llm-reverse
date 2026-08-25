# Running on the TPU instead

> **Base, not preset.** This project finetunes a pretrained checkpoint, so
> wherever an older note says `--preset`, the flag is now `--base` and the
> options are `small`, `large` and `large-gpu`. See
> [bases.py](bases.py) and [README.md](README.md#why-these-bases).


> **Reverse note.** The TPU path is unchanged by the reversal: it reads the
> same shards, which are already laid out in the direction the model reads.
> Only the tokenizer name and the missing fine-tuning stage differ from
> llm67m's version of this document.


A Kaggle TPU VM v3-8 is roughly ten to fifteen times the effective throughput of
the dual T4s, which turns the three-session plan for a 101M model into one
afternoon. It is also the less proven path in this repo: the CUDA trainer has
run for hours on real hardware, and the XLA one has been tested against a stubbed
`torch_xla` on CPU. Everything below exists because of that gap.

## The cell

One cell, same as the GPU path, with the accelerator set to **TPU VM v3-8** in
the notebook sidebar.

```python
!rm -rf /kaggle/working/code && git clone -q https://github.com/evanwang810/llm-reverse /kaggle/working/code
!DEVICE=tpu MONITOR=1 bash /kaggle/working/code/kaggle_run.sh 8.5 large 2e9
```

Note the **absolute path** to `kaggle_run.sh`. Each `!` line runs in
`/kaggle/working`, not in the cloned repo, so a bare `bash kaggle_run.sh` fails
with `No such file or directory` before anything starts. The script works out
where it lives and changes into that directory itself, so there is no need to
`cd` first and no reason to chain the two with `&&`.

`8.5`, not 11. **Kaggle caps TPU sessions at 9 hours**, where GPU gets 12. The
script warns if you ask for more, but it cannot extend the cap.

`tpu1session`, not `67m`. The whole point of the TPU is that it moves the
compute budget, and the best model for a budget scales with its square root. One
8.5 hour v3-8 session is about 3.9e18 FLOPs, where the optimum is near 173M total
parameters on 3.5B tokens. That was a from-scratch calculation; this project
finetunes Qwen3-0.6B instead, which needs far fewer tokens for the same effect.
The old sizing note read: 19 layers at GPT-2 small's width and
takes roughly 6.2 of the 8.5 hours. Running `67m` here would finish in about
three hours and leave half the session unspent on a model the T4s could have
reached anyway. Running the T4 base on a TPU is the most common way to get a
TPU that is fast and pointless.

`DEVICE=tpu` is optional. With no setting the script picks GPU if it sees a CUDA
device and TPU otherwise, so on a TPU notebook it does the right thing on its
own. Set it explicitly when you want the run to fail loudly rather than silently
train on the wrong accelerator.

## Measured

A full shakedown on a real v3-8, all ten stages green:

| | |
|---|---|
| throughput | **202,000 tokens/sec** on the 173M model |
| per step | 0.32s |
| effective | 163 TFLOP/s, 39% of the chip's 420 peak |
| against the dual T4 | **4.8x** |
| compilations | 4 during warmup, then flat |

The compile count is the one to watch. Flat means the step graph is static and
XLA is executing a cached program rather than building a new one; if it tracks
the step count instead, see the sync note below, because that is what a hundred
fold slowdown looks like from the outside.

At that rate 8 hours of training is about 5.9B tokens, so the
Chinchilla-optimal 3.5B for this model takes under 5 hours and the session has
room to over-train, which is the right trade for something you intend to run.

## The queue

Kaggle has far fewer TPUs than people who want them, and it has been that way
for a long time. Queues in the dozens and waits of one to two hours are normal,
not an outage. Committed runs queue longer than interactive ones because they
ask for a guaranteed multi-hour slot.

**Waiting costs nothing.** Quota is counted against execution time, not queue
time, so a commit that sits pending overnight has spent none of your 20 weekly
hours. Leave it queued; it will run when capacity frees.

Telling a queue apart from a hang takes one look at the log. Queued means
*zero* output, no banner at all. Once the container starts, the very first thing
printed is:

```
=== TPU preflight: proving the device works before spending the session ===
```

If you can see that line, you have a TPU and the code is running. If you cannot,
you are still waiting for hardware and nothing in this repo has executed yet.

Since checkpoints are interchangeable between the two trainers, the queue is
worth hedging against rather than waiting on: start a GPU commit as well, and
take whichever lands first.

## Shake it down first

Before committing 8.5 hours, spend fifteen minutes in an **interactive** session
running the whole pipeline small:

```python
!rm -rf /kaggle/working/code && git clone -q https://github.com/evanwang810/llm-reverse /kaggle/working/code
!python /kaggle/working/code/shakedown.py
```

This is not the preflight. `tpu_preflight.py` answers one question in ten
minutes using synthetic tensors: is this device worth a session. The shakedown
answers a different one: does the entire pipeline work here, end to end, on the
real hardware, through the real data path. It really tokenizes FineWeb-Edu,
really builds the corpus, really trains the preset you are about to use, really
saves a checkpoint, really resumes from it, really fine-tunes, and really loads
the result into a chat session.

Nine stages, each printing what it proves:

| stage | proves |
|---|---|
| environment | the box is what you think it is |
| dependencies | the packages a real run installs are importable |
| tokenize | network, HuggingFace streaming and the shard writer work |
| dataloader | batches have the right shape and repeat for the same step |
| model | the preset really is the size `config.py` claims |
| train | the real trainer runs the real preset on the real device |
| **resume** | **a restart continues the run instead of starting over** |
| chat | the checkpoint loads back and generates text |

The resume stage is the one worth the wait. Kaggle sessions die, and the whole
design assumes a restart picks up from the last checkpoint with no loss
discontinuity. That property is invisible in a short run and expensive to
discover is broken in a long one, so it is tested directly: train, stop, resume,
then run `check_resume.py` over the boundary and report the loss either side.

A failing stage does not stop the ones after it. A bundle showing four problems
is worth more than four separate runs finding one each.

It writes **`/kaggle/working/shakedown.zip`** containing the full log, a JSON
summary of every stage, and the `loss.csv`, `status.json` and `samples.txt` a
real run would produce. Download it from the file browser on the right.

It works on GPU too, picking `67m` and `train.py` instead. Useful for confirming
a change did not break the T4 path before spending a session on it.

## The preflight

Before it trains anything, the run spends about ten minutes proving the TPU
works. Five gates, cheapest first, and any failure exits without starting:

| gate | checks | why |
|---|---|---|
| 1 | `torch_xla` imports and reports a TPU | wrong accelerator setting |
| 2 | a matmul on device gives the right answer | backend wired up but broken |
| 3 | all replicas start and can `all_reduce` | the usual spawn failure |
| 4 | the real model trains at a real speed | **the one that matters** |
| 5 | a checkpoint survives a save and reload | untrustworthy output |

Gate 4 is the point of the whole exercise. A TPU that is present, passes every
correctness check and trains at 9k tokens/sec is the expensive failure: nothing
raises, the log looks healthy, and nine hours later there is a quarter of a
model. Two things cause it.

**Silent host fallback.** An operation with no XLA lowering runs on the CPU, and
every step pays a round trip off the device.

**Recompilation.** XLA compiles one program per distinct graph. Anything that
changes the shape, or bakes a changing Python scalar into the graph, makes it
compile again, and a compile costs far more than a step. The learning rate is
the most likely trigger, since it moves on every step during warmup and decay.
Gate 4 therefore varies the learning rate across its measured steps on purpose,
then reads XLA's `CompileTime` counter to see whether the count plateaued. Still
climbing means the run would spend its life compiling.

So gate 4 measures throughput and compares it to a floor, defaulting to 80k
tokens/sec against the T4 path's measured 42k. Below the floor it aborts:

```
PREFLIGHT FAILED at gate 4 (throughput)
  31k tok/s is below the 80k floor
  The TPU is running but something is falling back to the host.
```

Nothing has been spent at that point except the preflight.

To override: `MIN_TOK_S=50000` moves the floor, `PREFLIGHT=0` skips the check
entirely. Skipping is only sensible when you already know why it failed.

Nothing here can damage a TPU. The risk being managed is a wasted session.

## What differs from the CUDA path

`train_tpu.py` is a separate file rather than a branch inside `train.py`. The
two disagree on nearly everything structural — torchrun against `xmp.spawn`,
NCCL against XLA collectives, fp16 with a GradScaler against native bf16, eager
execution against a traced graph — and threading that through the working
trainer would put it at risk to add an unproven one. Everything genuinely shared
is imported, including `save_checkpoint`.

Five deliberate differences:

**The step is executed, every step.** This is the one that matters, and getting
it wrong cost four runs. XLA is lazy: operations accumulate into a pending graph
and nothing runs until something forces it. Almost every example gets the sync
for free because `MpDeviceLoader` issues one per iteration, and this trainer
feeds batches from its own index-based sampler instead, so nothing was issuing
it at all.

Left alone the graph never resets at the end of a step. It keeps growing, one
more forward, backward and optimizer update each time, until something reads a
value off the device and forces the whole accumulated thing to compile and run.
That graph is a different size every time, so it is a fresh compilation every
time, and compile cost climbs with graph size. What you see is a training loop
running about a hundred times too slowly, getting worse the longer it runs, and
eventually a worker dying with no Python traceback at all when the compiler
gives up. Every number in the log looks fine. The loss falls. The samples
improve. It is just a hundred times too slow, and nothing says why.

`xm.optimizer_step(optimizer, barrier=True)` is the documented fix when
`MpDeviceLoader` is not in the picture; `xla_compat.optimizer_step` passes it and
calls `sync()` afterwards regardless. The evaluation loop syncs per batch for the
same reason.

**One micro batch per graph, not the whole accumulation.** A v3 core has
15.75GB of HBM, and XLA has no flash-attention kernel so it decomposes SDPA into
matmuls and a softmax, materialising the full attention score matrix as
`batch x heads x 1024 x 1024` in fp32. At micro_batch 8 that is 384MB per layer.
Nineteen layers of it, times four micro batches all live in one graph, is 36.7GB
against a 15.75GB budget, which is what the first real run hit almost exactly.
Syncing after each micro batch's backward changes nothing arithmetically,
because gradients live in `.grad` and persist across graph boundaries, but peak
memory becomes one micro batch instead of `grad_accum` of them. With
micro_batch 4 that is about 7GB.

**No GradScaler.** bf16 has fp32's exponent range, so there is nothing to
rescale. The checkpoint still carries a null `scaler` key so the formats match.

**No DDP.** Gradients accumulate into `.grad` across micro batches and
`xm.optimizer_step` all-reduces once, which is what DDP's `no_sync` was
emulating anyway.

**An explicit weight sync at startup.** DDP broadcasts rank 0's weights at
construction; XLA has no equivalent. Eight replicas each training a slightly
different model would still produce a falling loss curve, so every replica seeds
identically and then all-reduces its parameters once before the first step.

**Control flags and loss values are read at most once every `log_every` steps.**
Pulling a scalar off the device forces the graph to execute and the host to
wait. Save timing comes off the master replica's clock alone and is distributed
through that same collective — eight wall clocks compared independently will not
agree on where a save boundary falls, and one replica waiting at a rendezvous
the others never reach is a hang with no watchdog to end it.

**Samples are generated on the host.** Incremental decoding grows the sequence
by one token per pass, so on device it would compile a fresh graph per token. On
the CPU it costs under a second and the log keeps its samples.

## No fine-tuning stage

There is no instruction-tuning step on either device. The conversations are
mixed into the pretraining corpus at `CHAT_FRAC` instead, so there is nothing
to chain after the trainer exits. See [README.md](README.md#conversations).

## Checkpoints are interchangeable

A checkpoint written by the TPU trainer resumes in the CUDA trainer and the
other way round, verified on a smoke run: loss continued at 6.54 across the
switch with no spike. So a TPU session that dies early can be finished on GPU,
and `chat.py` and the dashboard read either without knowing which
trainer produced it.

## If the preflight fails

| message | what it means |
|---|---|
| gate 1, `torch_xla` not importable | accelerator is not set to TPU. Do not pip install torch_xla; Kaggle's TPU image ships a matching build and installing over it breaks the runtime |
| gate 3, replicas failed to start | usually a stale TPU lock from an earlier cell in the same session. Restart the kernel and run this cell first |
| gate 4, recompilation | the step graph is not static. Report the number; it means something in the loop changes shape |
| gate 4, throughput | something is falling back to the host |
| gate 5, checkpoint mismatch | stop. The run would produce checkpoints you cannot trust |
| `AttributeError: module 'torch_xla.core.xla_model' has no attribute ...` | the API moved again. Every name is resolved in one place, `xla_compat.py`; add the new spelling there rather than at the call site |
| `BrokenProcessPool`, no traceback, tens of seconds per step | the graph is not being executed per step. Check the `compiles` figure in the log: if it tracks the step count, something in the loop stopped issuing a sync |
| `WARNING: N compilations over N steps` | the trainer noticed the same thing itself and will say so by step 40 rather than at hour eight |

## A note on torch_xla versions

Kaggle's TPU image moved from PT/XLA 2.1 to 2.8 without announcement, and 2.7
**removed** `xm.get_ordinal` and `xm.xrt_world_size` rather than deprecating
them, so code written against the older spelling dies on the first line of the
spawned function. Guessing a version and hardcoding its names failed twice, so
all of it now resolves through `xla_compat.py`, newest name first, older name as
a fallback. Anything unresolvable raises at import with the full list of what was
tried, rather than at replica zero four minutes into a run, one name at a time.

`tpu_preflight.py` prints which spellings it resolved, so the log says what API
it actually found.

Falling back costs one command: `DEVICE=gpu` on a GPU notebook, and the T4 path
runs exactly as before.
