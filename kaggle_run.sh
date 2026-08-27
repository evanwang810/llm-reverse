#!/usr/bin/env bash
# One command that does the whole run. Safe to re-run: tokenization resumes
# where it left off and training resumes from the newest checkpoint.
#
#   bash kaggle_run.sh [HOURS] [BASE] [MAX_TOKENS]
#
#   bash kaggle_run.sh 0.5 small 1e8      # the 30 minute model, on 2xT4
#   bash kaggle_run.sh 8.5 large 2e9      # the full session model, on TPU v5e-8
#   bash kaggle_run.sh 11.3 large-gpu 2e9 # same base on 2xT4 instead
#
# BASE picks the entry in bases.py, which sets the batch shape, the learning
# rate and which tokenizer the corpus is built with. The shards and the model
# must agree on that last one, so pass the same BASE to both stages.
#
# HOURS is your total budget including tokenizing. Everything else is derived
# from it, so there is one number to think about.
#
# Environment switches:
#   DEVICE=tpu      force the TPU path (default: autodetect)
#   MONITOR=1       status blocks with a loss chart instead of raw step lines
#   EXPORT=1        after training, write a downloadable HF folder to
#                   /kaggle/working/model. This is what makes one session end
#                   with a model rather than a checkpoint.
#   GGUF=1          also write an f16 .gguf for Ollama (implies EXPORT=1)
#   EXPORT_REPO=... push to this HuggingFace repo too, needs HF_TOKEN set
#   CHAT_FRAC=0.2   share of the corpus that is conversations
#   DIRECTION=...   reverse (default), forward, or both
set -euo pipefail

if [ "${GGUF:-0}" = "1" ]; then EXPORT=1; fi

HOURS="${1:-9}"
BASE="${2:-large}"
MAX_TOKENS="${3:-2e9}"

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKENS="${TOKENS_DIR:-/kaggle/working/tokens}"
RUN="${RUN_DIR:-/kaggle/working/run}"

echo "=== llm-reverse: ${HOURS}h budget, base ${BASE}, ${MAX_TOKENS} tokens ==="
cd "$CODE"

if [ "${SKIP_PIP:-0}" != "1" ]; then
  pip install -q datasets
  # Optional, for braille charts in the monitor. Not the PyPI package of the same
  # name. Failure here is fine, the monitor falls back to ASCII.
  if [ -n "${TERMPLOT_REPO:-}" ]; then
    pip install -q "git+${TERMPLOT_REPO}" || echo "termplot install failed, using ascii charts"
  fi
fi

# Kaggle mounts a notebook's output at /kaggle/input/notebooks/<user>/<slug>/,
# which nobody guesses correctly. Rather than make you type a path, go and find
# a finished token set anywhere under /kaggle/input and use the biggest one.
find_tokens() {
  python - "$1" <<'PY'
import json, sys
from pathlib import Path

want = Path(sys.argv[1])
if (want / "meta.json").exists():
    print(want)
    raise SystemExit

best = None
patterns = ["meta.json", "*/meta.json", "*/*/meta.json", "*/*/*/meta.json",
            "*/*/*/*/meta.json", "*/*/*/*/*/meta.json"]
for root in (Path("/kaggle/input"), Path("/kaggle/working")):
    if not root.exists():
        continue
    for pat in patterns:
        for meta in root.glob(pat):
            try:
                m = json.loads(meta.read_text())
            except Exception:
                continue
            if "shards" not in m:
                continue
            n = int(m.get("total_tokens", 0))
            if n > 0 and (best is None or n > best[0]):
                best = (n, meta.parent)
if best:
    print(best[1])
PY
}

if [ "${SMOKE:-0}" = "1" ]; then
  # Whole pipeline in a few minutes on synthetic data, same code path.
  echo "SMOKE=1: skipping tokenization, training a tiny model on generated data"
else
  FOUND="$(find_tokens "$TOKENS" 2>/dev/null || true)"
  if [ -n "$FOUND" ] && [ "$FOUND" != "$TOKENS" ]; then
    echo "found existing tokens at $FOUND, using those instead of $TOKENS"
    TOKENS="$FOUND"
  fi
  # Never try to write into the read-only input mount.
  case "$TOKENS" in
    /kaggle/input/*)
      if [ ! -f "$TOKENS/meta.json" ]; then
        echo "TOKENS_DIR=$TOKENS is under the read-only /kaggle/input and has no" >&2
        echo "meta.json. Check the real path in the file browser, or unset" >&2
        echo "TOKENS_DIR to tokenize fresh into /kaggle/working." >&2
        exit 1
      fi
      ;;
  esac
fi

if [ "${SMOKE:-0}" = "1" ]; then
  :
else
  # Resumable: if meta.json already has enough tokens this returns immediately.
  # A non-zero exit here is not conclusive. The HF streaming reader can abort
  # during interpreter shutdown, after the shards are already written, so ask
  # separately whether the data is actually complete rather than trusting the
  # exit code.
  python tokenize_reverse.py --base "$BASE" --out-dir "$TOKENS" --max-tokens "$MAX_TOKENS" \
    --direction "${DIRECTION:-reverse}" --chat-frac "${CHAT_FRAC:-0.2}" \
    || echo "tokenizer exited non-zero, checking the shards themselves"
  python tokenize_reverse.py --base "$BASE" --out-dir "$TOKENS" --max-tokens "$MAX_TOKENS" \
    --direction "${DIRECTION:-reverse}" --chat-frac "${CHAT_FRAC:-0.2}" --verify-only
fi

# Reserve time for tokenizing, and derive the schedule knobs from the budget:
# decay over roughly the last 15% of steps, and about 8 permanent milestones.
# Tokens are uint16, so the corpus alone is two bytes each, and a 173M model
# checkpoints at roughly 2GB with optimizer state. Kaggle gives about 21GB of
# working disk. Running out halfway through loses the session, so say so now.
python - "$MAX_TOKENS" "$RUN" "$TOKENS" <<'PY' || true
import json, shutil, sys
from pathlib import Path

tokens_gb = float(sys.argv[1]) * 2 / 1e9
root = Path(sys.argv[2])
while not root.exists() and root.parent != root:
    root = root.parent
free_gb = shutil.disk_usage(root).free / 1e9

# This runs after tokenization, so if the corpus is already written its bytes
# are gone from the free figure and must not be counted a second time.
meta = Path(sys.argv[3]) / "meta.json"
on_disk = 0.0
if meta.exists():
    try:
        on_disk = int(json.loads(meta.read_text()).get("total_tokens", 0)) * 2 / 1e9
    except Exception:
        pass

ckpt_gb = 8  # rolling checkpoint, milestones, and the temp copy while writing
still_needed = max(0.0, tokens_gb - on_disk) + ckpt_gb
print(f"disk: {free_gb:.1f}GB free, corpus {on_disk:.1f}GB written of "
      f"{tokens_gb:.1f}GB, still need about {still_needed:.1f}GB")
if still_needed > free_gb:
    print(f"WARNING: this may not fit. Lower the third argument, or expect the "
          f"run to die on a full disk part way through.", file=sys.stderr)
PY

read -r TRAIN_HOURS DECAY_STEPS MILESTONE_MIN <<EOF
$(python - "$HOURS" <<'PY'
import sys
hours = float(sys.argv[1])
train = max(0.4, hours - 0.45)
print(f"{train:.2f} {max(200, int(120 * train))} {max(15, int(train * 60 / 8))}")
PY
)
EOF

NGPU="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
DEVICE="${DEVICE:-auto}"
if [ "$DEVICE" = "auto" ]; then
  if [ "$NGPU" -ge 1 ]; then
    DEVICE=gpu
  elif python -c 'import torch_xla' >/dev/null 2>&1; then
    DEVICE=tpu
  else
    DEVICE=gpu
  fi
fi

if [ "$DEVICE" = "tpu" ]; then
  TRAIN_SCRIPT=train_tpu.py
  LAUNCH=(python)   # train_tpu.py spawns its own replicas, there is no torchrun
  # A v3 core has 15.75GB of HBM and attention scores are batch x heads x seq x
  # seq in fp32, since XLA has no flash kernel and decomposes SDPA. At
  # micro_batch 8 that is 384MB per layer, and nineteen layers of it does not
  # fit. Four halves it, and grad_accum absorbs the difference so tokens per
  # step is unchanged at 262,144.
  MICRO_BATCH="${MICRO_BATCH:-4}"
  GRAD_ACCUM="${GRAD_ACCUM:-8}"

  # Kaggle caps TPU sessions at 9 hours, not the 12 you get on GPU.
  if python -c "import sys; sys.exit(0 if $HOURS > 9 else 1)"; then
    echo "WARNING: ${HOURS}h exceeds Kaggle's 9 hour TPU session cap. The session will be" >&2
    echo "         killed before the deadline and the final checkpoint will be the last" >&2
    echo "         periodic one. Use 8.5 or less." >&2
  fi

  if [ "${PREFLIGHT:-1}" = "1" ]; then
    echo "=== TPU preflight: proving the device works before spending the session ==="
    if ! python tpu_preflight.py --base "$BASE" --run-dir "$RUN" \
        --micro-batch "$MICRO_BATCH" --grad-accum "$GRAD_ACCUM" \
        --min-tok-s "${MIN_TOK_S:-80000}"; then
      echo "" >&2
      echo "=== preflight failed, refusing to start training ===" >&2
      echo "Nothing has been spent except the preflight. Fix what it reported, or set" >&2
      echo "DEVICE=gpu to fall back to the T4 path. PREFLIGHT=0 skips this check, which" >&2
      echo "is only sensible if you already know why it failed." >&2
      exit 3
    fi
  fi
else
  TRAIN_SCRIPT=train.py
  if [ "$NGPU" -ge 2 ]; then
    LAUNCH=(torchrun --nproc_per_node="$NGPU")
  else
    echo "WARNING: found $NGPU gpu(s), not 2. Check the Accelerator setting is GPU T4 x2."
    LAUNCH=(python)
  fi
fi

echo "=== training ${TRAIN_HOURS}h on ${DEVICE}, decay ${DECAY_STEPS} steps, "\
"milestone every ${MILESTONE_MIN} min ==="

START_TS="$(date +%s)"
DEADLINE_SECONDS="$(python -c "print(int($TRAIN_HOURS * 3600))")"

TRAIN_CMD=("${LAUNCH[@]}" "$TRAIN_SCRIPT"
  --base "$BASE"
  --data-dir "$TOKENS"
  --run-dir "$RUN"
  --deadline-hours "$TRAIN_HOURS"
  --session-start "$START_TS"
  --auto-decay --decay-steps "$DECAY_STEPS"
  --keep-checkpoints 1 --keep-weights "${KEEP_WEIGHTS:-0}"
  --save-every-min "${SAVE_EVERY_MIN:-12}"
  --milestone-every-min "$MILESTONE_MIN")

if [ -n "${MICRO_BATCH:-}" ]; then TRAIN_CMD+=(--micro-batch "$MICRO_BATCH"); fi
if [ -n "${GRAD_ACCUM:-}" ]; then TRAIN_CMD+=(--grad-accum "$GRAD_ACCUM"); fi

# Anything you want to pass straight through to train.py, for shapes that are
# not in bases.py: TRAIN_EXTRA="--micro-batch 2 --grad-accum 32".
if [ -n "${TRAIN_EXTRA:-}" ]; then
  # shellcheck disable=SC2206
  TRAIN_CMD+=(${TRAIN_EXTRA})
fi

# Continuing a previous session: point at the ckpt_step*.pt from its output.
# Without this, resume still works by globbing /kaggle/input, but being explicit
# means a wrong path fails loudly instead of silently starting from scratch.
if [ -n "${RESUME_FROM:-}" ]; then
  if [ ! -f "$RESUME_FROM" ]; then
    echo "RESUME_FROM=$RESUME_FROM does not exist" >&2
    exit 1
  fi
  echo "resuming from $RESUME_FROM"
  TRAIN_CMD+=(--resume-from "$RESUME_FROM")
fi

# A long run on free hardware will occasionally die on something outside our
# control: a CUDA fault, an NCCL abort, a preempted GPU. Training resumes from
# its last checkpoint with optimizer and loss-scale state intact, so the right
# response is to start it again rather than lose the session. --session-start
# pins the deadline to the original clock, so restarts cannot extend the run.
train_with_restarts() {
  local attempt code now
  for attempt in $(seq 1 "${ATTEMPTS:-8}"); do
    if [ "$attempt" -gt 1 ]; then
      echo "=== restart $((attempt - 1)), resuming from the last checkpoint ==="
    fi
    code=0
    local started
    started="$(date +%s)"
    "${TRAIN_CMD[@]}" || code=$?
    if [ "$code" -eq 0 ]; then
      return 0
    fi
    now="$(date +%s)"
    # A crash three hours in is worth retrying: preemption, a transient NCCL
    # fault, a flaky mount. A crash in the first ninety seconds is not, because
    # nothing changes between attempts and it fails identically. The TPU
    # shakedown burned fifteen minutes reproducing one startup error eight times
    # before giving up, and on the long run that is a wasted session.
    if [ $((now - started)) -lt "${MIN_RETRY_SECONDS:-90}" ]; then
      echo "=== training exited $code after $((now - started))s, before it got going." >&2
      echo "    That is a startup failure, not a crash, so retrying would just" >&2
      echo "    reproduce it. Fix what it printed above." >&2
      return "$code"
    fi
    if [ $((now - START_TS)) -ge "$DEADLINE_SECONDS" ]; then
      echo "=== training exited $code but the deadline has passed, stopping ==="
      return 0
    fi
    echo "=== training exited $code with $(( (DEADLINE_SECONDS - now + START_TS) / 60 )) min left, retrying in 20s ==="
    sleep 20
  done
  echo "=== gave up after ${ATTEMPTS:-8} attempts ==="
  return 1
}

# Turn NCCL faults into a Python exception with a message instead of an opaque
# abort, so the next failure says what it actually was.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

if [ "${SMOKE:-0}" = "1" ]; then
  TRAIN_CMD+=(--smoke-test --max-steps "${SMOKE_STEPS:-80}")
fi

# With MONITOR=1, train in the background and put the monitor in front, so a
# committed run's log becomes readable status blocks with a loss chart instead
# of a wall of step lines.
CODE_EXIT=0
if [ "${MONITOR:-0}" = "1" ]; then
  mkdir -p "$RUN"
  rm -f "$RUN/train.exit"
  echo "monitor mode: full training output goes to $RUN/train.log"
  # The marker file, not the pid, tells the monitor when to stop. A finished
  # background job lingers as a zombie that still answers kill(pid, 0), so a
  # liveness check would never return. A stale heartbeat is not a signal either:
  # it just means rank 0 is mid-save or mid-restart.
  { c=0; train_with_restarts || c=$?; echo "$c" > "$RUN/train.exit"; } \
    > "$RUN/train.log" 2>&1 &
  TRAIN_PID=$!
  python monitor.py --run-dir "$RUN" --watch --exit-file "$RUN/train.exit" \
    --interval "${MONITOR_INTERVAL:-300}" --until-done --no-color || true
  wait "$TRAIN_PID" || true
  CODE_EXIT="$(cat "$RUN/train.exit" 2>/dev/null || echo 1)"
else
  train_with_restarts || CODE_EXIT=$?
fi

if [ "$CODE_EXIT" -ne 0 ]; then
  echo "=== training failed with $CODE_EXIT, not going on to fine-tuning ===" >&2
  exit "$CODE_EXIT"
fi

# There is no fine-tuning stage here on purpose. The conversations go into
# the finetuning corpus instead, at CHAT_FRAC. Bolting a forward-format SFT
# run onto a backward-trained model would stack two novel things on each
# other, and when the result came out wrong you would not know which broke.

# ---------------------------------------------------------------------------
# One-session export. EXPORT=1 turns the newest checkpoint into a HuggingFace
# folder under /kaggle/working, which is what the notebook Output tab serves,
# so a single Save & Run All ends with a model you can download. GGUF=1 also
# writes an f16 .gguf next to it for Ollama.
#
# It runs after training on purpose. If the session is killed early you still
# have checkpoints; you just do the export in a free CPU notebook instead.
# ---------------------------------------------------------------------------
if [ "${EXPORT:-0}" = "1" ]; then
  OUT_DIR="${EXPORT_DIR:-/kaggle/working/model}"
  CKPT="$(ls -1t "$RUN"/weights_*.pt "$RUN"/milestone_*.pt "$RUN"/ckpt_*.pt 2>/dev/null | head -1)"
  if [ -z "$CKPT" ]; then
    echo "=== EXPORT=1 but no checkpoint in $RUN ===" >&2
    exit 1
  fi
  echo "=== exporting $CKPT to $OUT_DIR ==="
  EXPORT_ARGS="--ckpt $CKPT --out $OUT_DIR --repo ${EXPORT_REPO:-local/llm-reverse}"
  # HF_TOKEN comes from Kaggle Secrets, never from a literal in the notebook.
  if [ -n "${EXPORT_REPO:-}" ] && [ -n "${HF_TOKEN:-}" ]; then
    EXPORT_ARGS="$EXPORT_ARGS --push"
    echo "    and pushing to ${EXPORT_REPO}"
  fi
  python publish_hf.py $EXPORT_ARGS

  if [ "${GGUF:-0}" = "1" ]; then
    echo "=== converting to GGUF ==="
    # Only the converter is needed for f16, not a compiled llama.cpp.
    # Quantizing needs the binary, which is a multi-minute build; do that
    # locally where you are not paying for accelerator time.
    if [ ! -d /kaggle/working/llama.cpp ]; then
      git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/working/llama.cpp
    fi
    pip install -q -r /kaggle/working/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
    python export_gguf.py --hf-dir "$OUT_DIR" --llama-cpp /kaggle/working/llama.cpp --outtype f16 \
      || echo "gguf conversion failed, the HF folder is still there"
  fi

  echo "=== done. Download from the notebook Output tab: $OUT_DIR ==="
  ls -la "$OUT_DIR"
fi
