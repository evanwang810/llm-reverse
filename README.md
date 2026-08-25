# llm-reverse

Finetune a pretrained model to write **backwards**. Every document is tokenized
forwards and then has its token order reversed, so the model predicts each token
from the ones that follow it. Give it the end of a passage and it writes what
came before. Give it an answer and it writes the question.

Two models, two devices, one codebase:

| | small | large |
| --- | --- | --- |
| base | [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) |
| device | 2×T4 | TPU v5e-8 |
| wall clock | ~30 min | ~8.5 h |
| tokens seen | ~60M | ~1.4B |
| for | proving the pipeline | the one you publish |

`large-gpu` is the same Qwen checkpoint sized for the T4 pair, at roughly a
quarter the tokens per session. Use it if you would rather not touch XLA.

Built on Kaggle's free tier, resumable across as many sessions as you like. The
operational shell is [llm67m](../llm67m)'s: same deadline handling, same atomic
checkpoints, same O(1) resume, same dashboard.

## Why finetune instead of pretrain

Reversal is a property of the *data*, not the architecture, so there is nothing
to gain from starting at noise. A pretrained model already knows English; what
it does not know is which way round to emit it. Starting from a converged
checkpoint means a session buys you adaptation rather than basic literacy.

**Full finetuning, not LoRA.** Reversing token order is a far larger
distribution shift than instruction tuning, closer to continued pretraining. A
low-rank adapter underfits it badly, and the failure looks like "the idea does
not work" rather than "the method was wrong", which is the worst kind of
negative result to get. Every entry in `bases.py` is sized for a full finetune,
at 16 bytes per parameter: half weights, half grads, fp32 master copy, and
Adam's two moments.

```bash
python bases.py     # both entries with the memory arithmetic
```

## Why these bases

Boring on purpose. The newest small models are actively bad for this experiment.

[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) is Apache-2.0 and
genuinely newer, but it is a hybrid: 24 layers as `6 × (3 × GatedDeltaNet → 1 ×
GatedAttention)`, plus sparse MoE, plus a vision encoder, on a 248,320-token
vocabulary. Gated DeltaNet is linear attention carrying an explicit recurrent
state, which is direction-dependent in a way softmax attention is not, so
reversing token order on it is a different and harder experiment than this one.
MoE routing re-routes hard under a shift this size, and router collapse during
full finetuning is a real failure mode. If the result came out bad you would not
know whether the reversal failed or the architecture did. Gemma 4 E2B has the
same problem from a different direction, via per-layer embeddings.

Qwen3-0.6B-Base is dense, text-only, plain GQA, 28 layers, 0.44B non-embedding,
Apache-2.0. The experiment is about the reversal, so everything else should be
the dullest thing that works.

## Why the TPU

Kaggle's TPU is a **v5e-8** now, not the v3-8 llm67m used: 16 GB HBM per chip,
128 GB total, 197 bf16 TFLOPS per chip. Three reasons it is the better device
here, not merely an equal one.

**Native bf16.** The T4 is Turing and has none, so it is fp16 with a GradScaler.
Qwen3 is bf16-native, and full-finetuning it in fp16 under a direction flip is
exactly the setup that produces loss-scale thrash and NaNs. On TPU that whole
class of problem does not exist.

**Separate quota.** TPU hours do not come out of your 30 GPU-hours, so it adds
budget rather than competing for it.

**Static shapes, already.** Variable-length inputs cause endless XLA recompiles,
which is why most people bounce off TPU finetuning. `data.py` emits fixed-size
blocks of exactly `block_size` tokens because of the Feistel sampler, so shapes
are static by construction. That cost was paid in llm67m.

0.6B at 16 bytes per parameter is 9.5 GB, which fits one chip with room for
activations, so plain data parallel across the eight chips works and no sharding
is needed.

## The reversal, precisely

Tokenize forwards, then reverse the ids. **Never reverse the characters and
re-tokenize.** BPE merges were learned on forward English, so a reversed string
tokenizes into fragments sharing almost no vocabulary with the forward corpus,
and you would be finetuning on a different language that merely looks familiar.

The end-of-document token stays at the end, so the packed stream reads

```
tN ... t1 t0 <eos> sM ... s1 s0 <eos>
```

and a normal left-to-right causal model is now predicting backwards.

## Conversations

Conversations go into the **finetuning mix** at `--chat-frac 0.2`, not a separate
instruction-tuning stage. A backward model plus a forward-format SFT run is two
novel things stacked on each other, and when the output came out wrong you would
not know which one broke.

They are formatted in the base tokenizer's own ChatML. Both SmolLM2 and Qwen3
already ship `<|im_start|>` and `<|im_end|>`, so nothing is added to the
vocabulary and no embedding matrix is resized. A forward exchange

```
<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\nA<|im_end|>
```

reverses to open with the `<|im_end|>` that closed the assistant turn. So the
jeopardy prompt at inference, `<|im_end|>` followed by the reversed answer, is a
**literal prefix of the training data** rather than an approximation of one. That
is why turns are joined by a newline instead of each ending in one: a trailing
newline would put a stray token in front and the prefix would no longer match.

Default source is [smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk),
Apache-2.0 and built for small models. `--chat-dataset` takes anything with a
`messages` list of `{role, content}` or a ShareGPT `{from, value}` list.

## Shard width

`uint16` tops out at 65,535. SmolLM2's 49k vocabulary fits; Qwen3's 151,669 does
not. The dtype follows the tokenizer, is recorded in `meta.json`, and is read
back by `data.py`. It is not a flag you set, because reading shards at the wrong
width yields plausible-looking garbage rather than an error.

Practically: the large corpus is 4 bytes per token, so 2B tokens is 8 GB. That is
comfortable inside Kaggle's 20 GB dataset storage, and 2B is plenty for a
finetune.

## Running it

Pre-tokenize in a free Kaggle **CPU** notebook. Kaggle pulls from HuggingFace far
faster than a home connection, spends no accelerator quota, and its output mounts
into the training notebook through Add Input → Notebook Output, so nothing large
crosses your uplink.

```bash
python tokenize_reverse.py --base large --out-dir /kaggle/working/tokens --max-tokens 2e9
```

Then one cell in the training notebook:

```bash
bash kaggle_run.sh 8.5 large 2e9        # TPU v5e-8
bash kaggle_run.sh 0.5 small 1e8        # 30 minute model on 2xT4
bash kaggle_run.sh 11.3 large-gpu 2e9   # Qwen on 2xT4 instead
```

`BASE` sets the batch shape, the learning rate and which tokenizer the corpus was
built with. **Pass the same `BASE` to both stages**: the shards and the model
must agree on the tokenizer, and `train.py` refuses to start if they do not,
because token ids from one tokenizer are meaningless to another model and
nothing downstream would raise.

Everything else, including the click-by-click walkthrough, is
[QUICKSTART.md](QUICKSTART.md) and [TPU.md](TPU.md).

## Talking to it

```bash
python chat.py
```

The REPL reads the direction stamp the trainer left. Two modes, `/mode` toggles:

- **prefix** (default): you type the end of a passage, it writes what came
  before. Degrades gracefully, so an early checkpoint still reads as something.
- **jeopardy**: you type an answer, it writes the question. Depends on the turn
  markers having been learned, so it looks broken until they have been.

No streaming in reverse mode. Every new token lands to the left of the last one,
and a terminal cannot show text growing that way.

## Publishing to HuggingFace

```bash
python publish_hf.py --ckpt run/weights_step0010000.pt --repo you/qwen3-0.6b-reverse --push
```

**The format is a folder.** `config.json`, a `.safetensors` file, tokenizer
files, and a `README.md` with a YAML header. Upload the folder; that is the whole
thing.

This is short because a finetuned Qwen3 is still a Qwen3. The architecture is one
transformers already ships, so there is no remote code, no `auto_map`, and nobody
downloading it needs `trust_remote_code=True`. Only the weights changed.

`publish_hf.py` reloads what it wrote and refuses to continue if the logits
disagree with the checkpoint.

**The model card is load-bearing here.** A reverse model called through
`pipeline("text-generation")` returns confident nonsense rather than an error, so
every first attempt fails silently unless the card leads with the wrapper. The
generated card does, and ships a `write_backwards` function that reverses the ids
in, reverses them out, and decodes **once** at the end.

Publishing from Kaggle: put your write token in **Add-ons → Secrets** as
`HF_TOKEN`, never inline in a cell.

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

Push a `weights_*.pt` or `milestone_*.pt`, not a `ckpt_*.pt`. Full checkpoints
carry two Adam moments per parameter and are three times the size for no benefit
to anyone downloading them.

## What to expect

**Backward loss lands a few percent above forward, and that is the correct
result.** Papadopoulos et al., *Arrows of Time for Large Language Models* (2024),
trained both directions across sizes and languages and found the backward model
consistently worse by a small, robust margin. If your backward loss matches
forward exactly, suspect the reversal never happened.

For the controlled version, `--direction both` emits every document twice, once
each way, with a leading mode marker. One model, identical text, and the gap is a
number you can put in the model card. It costs half your tokens per direction, so
it is a better experiment and a worse model.

**Expect a loss spike in the first few hundred steps**, then a fairly quick
recovery as the lexical knowledge transfers. That spike is the direction flip and
it is the signal the small model exists to show you.

**Generations read locally fine and feel weak at sentence openings.** Function
words are more forward-predictable than backward-predictable, so the model works
against the grain exactly where a forward model coasts.

## Tests

```bash
python test_reverse.py                     # 43 checks, both tokenizers
python tokenize_reverse.py --self-test     # the whole data pipeline, no network
python train.py --smoke-test --run-dir scratch --max-steps 30
python check_resume.py --run-dir scratch
```

Set `HF_HUB_OFFLINE=1` after the first run; the Hub round-trip is most of the
runtime.

`test_reverse.py` covers the failures that are silently wrong rather than loudly
wrong, which is the only reason it exists. A model finetuned on mis-oriented
shards still converges and still emits fluent English. It is just fluent English
facing the wrong way, and no loss curve will tell you.

The check that catches the most: `render(encode_prompt(x)) == x`, including the
space at the join. Every BPE tokenizer keeps a word's leading space on the word's
own token, so decoding the generated span and the prompt separately drops that
space and fuses two words, every single time.

## Files

| file | what it does |
| --- | --- |
| `bases.py` | the two base models and the memory arithmetic behind them |
| `basemodel.py` | wraps an HF causal LM in the interface the trainer already speaks |
| `revtext.py` | the only place human order becomes model order and back |
| `tokenize_reverse.py` | web + conversation mix, ChatML, reversal, shards |
| `data.py` | O(1)-resumable deterministic loader, carries direction and dtype |
| `config.py` | schedule defaults. Finetuning rates, not pretraining ones |
| `train.py` | DDP, fp16, WSD schedule, resume, deadline, atomic saves |
| `train_tpu.py` | the same run on a TPU v5e-8, bf16, no GradScaler |
| `tpu_preflight.py` | proves the device works before a session is spent on it |
| `chat.py` | terminal chat, reverse-aware, prefix and jeopardy modes |
| `dashboard.py` | Gradio UI: live graphs, completions, chat, all reverse-aware |
| `publish_hf.py` | checkpoint to HuggingFace repo, with a reload check |
| `kaggle_run.sh` | one command for a whole session, takes an hour budget |
| `shakedown.py` | end-to-end rehearsal including a direction check |
| `test_reverse.py` | the reversal properties |

## Everything else

Resume behaviour, the WSD schedule, checkpoint pruning, the fp16 loss scale
problem on Turing, the Feistel-permuted dataloader: all identical to llm67m and
documented in [QUICKSTART.md](QUICKSTART.md), [ONERUN.md](ONERUN.md) and
[TPU.md](TPU.md).

Two additions. The data fingerprint now includes the direction and the dtype, so
resuming a reverse run onto forward shards warns rather than quietly training a
model that faces both ways. And `train.py` refuses to start when the shards'
tokenizer does not match the requested base.
