#!/usr/bin/env python
"""Gradio dashboard: live training graphs, progress bar, save button, chat.

Runs entirely on CPU so it does not touch your GPU quota. Launch it in a second
notebook cell while training runs, or in a separate session pointed at the
checkpoint dataset:

    python dashboard.py --run-dir /kaggle/working/run --share

It talks to the trainer through files in the run dir, so the Save / Decay /
Stop buttons work even though the trainer is a separate process.
"""

from __future__ import annotations

import argparse
import html
import math
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import gradio as gr

import basemodel
import bases
import revtext
from runstate import RunDir, default_search_dirs, find_checkpoints

# Gradio moved theme and css from Blocks() to launch() in 6.0, so branch on it.
GR_MAJOR = int(gr.__version__.split(".")[0])

torch.set_num_threads(max(1, (os.cpu_count() or 4)))

INK = "#e8eaed"
MUTED = "#8b93a1"
ACCENT = "#7aa2f7"
ACCENT2 = "#f7768e"
GOOD = "#9ece6a"
PANEL = "#161822"

_model_cache: dict[str, tuple] = {}
_last_probs: list[list] = []


# --------------------------------------------------------------------------- #
# checkpoints and generation
# --------------------------------------------------------------------------- #


def get_tokenizer(hf_name: str = ""):
    """The base model's own tokenizer. There is no project-wide encoding."""
    if not hf_name:
        entry = _active()
        hf_name = entry["base"] if entry else bases.get("large").hf_name
    return revtext.load_tokenizer(hf_name)


def _show(enc, token: int) -> str:
    if token >= len(enc):
        return f"<{token}>"
    """Printable form of a token id, including the turn tokens above the BPE range."""
    if False:
        return {50257: "<|user|>", 50258: "<|assistant|>"}.get(token, f"<|{token}|>")
    return repr(enc.decode([token], skip_special_tokens=False))


def checkpoint_choices(run_dir: Path) -> list[str]:
    cands = find_checkpoints([run_dir, *default_search_dirs()])
    return [f"step {c['step']:,} [{c['kind']}, {c['size_mb']:.0f} MB]  {c['path']}"
            for c in cands]


def _path_from_choice(choice: str) -> Path:
    """Accepts a dropdown label or a bare path typed into the override box."""
    choice = (choice or "").strip().strip('"')
    return Path(choice.split("]", 1)[1].strip() if "]" in choice else choice)


def load_checkpoint(choice: str, override_path: str = ""):
    target = override_path.strip() or choice
    if not target:
        return "pick a checkpoint from the list, or type a path"
    path = _path_from_choice(target)
    if not path.exists():
        return f"no such file: {path}"
    key = str(path)
    if key not in _model_cache:
        model, ckpt = basemodel.load_from_checkpoint(path, device="cpu")
        _model_cache.clear()  # one model at a time keeps RAM sane
        _model_cache[key] = {
            "model": model,
            "step": ckpt.get("step"),
            "fingerprint": ckpt.get("data_fingerprint", "?"),
            "base": model.cfg.hf_name,
            "block_size": model.cfg.block_size,
            "direction": str(ckpt.get("direction", "forward")),
        }
    e = _model_cache[key]
    n = sum(p.numel() for p in e["model"].parameters())
    if revtext.is_reverse(e["direction"]):
        kind = ("reverse model: your prompt is the END of the passage and the "
                "text grows to the left of it")
    elif False:
        kind = "instruction-tuned, the chat tab will use its turn tokens"
    else:
        kind = "base model, it continues text rather than answering"
    return (f"loaded step {e['step']:,} from {path.name}\n"
            f"{n / 1e6:.1f}M params, block_size {e['model'].cfg.block_size}\n"
            f"{kind}")


def _active():
    return next(iter(_model_cache.values())) if _model_cache else None


@torch.no_grad()
def generate_stream(choice: str, prompt: str, max_tokens: int, temperature: float,
                    top_k: int, greedy: bool, prefix_ids: list[int] | None = None,
                    stats: bool = True):
    global _last_probs
    if not _model_cache:
        msg = load_checkpoint(choice)
        if not _model_cache:
            yield msg, []
            return
    model = _active()["model"]
    enc = get_tokenizer()

    direction = _active()["direction"]
    reverse = revtext.is_reverse(direction)
    if prefix_ids is not None:
        ids = list(prefix_ids)
    else:
        ids = revtext.encode_prompt(enc, prompt, direction)
    ids = ids[-(model.cfg.block_size - 1) :]
    idx = torch.tensor([ids], dtype=torch.long)

    rows: list[list] = []
    # A reverse model writes leftwards, so there is no growing suffix to append
    # to. Re-render the whole sequence each time instead; a few hundred tokens
    # through the tokenizer is far cheaper than the forward pass behind it.
    text = revtext.render(enc, ids, direction) if reverse else prompt
    t0 = time.time()
    for i in range(int(max_tokens)):
        window = idx[:, -model.cfg.block_size :]
        logits, _ = model(window)
        logits = logits[0, -1].float()
        probs_full = F.softmax(logits, dim=-1)
        entropy = float(-(probs_full * probs_full.clamp_min(1e-12).log()).sum())

        top_p, top_i = probs_full.topk(5)
        top5 = ", ".join(f"{_show(enc, int(t))}={float(p):.3f}" for p, t in zip(top_p, top_i))

        if greedy:
            nxt = int(logits.argmax())
        else:
            scaled = logits / max(1e-4, temperature)
            if top_k > 0:
                kth = torch.topk(scaled, min(int(top_k), scaled.numel()))[0][-1]
                scaled = scaled.masked_fill(scaled < kth, float("-inf"))
            nxt = int(torch.multinomial(F.softmax(scaled, dim=-1), 1))

        rows.append([i, _show(enc, nxt), round(float(probs_full[nxt]), 4),
                     round(entropy, 3), top5])
        if nxt == revtext.eos_id(enc) or (not reverse and nxt >= len(enc)):
            break
        idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
        if reverse:
            text = revtext.render(enc, idx[0].tolist(), direction)
        else:
            text += ("" if nxt >= len(enc)
                     else enc.decode([nxt], skip_special_tokens=False))
        if i % 4 == 0 or i == int(max_tokens) - 1:
            yield text, rows

    _last_probs = rows
    rate = len(rows) / max(1e-6, time.time() - t0)
    suffix = f"\n\n[{len(rows)} tokens, {rate:.1f} tok/s on cpu]" if stats else ""
    yield text + suffix, rows


def generate_with_override(choice: str, override_path: str, prompt: str, max_tokens: int,
                           temperature: float, top_k: int, greedy: bool):
    """A typed-in path wins over the dropdown selection."""
    yield from generate_stream(override_path.strip() or choice, prompt, max_tokens,
                               temperature, top_k, greedy)


def _turns(history, message: str) -> list[tuple[str, str]]:
    out = []
    for turn in history or []:
        if isinstance(turn, dict):
            out.append((turn["role"], turn["content"]))
        else:
            out.append(("user", turn[0]))
            out.append(("assistant", turn[1]))
    out.append(("user", message))
    return out


def chat_fn(message: str, history, choice: str, max_tokens: int, temperature: float, top_k: int):
    """Argument order is fixed by gr.ChatInterface: (message, history, *extras).

    An instruction-tuned checkpoint gets its real <|user|> / <|assistant|> turn
    tokens. A base checkpoint has never seen a conversation, so the honest thing
    is to lay the transcript out as plain text and let it continue.
    """
    if not _model_cache:
        msg = load_checkpoint(choice)
        if not _model_cache:
            yield msg
            return
    entry = _active()
    enc = get_tokenizer()
    turns = _turns(history, message)

    if revtext.is_reverse(entry["direction"]):
        # Jeopardy: the last message is treated as an answer and the model
        # writes the question behind it. History is dropped on purpose, since a
        # reversed exchange runs answer, marker, question and an earlier turn
        # would land on the wrong side of the text being written.
        ids = revtext.encode_prompt(enc, message, entry["direction"], jeopardy=True)
        for text, _ in generate_stream(choice, "", max_tokens, temperature, top_k,
                                       False, prefix_ids=ids, stats=False):
            yield text.strip()
        return

    # There is no separate instruction-tuned stage in this project; the
    # conversations live in the finetuning corpus. A forward run is prompted as
    # plain text, which is what the transcript below does.

    transcript = "".join(f"{role.capitalize()}: {content}\n" for role, content in turns)
    transcript += "Assistant:"
    for text, _ in generate_stream(choice, transcript, max_tokens, temperature, top_k,
                                   False, stats=False):
        yield text[len(transcript):].split("\nUser:")[0]


# --------------------------------------------------------------------------- #
# live view
# --------------------------------------------------------------------------- #


def _bar(label: str, frac: float, right: str, color: str) -> str:
    frac = max(0.0, min(1.0, frac))
    return f"""
    <div style="margin:10px 0">
      <div style="display:flex;justify-content:space-between;font:600 12px ui-monospace,monospace;
                  color:{MUTED};letter-spacing:.08em;text-transform:uppercase">
        <span>{html.escape(label)}</span><span style="color:{INK}">{html.escape(right)}</span>
      </div>
      <div style="height:14px;background:#0d0f16;border-radius:7px;overflow:hidden;margin-top:6px;
                  border:1px solid #262a38">
        <div style="height:100%;width:{frac * 100:.2f}%;background:linear-gradient(90deg,{color},{color}aa);
                    border-radius:7px;transition:width .4s ease"></div>
      </div>
    </div>"""


def _card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div style="flex:1 1 130px;min-width:130px;background:{PANEL};border:1px solid #262a38;
                border-radius:12px;padding:12px 14px">
      <div style="font:600 10px ui-monospace,monospace;color:{MUTED};letter-spacing:.1em;
                  text-transform:uppercase">{html.escape(label)}</div>
      <div style="font:700 22px ui-monospace,monospace;color:{INK};margin-top:4px">{html.escape(value)}</div>
      <div style="font:500 11px ui-monospace,monospace;color:{MUTED};margin-top:2px">{html.escape(sub)}</div>
    </div>"""


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def render_live(run_dir_str: str):
    rs = RunDir(run_dir_str)
    st = rs.read_status()
    if st is None:
        return (f"<div style='color:{MUTED};font:14px ui-monospace,monospace;padding:24px'>"
                f"no status.json in {html.escape(run_dir_str)} yet. "
                f"Start training, or point --run-dir at the right folder.</div>", None)

    stale = time.time() - st.get("heartbeat", 0)
    alive = st.get("alive", False) and stale < 180
    dot = GOOD if alive else ACCENT2
    label = "training" if alive else ("finished" if st.get("stop_reason") else "stalled or not running")

    elapsed = st.get("elapsed_s", 0.0)
    remaining = st.get("remaining_s", 0.0)
    total_time = elapsed + remaining
    bars = _bar("session time", elapsed / total_time if total_time else 1.0,
                f"{_fmt_hms(elapsed)} / {_fmt_hms(total_time)}", ACCENT)
    if st.get("max_steps"):
        bars += _bar("steps", st["step"] / st["max_steps"],
                     f"{st['step']:,} / {st['max_steps']:,}", GOOD)
    if st.get("decay_start") is not None:
        done = st["step"] - st["decay_start"]
        bars += _bar("lr decay", done / max(1, st.get("decay_steps", 1)),
                     f"{max(0, done):,} / {st.get('decay_steps', 0):,} steps", ACCENT2)

    loss = st.get("loss_ema") or st.get("loss")
    ppl = math.exp(min(20.0, loss)) if loss else None
    cards = "".join([
        _card("step", f"{st.get('step', 0):,}", f"{st.get('phase', '?')} phase"),
        _card("train loss", f"{loss:.4f}" if loss else "n/a",
              f"ppl {ppl:.1f}" if ppl else ""),
        _card("val loss", f"{st['val_loss']:.4f}" if st.get("val_loss") else "pending",
              f"best {st['best_val']:.4f}" if st.get("best_val") else ""),
        _card("throughput", f"{(st.get('tok_per_s') or 0) / 1e3:.1f}k/s",
              f"{st.get('secs_per_step', 0):.2f}s per step"),
        _card("tokens seen", f"{st.get('tokens_seen', 0) / 1e9:.3f}B",
              f"epoch {st.get('epoch', 0):.2f}"),
        _card("time left", _fmt_hms(remaining), f"saved @ {st.get('last_save_step', 0):,}"),
    ])

    header = f"""
    <div style="font:600 13px ui-monospace,monospace;color:{INK};display:flex;align-items:center;gap:8px">
      <span style="width:9px;height:9px;border-radius:50%;background:{dot};
                   box-shadow:0 0 10px {dot}"></span>{html.escape(label)}
      <span style="color:{MUTED};font-weight:500">
        {st.get('non_embedding_params', 0) / 1e6:.1f}M non-emb params on {st.get('world_size', 1)} gpu,
        {st.get('tokens_per_step', 0):,} tokens/step, heartbeat {stale:.0f}s ago
      </span>
    </div>"""
    body = (f"{header}{bars}"
            f"<div style='display:flex;gap:10px;flex-wrap:wrap;margin-top:14px'>{cards}</div>")
    return body, loss_figure(rs, x_axis="step", smooth=True, logy=False)


def loss_figure(rs: RunDir, x_axis: str = "step", smooth: bool = True, logy: bool = False):
    rows = rs.read_csv()
    if not rows:
        return None
    def col(name, cast=float):
        out = []
        for r in rows:
            v = r.get(name, "")
            out.append(cast(v) if v not in ("", None) else None)
        return out

    steps = col("step", lambda v: int(float(v)))
    tokens = col("tokens", lambda v: int(float(v)))
    x = steps if x_axis == "step" else [t / 1e9 for t in tokens]
    loss = col("loss")
    ema = col("loss_ema")
    lr = col("lr")
    val = col("val_loss")

    plt.style.use("dark_background")
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 5.2), height_ratios=[3, 1], sharex=True)
    fig.patch.set_facecolor("#0d0f16")
    for a in (ax, ax2):
        a.set_facecolor("#0d0f16")
        a.grid(alpha=0.14, linewidth=0.6)
        for spine in a.spines.values():
            spine.set_color("#262a38")
        a.tick_params(colors=MUTED, labelsize=8)

    ax.plot(x, loss, color=ACCENT, alpha=0.28, linewidth=0.9, label="train")
    if smooth:
        ax.plot(x, ema, color=ACCENT, linewidth=1.9, label="train (ema)")
    vx = [xi for xi, v in zip(x, val) if v is not None]
    vy = [v for v in val if v is not None]
    if vy:
        ax.plot(vx, vy, color=ACCENT2, linewidth=1.6, marker="o", markersize=2.6, label="val")
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel("cross entropy", color=MUTED, fontsize=9)
    ax.legend(facecolor=PANEL, edgecolor="#262a38", labelcolor=INK, fontsize=8, loc="upper right")
    if loss and loss[-1]:
        ax.set_title(f"latest {loss[-1]:.4f}   ppl {math.exp(min(20, loss[-1])):.1f}",
                     color=INK, fontsize=10, loc="left")

    ax2.plot(x, lr, color=GOOD, linewidth=1.4)
    ax2.set_ylabel("lr", color=MUTED, fontsize=9)
    ax2.set_xlabel("step" if x_axis == "step" else "tokens (B)", color=MUTED, fontsize=9)
    fig.tight_layout()
    return fig


def refresh_plot(run_dir_str: str, x_axis: str, smooth: bool, logy: bool):
    return loss_figure(RunDir(run_dir_str), x_axis, smooth, logy)


def csv_tail(run_dir_str: str, n: int = 40):
    rows = RunDir(run_dir_str).read_csv()
    keep = rows[-int(n):]
    cols = ["step", "loss", "loss_ema", "val_loss", "lr", "tok_per_s", "grad_norm", "scale", "phase"]
    return [[r.get(c, "") for c in cols] for r in keep], cols


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #

CSS = """
.gradio-container {max-width: 1180px !important}
footer {display:none !important}
#title {font: 700 20px ui-monospace, SFMono-Regular, monospace; letter-spacing:-.01em}
#subtitle {color:#8b93a1; font: 500 12px ui-monospace, monospace; margin-top:-6px}
"""


def build_app(run_dir: str, refresh_s: float):
    """Returns (demo, extra_launch_kwargs)."""
    theme = gr.themes.Base(
        primary_hue="blue", neutral_hue="slate", font=["ui-monospace", "monospace"]
    ).set(body_background_fill="#0d0f16", block_background_fill="#161822",
          block_border_color="#262a38", body_text_color="#e8eaed")
    style = {"theme": theme, "css": CSS}
    blocks_kwargs = {"title": "llm-reverse trainer"}
    launch_extra: dict = {}
    if GR_MAJOR >= 6:
        launch_extra = style
    else:
        blocks_kwargs.update(style)

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown("### llm-reverse", elem_id="title")
        gr.Markdown("live training monitor and checkpoint playground, cpu only", elem_id="subtitle")
        run_dir_box = gr.Textbox(value=run_dir, label="run dir", scale=1)

        with gr.Tab("live"):
            live_html = gr.HTML()
            live_plot = gr.Plot(label="")
            with gr.Row():
                b_save = gr.Button("save checkpoint now", variant="primary")
                b_decay = gr.Button("start lr decay")
                b_stop = gr.Button("stop and save", variant="stop")
                b_refresh = gr.Button("refresh")
            action_out = gr.Markdown()

            def _flag(run_dir_str, flag, note):
                RunDir(run_dir_str).request(flag)
                return f"`{flag}` requested at {time.strftime('%H:%M:%S')}. {note}"

            b_save.click(lambda d: _flag(d, "SAVE_NOW", "The trainer picks it up within a few steps."),
                         [run_dir_box], action_out)
            b_decay.click(lambda d: _flag(d, "DECAY_NOW",
                                          "LR now decays to min over --decay-steps, then training exits."),
                          [run_dir_box], action_out)
            b_stop.click(lambda d: _flag(d, "STOP_NOW", "Final checkpoint is written before exit."),
                         [run_dir_box], action_out)
            b_refresh.click(render_live, [run_dir_box], [live_html, live_plot])
            demo.load(render_live, [run_dir_box], [live_html, live_plot])
            if hasattr(gr, "Timer"):
                gr.Timer(refresh_s).tick(render_live, [run_dir_box], [live_html, live_plot])

        with gr.Tab("completion"):
            with gr.Row():
                ckpt = gr.Dropdown(choices=checkpoint_choices(Path(run_dir)),
                                   label="checkpoint", scale=4)
                b_scan = gr.Button("rescan", scale=1)
                b_load = gr.Button("load", variant="primary", scale=1)
            dropped = gr.File(label="or drag a .pt model file here", file_types=[".pt"],
                              type="filepath", height=110)
            ckpt_path = gr.Textbox(
                label="or point at a file directly (overrides the dropdown)",
                placeholder=r"C:\path\to\milestone_step0004180.pt", lines=1)
            load_msg = gr.Markdown()
            prompt = gr.Textbox(value="The mitochondrion is", lines=4, label="prompt")
            with gr.Row():
                max_tok = gr.Slider(8, 512, value=128, step=8, label="max new tokens")
                temp = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="temperature")
                topk = gr.Slider(0, 200, value=50, step=1, label="top-k (0 = off)")
                greedy = gr.Checkbox(value=False, label="greedy")
            b_gen = gr.Button("generate", variant="primary")
            out = gr.Textbox(lines=12, label="completion")
            gr.Markdown("per-token diagnostics: low prob with low entropy means confidently wrong, "
                        "low prob with high entropy means it simply does not know yet")
            probs = gr.Dataframe(
                headers=["i", "token", "p(chosen)", "entropy", "top-5"],
                datatype=["number", "str", "number", "number", "str"],
                wrap=True, label="",
            )
            b_scan.click(lambda d: gr.update(choices=checkpoint_choices(Path(d))),
                         [run_dir_box], ckpt)
            b_load.click(load_checkpoint, [ckpt, ckpt_path], load_msg)
            # Dropping a file fills the override box, then loads it. Everything
            # downstream already prefers the override, including the chat tab,
            # which shares the same one-model cache.
            dropped.upload(lambda f: f or "", [dropped], ckpt_path).then(
                load_checkpoint, [ckpt, ckpt_path], load_msg)
            b_gen.click(generate_with_override,
                        [ckpt, ckpt_path, prompt, max_tok, temp, topk, greedy], [out, probs])

        with gr.Tab("chat"):
            gr.Markdown("This is a base language model with no instruction tuning, so it will "
                        "continue text rather than answer you. Useful for vibe checks anyway.")
            chat_ckpt = gr.Dropdown(choices=checkpoint_choices(Path(run_dir)), label="checkpoint")
            chat_load_msg = gr.Markdown()
            chat_ckpt.change(load_checkpoint, [chat_ckpt], chat_load_msg)
            with gr.Row():
                c_max = gr.Slider(8, 256, value=80, step=8, label="max new tokens")
                c_temp = gr.Slider(0.1, 2.0, value=0.9, step=0.05, label="temperature")
                c_topk = gr.Slider(0, 200, value=50, step=1, label="top-k")
            extras = [chat_ckpt, c_max, c_temp, c_topk]
            try:
                gr.ChatInterface(fn=chat_fn, additional_inputs=extras, type="messages")
            except TypeError:  # gradio dropped the type kwarg
                gr.ChatInterface(fn=chat_fn, additional_inputs=extras)

        with gr.Tab("loss log"):
            with gr.Row():
                x_axis = gr.Radio(["step", "tokens"], value="step", label="x axis")
                smooth = gr.Checkbox(value=True, label="show ema")
                logy = gr.Checkbox(value=False, label="log y")
                b_plot = gr.Button("replot", variant="primary")
            big_plot = gr.Plot()
            tail = gr.Dataframe(label="last rows of loss.csv", wrap=True)
            b_plot.click(refresh_plot, [run_dir_box, x_axis, smooth, logy], big_plot)
            b_plot.click(lambda d: csv_tail(d)[0], [run_dir_box], tail)
            demo.load(refresh_plot, [run_dir_box, x_axis, smooth, logy], big_plot)

    return demo, launch_extra


def launch(run_dir: str, refresh: float = 5.0, share: bool = True, port: int = 7860,
           inline: bool | None = None, blocking: bool = True):
    """Convenience entry point for a notebook cell.

    inline=True embeds the whole UI in the cell output instead of only printing
    a link. On Kaggle that iframe points at the share URL, so share must stay
    True; there is no port forwarding to fall back on.

    blocking=False returns immediately and keeps serving from a background
    thread, so you can carry on running other cells while the UI stays up.
    """
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    demo, extra = build_app(run_dir, refresh)
    return demo.queue().launch(
        share=share, server_port=port, server_name="0.0.0.0",
        inline=inline, prevent_thread_lock=not blocking, **extra,
    )


def watch_inline(run_dir: str = "/kaggle/working/run", interval: float = 15.0,
                 count: int = 0, until_done: bool = True):
    """Live status cards and loss chart drawn straight into the cell output.

    No web server, no share link, no tunnel, so nothing here can be blocked by
    Kaggle's networking. The trade is that it is a view and not an app: no
    buttons, no prompt box. For those, use launch(inline=True).
    """
    from IPython.display import HTML, clear_output, display

    rs = RunDir(run_dir)
    shown = 0
    while True:
        body, fig = render_live(run_dir)
        clear_output(wait=True)
        display(HTML(f"<div style='background:#0d0f16;padding:16px;border-radius:12px'>{body}</div>"))
        if fig is not None:
            display(fig)
            plt.close(fig)
        shown += 1

        st = rs.read_status()
        if until_done and st and st.get("stop_reason"):
            print(f"training finished: {st['stop_reason']}, "
                  f"final checkpoint {st.get('final_checkpoint')}")
            return
        if count and shown >= count:
            return
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            return


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="/kaggle/working/run")
    p.add_argument("--refresh", type=float, default=5.0)
    p.add_argument("--share", action="store_true", help="public gradio link, needed on Kaggle")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    launch(args.run_dir, args.refresh, args.share, args.port)


if __name__ == "__main__":
    main()
