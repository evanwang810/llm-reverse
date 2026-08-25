"""Terminal presentation for chat.py: colour, boxes, menus, wrapped streaming.

Kept separate so chat.py stays about the model and this stays about the screen.
Pure stdlib, same as termplot, so the chat tool still runs anywhere a checkpoint
does with nothing to install.
"""

from __future__ import annotations

import os
import shutil
import sys

# --------------------------------------------------------------------------- #
# colour
# --------------------------------------------------------------------------- #

ENABLED = True


def _init_windows_vt() -> bool:
    """Turn on ANSI processing for the current console.

    Windows terminals understand escape codes but do not act on them until
    ENABLE_VIRTUAL_TERMINAL_PROCESSING is set on the handle, so without this the
    whole interface arrives as visible escape sequences.
    """
    if os.name != "nt":
        return True
    try:
        import ctypes

        k = ctypes.windll.kernel32
        handle = k.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not k.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(k.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


UNICODE = True
G = {}  # glyphs, filled by detect()


def _can_encode(probe: str) -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        probe.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def detect() -> bool:
    """Work out what this terminal can actually display, and pick glyphs to match.

    A Windows console reports cp1252, which cannot encode box drawing at all, so
    printing a panel raises UnicodeEncodeError and takes the program with it.
    Switching the stream to UTF-8 fixes it on anything modern; where it does not,
    fall back to ASCII rather than crash or print mojibake.
    """
    global ENABLED, UNICODE

    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        ENABLED = False
    elif not _init_windows_vt():
        ENABLED = False

    if not _can_encode("╭─╮│╰╯›▉"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    UNICODE = _can_encode("╭─╮│╰╯›▉")

    G.update(
        tl="╭" if UNICODE else "+", tr="╮" if UNICODE else "+",
        bl="╰" if UNICODE else "+", br="╯" if UNICODE else "+",
        h="─" if UNICODE else "-", v="│" if UNICODE else "|",
        arrow="›" if UNICODE else ">", bar="▉" if UNICODE else "#",
    )
    return ENABLED


detect()


def off() -> None:
    global ENABLED
    ENABLED = False


def _sgr(code: str):
    def wrap(text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if ENABLED else text
    return wrap


# A small deliberate palette rather than the whole rainbow: one hue for the
# person, one for the model, and everything else in grey so the conversation is
# what stands out.
dim = _sgr("38;5;244")
faint = _sgr("38;5;240")
bold = _sgr("1")
YOU_CODE = "39"             # blue, as a chip background
you = _sgr("38;5;39")
BOT_CODE = "38;5;79"        # teal green, also used by the streamer
BOT_CODE_N = "79"           # the same hue as a chip background
bot = _sgr(BOT_CODE)
accent = _sgr("38;5;180")   # warm sand
warn = _sgr("38;5;215")
bad = _sgr("38;5;203")
good = _sgr("38;5;114")
inverse = _sgr("7")


def width(default: int = 88) -> int:
    try:
        return max(48, min(shutil.get_terminal_size().columns, 110))
    except Exception:
        return default


def rule() -> str:
    return faint(G["h"] * width())


def panel(title: str, rows: list[tuple[str, str]]) -> str:
    """A titled box of label/value pairs."""
    w = width()
    label_w = max((len(a) for a, _ in rows), default= 0)
    head = f"{G['tl']}{G['h']} {title} " + G["h"] * max(0, w - len(title) - 4) + G["tr"]
    out = [faint(head)]
    for a, b in rows:
        pad = " " * (label_w - len(a))
        out.append(f"{faint(G['v'])} {dim(a)}{pad}  {b}")
    out.append(faint(G["bl"] + G["h"] * (w - 2) + G["br"]))
    return "\n".join(out)


def banner(subtitle: str = "a small GPT, trained from scratch") -> str:
    """A wordmark, not ASCII art.

    Figlet-style lettering only survives at a fixed width and in a font whose
    box characters line up; a rule with a title in it reads as deliberate at any
    terminal size and cannot come out garbled.
    """
    w = width()
    title = "llm-reverse"
    bar = G["h"] * 3 + f" {title} " + G["h"] * max(0, w - len(title) - 6)
    return f"\n{accent(bar)}\n{dim('    ' + subtitle)}"


# --------------------------------------------------------------------------- #
# streaming
# --------------------------------------------------------------------------- #


def chip(label: str, code: str) -> str:
    """A role badge: the colour as a background with dark text on it.

    Reads as a label rather than as coloured text, which is what separates one
    speaker's block from the next when the reply itself is also coloured.
    """
    if not ENABLED:
        return f"[{label}]"
    return f"\x1b[48;5;{code};38;5;235m {label} \x1b[0m"


class Streamer:
    """Writes generated text into a hanging-indent column, wrapping on words.

    Tokens arrive as fragments, not words, so wrapping has to happen against a
    running column count with a small buffer for the current word. Without it a
    reply is one long line that the terminal hard-wraps mid-word into the left
    margin, which stops it looking like a conversation.
    """

    def __init__(self, indent: int = 2, code: str = "", rail: bool = True) -> None:
        self.indent = indent
        esc = "\x1b"
        # A rail down the left of every wrapped line, so a multi-line reply
        # reads as one block rather than as loose lines under a label.
        if rail and code and ENABLED:
            self.rail = f"{esc}[{code}m{G['v']}{esc}[0m "
        elif rail:
            self.rail = f"{G['v']} "
        else:
            self.rail = ""
        # Opened once and closed once around the text rather than per word: the
        # result looks identical and the stream stays readable if it is piped.
        self.open = f"{esc}[{code}m" if (code and ENABLED) else ""
        self.close = f"{esc}[0m" if self.open else ""
        self.gutter = 2 if rail else 0
        self.col = self.gutter
        self.limit = width() - 4
        self.word = ""
        self.started = False

    def _put(self, s: str) -> None:
        enc = sys.stdout.encoding or "utf-8"
        try:
            sys.stdout.write(s)
        except UnicodeEncodeError:
            # A Windows console is cp1252 and BPE decodes to plenty it cannot
            # represent. Losing a glyph beats losing the reply.
            sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace"))

    def _newline(self) -> None:
        """Close the colour, draw the rail, reopen, so the rail keeps its hue."""
        self._put(self.close + "\n" + self.rail + self.open)
        self.col = self.gutter

    def _flush_word(self) -> None:
        if not self.word:
            return
        if self.col + len(self.word) > self.limit:
            self._newline()
        self._put(self.word)
        self.col += len(self.word)
        self.word = ""

    def feed(self, piece: str) -> None:
        if not self.started:
            self._put(self.rail + self.open)
            self.started = True
        for ch in piece:
            if ch == "\n":
                self._flush_word()
                self._newline()
            elif ch == " ":
                self.word += ch
                self._flush_word()
            else:
                self.word += ch
                if len(self.word) > 40:  # a pathological unbroken run
                    self._flush_word()
        sys.stdout.flush()

    def done(self) -> None:
        self._flush_word()
        self._put(self.close + "\n")
        sys.stdout.flush()


# --------------------------------------------------------------------------- #
# menus
# --------------------------------------------------------------------------- #


def choose(title: str, options: list[tuple[str, str]], allow_back: bool = True) -> int | None:
    """Numbered menu. Returns the zero-based index, or None for back/quit."""
    print()
    print(bold(title))
    for i, (label, note) in enumerate(options, 1):
        tail = f"   {dim(note)}" if note else ""
        print(f"  {accent(f'{i}')}  {label}{tail}")
    if allow_back:
        print(f"  {accent('q')}  {dim('back')}")
    while True:
        try:
            raw = input(f"\n{accent('>')} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw in ("q", "quit", "exit", "") and allow_back:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(bad("  pick a number from the list"))


def ask(prompt: str, current, cast):
    """Read one value, keeping the current one on empty input."""
    try:
        raw = input(f"{prompt} {dim(f'[{current}]')} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return current
    if not raw:
        return current
    try:
        return cast(raw)
    except ValueError:
        print(bad(f"  not a valid value, keeping {current}"))
        return current


def human_age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    if seconds < 172800:
        return f"{seconds / 3600:.0f}h ago"
    return f"{seconds / 86400:.0f}d ago"
