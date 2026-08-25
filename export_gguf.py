#!/usr/bin/env python
"""Turn a published folder into a GGUF, and write an Ollama Modelfile beside it.

    python export_gguf.py --hf-dir hf_export/qwen3-0.6b-reverse --quant Q4_K_M

Needs a llama.cpp checkout for its converter. Point at one with --llama-cpp or
$LLAMA_CPP, or clone it:

    git clone https://github.com/ggml-org/llama.cpp
    pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

**Read the warning this prints.** A GGUF of a reverse model runs fine and
answers every prompt with confident nonsense, because llama.cpp and Ollama both
feed it text left to right and read the output the same way. Neither has a hook
for reversing token order. `serve_reverse.py` is the client that does it
properly; this script only produces the file that client talks to.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODELFILE = """FROM ./{gguf}

# A reverse model. It predicts each token from the ones that FOLLOW it, so a
# prompt typed left to right is being read by the model as the end of a passage,
# and its output is in reverse token order.
#
# Ollama cannot reverse token order for you. Talking to this directly will
# produce fluent-looking nonsense. Use:
#
#     python serve_reverse.py --backend ollama --model {name}
#
# and read serve_reverse.py's note on why the llama.cpp backend is the exact one.

PARAMETER temperature 0.8
PARAMETER top_k 40
PARAMETER num_predict 64

# No template on purpose. Any wrapper text would be prepended in forward order,
# which is the wrong end of the sequence for this model.
TEMPLATE \"\"\"{{{{ .Prompt }}}}\"\"\"
"""


def find_converter(explicit: str) -> Path:
    roots = [explicit, os.environ.get("LLAMA_CPP", ""), "llama.cpp",
             "../llama.cpp", str(Path.home() / "llama.cpp")]
    for root in roots:
        if not root:
            continue
        script = Path(root) / "convert_hf_to_gguf.py"
        if script.exists():
            return script
    raise SystemExit(
        "could not find convert_hf_to_gguf.py. Clone llama.cpp and pass "
        "--llama-cpp <path>, or set $LLAMA_CPP:\n"
        "  git clone https://github.com/ggml-org/llama.cpp\n"
        "  pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt")


def find_quantizer(root: Path) -> Path | None:
    """llama-quantize moves around between builds, so look in the usual places."""
    names = ("llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe")
    for sub in ("build/bin", "build", ".", "bin"):
        for name in names:
            candidate = root / sub / name
            if candidate.exists():
                return candidate
    found = shutil.which("llama-quantize")
    return Path(found) if found else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-dir", required=True, help="a folder written by publish_hf.py")
    p.add_argument("--out", default="", help="output .gguf path")
    p.add_argument("--llama-cpp", default="", help="path to a llama.cpp checkout")
    p.add_argument("--outtype", default="f16", choices=("f32", "f16", "bf16", "q8_0"),
                   help="precision of the unquantized conversion")
    p.add_argument("--quant", default="",
                   help="also produce a quantized copy, e.g. Q4_K_M or Q5_K_M")
    p.add_argument("--name", default="", help="Ollama model name, default the folder name")
    args = p.parse_args()

    hf_dir = Path(args.hf_dir)
    if not (hf_dir / "config.json").exists():
        raise SystemExit(f"{hf_dir} has no config.json. Run publish_hf.py first.")

    name = args.name or hf_dir.name
    converter = find_converter(args.llama_cpp)
    out = Path(args.out) if args.out else hf_dir / f"{name}-{args.outtype}.gguf"

    print(f"converting {hf_dir} -> {out}")
    cmd = [sys.executable, str(converter), str(hf_dir),
           "--outfile", str(out), "--outtype", args.outtype]
    if subprocess.call(cmd) != 0:
        raise SystemExit("conversion failed")

    final = out
    if args.quant:
        quantizer = find_quantizer(converter.parent)
        if quantizer is None:
            print(f"\nno llama-quantize binary found, skipping --quant {args.quant}.")
            print("Build llama.cpp first: cmake -B build && cmake --build build -j")
        else:
            qout = out.with_name(f"{name}-{args.quant}.gguf")
            print(f"\nquantizing -> {qout}")
            if subprocess.call([str(quantizer), str(out), str(qout), args.quant]) != 0:
                raise SystemExit("quantization failed")
            final = qout

    modelfile = hf_dir / "Modelfile"
    modelfile.write_text(MODELFILE.format(gguf=final.name, name=name),
                         encoding="utf-8", newline="\n")

    size = final.stat().st_size / 1e9
    print(f"\n{final}  {size:.2f} GB")
    print(f"{modelfile}")
    print(f"\n  cd {hf_dir} && ollama create {name} -f Modelfile")
    print(f"\nThen talk to it through the wrapper, NOT through `ollama run`:")
    print(f"  python serve_reverse.py --backend ollama --model {name}")
    print("\nWhy: a reverse model reads its prompt as the END of a passage and")
    print("emits tokens in reverse order. Ollama has no hook to flip either, so")
    print("`ollama run` gives you fluent nonsense with no error. See")
    print("serve_reverse.py for the exact llama.cpp path.")


if __name__ == "__main__":
    main()
