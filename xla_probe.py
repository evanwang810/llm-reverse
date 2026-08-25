"""Ask a throwaway subprocess what the TPU is, without claiming it here.

A TPU is held by one process at a time, and initialising the runtime to read a
device count is enough to hold it. Any process that then forks replicas, or
launches a trainer, finds the device already taken and dies inside the XLA
runtime with "Check failed: reporting_closure_ == nullptr", which reads like a
bug in the trainer and is not.

So this module deliberately does not import torch_xla. It shells out, reads the
answer, and lets the child exit and release the device.
"""

from __future__ import annotations

import json
import subprocess
import sys

_SRC = r"""
import json, os
for v in ("TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"):
    os.environ.pop(v, None)
os.environ.setdefault("PJRT_DEVICE", "TPU")
out = {}
try:
    import torch_xla
    from torch_xla import runtime as xr
    out["torch_xla"] = torch_xla.__version__
    out["xla_device_type"] = xr.device_type()
    out["xla_devices"] = xr.global_runtime_device_count()
except Exception as e:
    out["xla_error"] = f"{type(e).__name__}: {e}"
try:
    import xla_compat
    out["xla_api"] = xla_compat.describe()
except Exception as e:
    out["xla_api_error"] = f"{type(e).__name__}: {e}"
print("XLAPROBE" + json.dumps(out))
"""


def probe(cwd: str | None = None, timeout: float = 180) -> dict:
    """Version, device type, device count, and which API spellings resolved."""
    try:
        r = subprocess.run([sys.executable, "-c", _SRC], capture_output=True,
                           text=True, timeout=timeout, cwd=cwd)
    except Exception as e:
        return {"xla_error": f"{type(e).__name__}: {e}"}
    for line in (r.stdout or "").splitlines():
        if line.startswith("XLAPROBE"):
            return json.loads(line[len("XLAPROBE"):])
    return {"xla_error": "probe produced no result", "stderr": (r.stderr or "")[-400:]}
