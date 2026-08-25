"""Resolve the torch_xla API once, whatever version happens to be installed.

Kaggle's TPU image moved from PT/XLA 2.1 to 2.8 with no announcement, and 2.7
removed xm.get_ordinal and xm.xrt_world_size outright rather than deprecating
them, so code written against the older names dies at the first line of the
spawned function with an AttributeError. Guessing a version and hardcoding its
spelling has now failed twice, so the guessing happens here and nowhere else.

Every name is looked up in preference order, newest first, and anything still
missing raises at import with the whole list rather than at replica zero, four
minutes into a run, one name at a time.

Resolution is attribute lookup only. Nothing here calls into the runtime, so
importing this module does not claim the TPU and is safe in the process that
later calls xmp.spawn.
"""

from __future__ import annotations

import os

# Kaggle's launcher presets these and torch_xla reads them at import, coming up
# half configured if they survive. Popping here means every entry point gets it
# right by importing this module, rather than by remembering to.
for _var in ("TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"):
    os.environ.pop(_var, None)
os.environ.setdefault("PJRT_DEVICE", "TPU")

import torch_xla  # noqa: E402
import torch_xla.core.xla_model as _xm  # noqa: E402

try:
    from torch_xla import runtime as _xr
except ImportError:  # pre-2.0
    _xr = None

VERSION = getattr(torch_xla, "__version__", "unknown")


def _pick(name: str, *candidates):
    """First candidate that exists, as (owner, attribute) pairs."""
    for owner, attr in candidates:
        if owner is not None and hasattr(owner, attr):
            return getattr(owner, attr)
    tried = ", ".join(f"{getattr(o, '__name__', o)}.{a}" for o, a in candidates if o is not None)
    raise ImportError(
        f"torch_xla {VERSION} has no {name}. Tried: {tried}. "
        "The API moved again; add the new spelling to xla_compat.py."
    )


# xm.get_ordinal and xm.xrt_world_size were removed in 2.7, not deprecated.
ordinal = _pick("global ordinal", (_xr, "global_ordinal"), (_xm, "get_ordinal"))
world_size = _pick("world size", (_xr, "world_size"), (_xm, "xrt_world_size"))

# xm.xla_device still works in 2.8 but warns once per replica, which is eight
# warnings before anything useful appears in the log.
device = _pick("device", (torch_xla, "device"), (_xm, "xla_device"))

# mark_step became sync in 2.6.
sync = _pick("sync", (torch_xla, "sync"), (_xm, "mark_step"))
wait = _pick("wait_device_ops", (_xm, "wait_device_ops"), (torch_xla, "wait_device_ops"))

_optimizer_step = _pick("optimizer_step", (_xm, "optimizer_step"))


def optimizer_step(optimizer):
    """Reduce gradients across replicas, step, and then execute the graph.

    The last part is the whole point. XLA is lazy: operations accumulate into a
    pending graph and nothing runs until something forces it. MpDeviceLoader
    issues that sync on every iteration, which is why almost every example gets
    it for free; this trainer feeds batches from its own index-based sampler
    instead, so nothing was issuing it at all.

    Left alone, the graph does not reset at the end of a step. It keeps growing,
    one more forward, backward and optimizer update per step, until something
    reads a value off the device and forces the whole accumulated thing to
    compile and run. That graph is a different size every time, so it is a fresh
    compilation every time, and compile cost grows with graph size. The result
    is a training loop that runs about a hundred times too slowly, gets worse
    the longer it runs, and eventually dies with no Python traceback when the
    compiler gives up on a graph that never stops growing.

    barrier=True is the documented way to issue the sync when MpDeviceLoader is
    not in the picture. The fallback covers versions whose signature lacks it.
    """
    try:
        out = _optimizer_step(optimizer, barrier=True)
    except TypeError:  # a version whose signature has no barrier
        out = _optimizer_step(optimizer)
    # Unconditional, not an else. Whether barrier=True was accepted, and whether
    # it still means what it meant, is not worth betting a session on; a second
    # sync with nothing pending costs nothing.
    sync()
    return out
mesh_reduce = _pick("mesh_reduce", (_xm, "mesh_reduce"))
rendezvous = _pick("rendezvous", (_xm, "rendezvous"))
save = _pick("save", (_xm, "save"))

_all_reduce = _pick("all_reduce", (_xm, "all_reduce"))
REDUCE_SUM = getattr(_xm, "REDUCE_SUM", "sum")


def all_reduce_sum(value, scale: float = 1.0):
    """Sum across replicas, optionally scaled. Accepts a tensor or a list."""
    return _all_reduce(REDUCE_SUM, value, scale=scale)


def mean(value):
    """Average a tensor across replicas."""
    return _all_reduce(REDUCE_SUM, value, scale=1.0 / world_size())


def describe() -> str:
    return (f"torch_xla {VERSION}: ordinal={ordinal.__module__}.{ordinal.__name__}, "
            f"world_size={world_size.__module__}.{world_size.__name__}, "
            f"device={device.__module__}.{device.__name__}, "
            f"sync={sync.__module__}.{sync.__name__}")
