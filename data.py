"""Deterministic dataloader over token shards, uint16 or uint32.

The corpus is cut into non-overlapping blocks of block_size tokens. Global
sample order is a pseudo-random permutation of block indices, computed with a
4-round Feistel network so the k-th element is O(1) to evaluate. That means
resuming at step 41,237 costs nothing: there is no fast-forward loop, the
sampler just recomputes the indices for that step. Same seed plus same step
plus same rank always yields the same batch, which is what makes a resume
show no loss discontinuity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

MASK32 = 0xFFFFFFFF


def _mix(x: int, key: int) -> int:
    x = (x + key) & MASK32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & MASK32
    x ^= x >> 15
    x = (x * 0x846CA68B) & MASK32
    x ^= x >> 16
    return x


def permute_index(i: int, n: int, seed: int) -> int:
    """Bijection on [0, n) via a balanced Feistel permutation with cycle walking."""
    if n <= 1:
        return 0
    bits = (n - 1).bit_length()
    half = (bits + 1) // 2
    mask = (1 << half) - 1
    for _ in range(64):  # cycle walking terminates fast; bound it anyway
        left, right = (i >> half) & mask, i & mask
        for r in range(4):
            left, right = right, left ^ (_mix(right, (seed + r * 0x9E3779B9) & MASK32) & mask)
        i = (left << half) | right
        if i < n:
            return i
    return i % n


class Corpus:
    """Memory-mapped view over one split of a tokenized dataset."""

    def __init__(self, data_dir: str | Path, block_size: int, split: str = "train") -> None:
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"no meta.json in {self.data_dir}")
        meta = json.loads(meta_path.read_text())
        self.vocab_size: int = meta["vocab_size"]
        self.tokenizer: str = meta["tokenizer"]
        # "reverse", "forward" or "both". Nothing in the training loop cares,
        # because the shards are already laid out in the reading direction they
        # were built for. It matters at inference, so it rides along in the
        # checkpoint and in the fingerprint.
        self.direction: str = meta.get("direction", "forward")
        # uint16 caps at 65,535 token ids, which fits SmolLM2's 49k vocabulary
        # and not Qwen3's 151,669. The tokenizer decides, the shards record it,
        # and reading them back with the wrong width yields plausible-looking
        # garbage rather than an error, so it is never inferred here.
        self.dtype: str = meta.get("dtype", "uint16")
        self.base: str = meta.get("base", meta.get("tokenizer", "?"))
        names = meta["shards"][split]
        if not names:
            raise ValueError(f"split '{split}' has no shards in {meta_path}")
        self.shards = [np.memmap(self.data_dir / n, dtype=np.dtype(self.dtype), mode="r")
                       for n in names]
        self.block_size = block_size

        counts = [max(0, (len(s) - 1) // block_size) for s in self.shards]
        self.cum = np.cumsum([0] + counts)
        self.n_blocks = int(self.cum[-1])
        self.total_tokens = int(sum(len(s) for s in self.shards))
        if self.n_blocks == 0:
            raise ValueError(f"split '{split}' too small for block_size {block_size}")

    def block(self, idx: int) -> np.ndarray:
        """block_size + 1 tokens, never crossing a shard boundary."""
        j = int(np.searchsorted(self.cum, idx, side="right") - 1)
        off = (idx - int(self.cum[j])) * self.block_size
        return np.asarray(self.shards[j][off : off + self.block_size + 1], dtype=np.int64)

    def fingerprint(self) -> str:
        return (f"{self.base}:{self.direction}:{self.dtype}:"
                f"{self.vocab_size}:{self.total_tokens}")


class BatchSampler:
    def __init__(
        self,
        corpus: Corpus,
        micro_batch: int,
        grad_accum: int,
        world_size: int,
        rank: int,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        self.corpus = corpus
        self.micro_batch = micro_batch
        self.grad_accum = grad_accum
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle

    def _indices(self, step: int, micro: int) -> list[int]:
        base = ((step * self.grad_accum + micro) * self.world_size + self.rank) * self.micro_batch
        n = self.corpus.n_blocks
        out = []
        for j in range(self.micro_batch):
            epoch, pos = divmod(base + j, n)
            out.append(permute_index(pos, n, self.seed + epoch * 7919) if self.shuffle else pos)
        return out

    def batch(self, step: int, micro: int, device: torch.device):
        rows = np.stack([self.corpus.block(i) for i in self._indices(step, micro)])
        t = torch.from_numpy(rows)
        x, y = t[:, :-1].contiguous(), t[:, 1:].contiguous()
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def epoch_at(self, step: int) -> float:
        seen = step * self.grad_accum * self.world_size * self.micro_batch
        return seen / max(1, self.corpus.n_blocks)


def val_batches(corpus: Corpus, micro_batch: int, n_batches: int, device: torch.device):
    """Fixed, deterministic validation batches. Same every time it is called."""
    for b in range(n_batches):
        rows = np.stack([corpus.block(b * micro_batch + j) for j in range(micro_batch)])
        t = torch.from_numpy(rows)
        yield t[:, :-1].contiguous().to(device), t[:, 1:].contiguous().to(device)


if __name__ == "__main__":
    # Sanity check: the permutation really is a bijection.
    for n in (1, 2, 7, 1000, 65537):
        seen = {permute_index(i, n, 1337) for i in range(n)}
        assert seen == set(range(n)), f"not a permutation for n={n}"
    print("permute_index is bijective for all tested sizes")
