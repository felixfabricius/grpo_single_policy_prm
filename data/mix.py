# grpo_single_policy_prm/data/mix.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

from .schema import Example


@dataclass
class MixedStream:
    """
    Weighted mixture over multiple dataset loaders.

    - Only includes loaders that are .is_available() OR non-empty .prepare(split)
      unless 'strict' was set on that loader (then prepare() would have raised).
    - Sampling is *with replacement* across loaders based on weights.
    - Each loader is consumed sequentially; if one exhausts, we keep sampling
      from the remaining available loaders.
    """
    streams: List[Tuple[str, float, Iterable[Example]]]

    def __iter__(self) -> Iterator[Example]:
        # Materialize iterators
        iters: List[Tuple[str, float, Iterator[Example], Example | None]] = []
        for name, w, it in self.streams:
            iters.append((name, w, iter(it), None))

        active = [i for i in range(len(iters))]
        weights = [self.streams[i][1] for i in active]

        rng = random.Random(12345)
        while active:
            # Sample an active iterator index by weight
            if sum(weights) <= 0:
                # fallback to uniform if weird weights
                pick = rng.choice(active)
            else:
                pick = rng.choices(active, weights=weights, k=1)[0]

            name, w, it, _ = iters[pick]
            try:
                ex = next(it)
                yield ex
            except StopIteration:
                # Remove from active
                del active[active.index(pick)]
                # Recompute weights
                weights = [self.streams[i][1] for i in active]
                continue


def build_mixed_stream(
    loaders_with_weights: Dict[str, Tuple[Iterable[Example], float]],
) -> MixedStream:
    """
    Args:
        loaders_with_weights: mapping from dataset_name -> (iterable, weight)

    Returns:
        MixedStream over available iterables.
    """
    streams: List[Tuple[str, float, Iterable[Example]]] = []
    for name, (it, w) in loaders_with_weights.items():
        streams.append((name, float(w), it))
    return MixedStream(streams=streams)
