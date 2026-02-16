"""
src/reservoir.py

Reservoir sampling buffers for Deep CFR training.

Uses Vitter's Algorithm R to maintain a fixed-capacity uniform random sample
of all training samples ever generated. Two separate buffers are used:
  - Mv: advantage/regret samples
  - Mpi: strategy samples

Samples store iteration number for linear CFR weighting during training.
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .encoding import INPUT_DIM, NUM_ACTIONS
from .cfr.exceptions import ReservoirIOError

logger = logging.getLogger(__name__)


@dataclass
class ReservoirSample:
    """A single training sample for the reservoir buffer."""

    features: np.ndarray  # (INPUT_DIM,) float32 -- encoded infoset
    target: np.ndarray  # (NUM_ACTIONS,) float32 -- regrets or strategy
    action_mask: np.ndarray  # (NUM_ACTIONS,) bool -- legal actions
    iteration: int  # CFR iteration number for weighting
    infoset_key_raw: Optional[Tuple] = None  # Optional debugging metadata


class ReservoirBuffer:
    """
    Fixed-capacity reservoir sampling buffer using Vitter's Algorithm R.

    Guarantees a uniform random sample over all items ever added,
    regardless of how many items have been seen.
    """

    def __init__(self, capacity: int = 2_000_000):
        self.capacity = capacity
        self.buffer: List[ReservoirSample] = []
        self.seen_count: int = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, sample: ReservoirSample):
        """
        Add a sample to the buffer using reservoir sampling.

        If the buffer is not full, the sample is appended directly.
        Once full, each new sample has a (capacity / seen_count) probability
        of replacing a random existing sample.
        """
        self.seen_count += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            idx = random.randint(0, self.seen_count - 1)
            if idx < self.capacity:
                self.buffer[idx] = sample

    def sample_batch(self, batch_size: int) -> List[ReservoirSample]:
        """
        Sample a random batch from the buffer.

        Args:
            batch_size: Number of samples to draw (without replacement).

        Returns:
            List of ReservoirSample, length = min(batch_size, buffer size).
        """
        actual_size = min(batch_size, len(self.buffer))
        if actual_size == 0:
            return []
        return random.sample(self.buffer, actual_size)

    def save(self, path: str):
        """
        Save the buffer to disk as a compressed numpy archive.

        Stores features, targets, masks, and iterations as arrays,
        plus scalar metadata for seen_count and capacity.

        Raises:
            ReservoirIOError: If file I/O operations fail.
        """
        try:
            filepath = Path(path)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            n = len(self.buffer)
            if n == 0:
                np.savez_compressed(
                    str(filepath),
                    features=np.empty((0, INPUT_DIM), dtype=np.float32),
                    targets=np.empty((0, NUM_ACTIONS), dtype=np.float32),
                    masks=np.empty((0, NUM_ACTIONS), dtype=bool),
                    iterations=np.empty(0, dtype=np.int64),
                    meta=np.array([self.seen_count, self.capacity], dtype=np.int64),
                )
                logger.info("Saved empty reservoir buffer to %s", filepath)
                return

            features = np.stack([s.features for s in self.buffer])
            targets = np.stack([s.target for s in self.buffer])
            masks = np.stack([s.action_mask for s in self.buffer])
            iterations = np.array(
                [s.iteration for s in self.buffer], dtype=np.int64
            )

            np.savez_compressed(
                str(filepath),
                features=features,
                targets=targets,
                masks=masks,
                iterations=iterations,
                meta=np.array([self.seen_count, self.capacity], dtype=np.int64),
            )
            logger.info(
                "Saved reservoir buffer (%d samples, %d seen) to %s",
                n,
                self.seen_count,
                filepath,
            )
        except (OSError, IOError, PermissionError) as e:
            logger.error("Failed to save reservoir buffer to %s: %s", path, e)
            raise ReservoirIOError(f"Failed to save reservoir buffer to {path}: {e}") from e
        except Exception as e:
            logger.error("Unexpected error saving reservoir buffer to %s: %s", path, e)
            raise ReservoirIOError(f"Unexpected error saving reservoir buffer to {path}: {e}") from e

    def load(self, path: str):
        """
        Load the buffer from a numpy archive saved by save().

        Replaces the current buffer contents entirely.

        Raises:
            ReservoirIOError: If file I/O operations fail or file is corrupted.
        """
        try:
            filepath = Path(path)
            # Handle .npz extension
            if not filepath.suffix:
                filepath = filepath.with_suffix(".npz")
            if not str(path).endswith(".npz") and not filepath.exists():
                filepath = Path(str(path) + ".npz")

            data = np.load(str(filepath))

            meta = data["meta"]
            self.seen_count = int(meta[0])
            saved_capacity = int(meta[1])

            features = data["features"]
            targets = data["targets"]
            masks = data["masks"]
            iterations = data["iterations"]

            n = len(features)
            self.buffer = []
            for i in range(n):
                self.buffer.append(
                    ReservoirSample(
                        features=features[i],
                        target=targets[i],
                        action_mask=masks[i],
                        iteration=int(iterations[i]),
                    )
                )

            # If loaded capacity differs from current, log and keep current capacity
            if saved_capacity != self.capacity:
                logger.info(
                    "Loaded buffer had capacity %d, current capacity is %d. "
                    "Adjusting buffer if needed.",
                    saved_capacity,
                    self.capacity,
                )
                if len(self.buffer) > self.capacity:
                    # Truncate to current capacity via random subsample
                    self.buffer = random.sample(self.buffer, self.capacity)

            logger.info(
                "Loaded reservoir buffer: %d samples, %d seen, capacity %d from %s",
                len(self.buffer),
                self.seen_count,
                self.capacity,
                filepath,
            )
        except FileNotFoundError as e:
            logger.error("Reservoir buffer file not found: %s", path)
            raise ReservoirIOError(f"Reservoir buffer file not found: {path}") from e
        except (OSError, IOError, PermissionError) as e:
            logger.error("Failed to load reservoir buffer from %s: %s", path, e)
            raise ReservoirIOError(f"Failed to load reservoir buffer from {path}: {e}") from e
        except (KeyError, ValueError, IndexError) as e:
            logger.error("Corrupted reservoir buffer file %s: %s", path, e)
            raise ReservoirIOError(f"Corrupted reservoir buffer file {path}: {e}") from e
        except Exception as e:
            logger.error("Unexpected error loading reservoir buffer from %s: %s", path, e)
            raise ReservoirIOError(f"Unexpected error loading reservoir buffer from {path}: {e}") from e

    def resize(self, new_capacity: int):
        """
        Resize the buffer capacity.

        If shrinking, randomly subsample the current buffer to the new capacity.
        If growing, just update the capacity (new samples will fill naturally).
        The reservoir sampling property is preserved.

        Args:
            new_capacity: The new maximum capacity.
        """
        old_capacity = self.capacity
        self.capacity = new_capacity

        if len(self.buffer) > new_capacity:
            self.buffer = random.sample(self.buffer, new_capacity)
            logger.info(
                "Resized buffer from %d to %d (truncated %d samples)",
                old_capacity,
                new_capacity,
                old_capacity - new_capacity,
            )
        else:
            logger.info(
                "Resized buffer capacity from %d to %d (current size %d, no truncation)",
                old_capacity,
                new_capacity,
                len(self.buffer),
            )

    def clear(self):
        """Clear all samples and reset the counter."""
        self.buffer.clear()
        self.seen_count = 0


def samples_to_tensors(
    samples: List[ReservoirSample],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a list of ReservoirSamples into batched numpy arrays suitable
    for conversion to PyTorch tensors.

    Args:
        samples: List of ReservoirSample instances.

    Returns:
        Tuple of (features, targets, masks, iterations):
          - features: (N, INPUT_DIM) float32
          - targets: (N, NUM_ACTIONS) float32
          - masks: (N, NUM_ACTIONS) bool
          - iterations: (N,) int64
    """
    if not samples:
        return (
            np.empty((0, INPUT_DIM), dtype=np.float32),
            np.empty((0, NUM_ACTIONS), dtype=np.float32),
            np.empty((0, NUM_ACTIONS), dtype=bool),
            np.empty(0, dtype=np.int64),
        )

    features = np.stack([s.features for s in samples])
    targets = np.stack([s.target for s in samples])
    masks = np.stack([s.action_mask for s in samples])
    iterations = np.array([s.iteration for s in samples], dtype=np.int64)

    return features, targets, masks, iterations
