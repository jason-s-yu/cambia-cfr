"""
Tests for src/reservoir.py

Covers:
- ReservoirBuffer: add, capacity limits, reservoir sampling statistics,
  sample_batch, save/load round-trips, resize, clear
- ReservoirSample: data integrity
- samples_to_tensors: batched conversion
"""

import os
import tempfile

import numpy as np
import pytest

from src.encoding import INPUT_DIM, NUM_ACTIONS
from src.reservoir import ReservoirBuffer, ReservoirSample, samples_to_tensors


def _make_sample(iteration=1, value=0.0):
    """Create a ReservoirSample with distinctive data."""
    return ReservoirSample(
        features=np.full(INPUT_DIM, value, dtype=np.float32),
        target=np.full(NUM_ACTIONS, value, dtype=np.float32),
        action_mask=np.ones(NUM_ACTIONS, dtype=bool),
        iteration=iteration,
    )


def _make_sample_random(iteration=1):
    """Create a ReservoirSample with random data."""
    return ReservoirSample(
        features=np.random.randn(INPUT_DIM).astype(np.float32),
        target=np.random.randn(NUM_ACTIONS).astype(np.float32),
        action_mask=np.random.randint(0, 2, NUM_ACTIONS).astype(bool),
        iteration=iteration,
    )


# ===== ReservoirSample =====


class TestReservoirSample:
    def test_fields_stored(self):
        s = _make_sample(iteration=5, value=3.14)
        assert s.iteration == 5
        assert s.features[0] == pytest.approx(3.14)
        assert s.target[0] == pytest.approx(3.14)
        assert s.action_mask[0] is np.bool_(True)
        assert s.infoset_key_raw is None

    def test_optional_metadata(self):
        s = ReservoirSample(
            features=np.zeros(INPUT_DIM, dtype=np.float32),
            target=np.zeros(NUM_ACTIONS, dtype=np.float32),
            action_mask=np.ones(NUM_ACTIONS, dtype=bool),
            iteration=1,
            infoset_key_raw=("test", 42),
        )
        assert s.infoset_key_raw == ("test", 42)


# ===== ReservoirBuffer: Basic Operations =====


class TestReservoirBufferBasic:
    def test_empty_buffer(self):
        buf = ReservoirBuffer(capacity=100)
        assert len(buf) == 0
        assert buf.seen_count == 0

    def test_add_below_capacity(self):
        buf = ReservoirBuffer(capacity=10)
        for i in range(5):
            buf.add(_make_sample(iteration=i))
        assert len(buf) == 5
        assert buf.seen_count == 5

    def test_add_at_capacity(self):
        buf = ReservoirBuffer(capacity=5)
        for i in range(5):
            buf.add(_make_sample(iteration=i))
        assert len(buf) == 5
        assert buf.seen_count == 5

    def test_add_beyond_capacity(self):
        """Buffer size stays at capacity when more items are added."""
        buf = ReservoirBuffer(capacity=5)
        for i in range(100):
            buf.add(_make_sample(iteration=i))
        assert len(buf) == 5
        assert buf.seen_count == 100

    def test_len(self):
        buf = ReservoirBuffer(capacity=10)
        assert len(buf) == 0
        buf.add(_make_sample())
        assert len(buf) == 1

    def test_clear(self):
        buf = ReservoirBuffer(capacity=10)
        for i in range(5):
            buf.add(_make_sample())
        buf.clear()
        assert len(buf) == 0
        assert buf.seen_count == 0


# ===== ReservoirBuffer: Sampling =====


class TestReservoirBufferSampling:
    def test_sample_batch_size(self):
        buf = ReservoirBuffer(capacity=100)
        for i in range(50):
            buf.add(_make_sample(iteration=i))
        batch = buf.sample_batch(10)
        assert len(batch) == 10

    def test_sample_batch_exceeds_buffer(self):
        """Requesting more than buffer size returns all available."""
        buf = ReservoirBuffer(capacity=100)
        for i in range(3):
            buf.add(_make_sample(iteration=i))
        batch = buf.sample_batch(100)
        assert len(batch) == 3

    def test_sample_batch_empty(self):
        buf = ReservoirBuffer(capacity=100)
        batch = buf.sample_batch(10)
        assert len(batch) == 0

    def test_sample_batch_returns_samples(self):
        buf = ReservoirBuffer(capacity=100)
        for i in range(10):
            buf.add(_make_sample(iteration=i))
        batch = buf.sample_batch(5)
        for s in batch:
            assert isinstance(s, ReservoirSample)

    def test_sample_batch_no_duplicates(self):
        """sample_batch uses random.sample (without replacement)."""
        buf = ReservoirBuffer(capacity=100)
        for i in range(20):
            buf.add(_make_sample(iteration=i, value=float(i)))
        batch = buf.sample_batch(15)
        # Use iteration as unique identifier
        iters = [s.iteration for s in batch]
        assert len(set(iters)) == 15


# ===== ReservoirBuffer: Reservoir Sampling Property =====


class TestReservoirSamplingProperty:
    def test_reservoir_property_statistical(self):
        """Each item has approximately equal probability of being in the buffer.

        With capacity=100 and 1000 items added, each item has a 10% chance
        of being retained. Over many runs, the distribution should be
        approximately uniform.
        """
        np.random.seed(42)
        capacity = 100
        total_items = 1000
        num_trials = 500

        counts = np.zeros(total_items)
        for _ in range(num_trials):
            buf = ReservoirBuffer(capacity=capacity)
            for i in range(total_items):
                buf.add(_make_sample(iteration=i, value=float(i)))
            for s in buf.buffer:
                counts[s.iteration] += 1

        # Expected count per item: num_trials * (capacity / total_items) = 50
        expected = num_trials * capacity / total_items
        # Allow 40% deviation (generous for statistical test)
        assert np.all(counts > expected * 0.4), "Some items never retained"
        assert np.all(counts < expected * 1.6), "Some items retained too often"

    def test_early_items_not_overrepresented(self):
        """Early items should not be overrepresented vs late items."""
        np.random.seed(123)
        capacity = 50
        total = 500
        num_trials = 300

        early_count = 0
        late_count = 0
        for _ in range(num_trials):
            buf = ReservoirBuffer(capacity=capacity)
            for i in range(total):
                buf.add(_make_sample(iteration=i))
            for s in buf.buffer:
                if s.iteration < total // 2:
                    early_count += 1
                else:
                    late_count += 1

        total_retained = early_count + late_count
        early_ratio = early_count / total_retained
        # Should be approximately 0.5 (within 10%)
        assert 0.4 < early_ratio < 0.6, (
            f"Early/late ratio {early_ratio} is too skewed"
        )


# ===== ReservoirBuffer: Save/Load =====


class TestReservoirBufferSaveLoad:
    def test_save_load_round_trip(self):
        """Save and load preserves buffer contents exactly."""
        buf = ReservoirBuffer(capacity=100)
        for i in range(10):
            buf.add(_make_sample_random(iteration=i))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_buffer.npz")
            buf.save(path)

            buf2 = ReservoirBuffer(capacity=100)
            buf2.load(path)

            assert len(buf2) == len(buf)
            assert buf2.seen_count == buf.seen_count
            for s1, s2 in zip(buf.buffer, buf2.buffer):
                np.testing.assert_array_equal(s1.features, s2.features)
                np.testing.assert_array_equal(s1.target, s2.target)
                np.testing.assert_array_equal(s1.action_mask, s2.action_mask)
                assert s1.iteration == s2.iteration

    def test_save_load_empty_buffer(self):
        buf = ReservoirBuffer(capacity=100)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.npz")
            buf.save(path)

            buf2 = ReservoirBuffer(capacity=100)
            buf2.load(path)
            assert len(buf2) == 0
            assert buf2.seen_count == 0

    def test_save_load_preserves_seen_count(self):
        """seen_count is preserved through save/load even when buffer is full."""
        buf = ReservoirBuffer(capacity=5)
        for i in range(100):
            buf.add(_make_sample(iteration=i))
        assert buf.seen_count == 100

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "seen.npz")
            buf.save(path)

            buf2 = ReservoirBuffer(capacity=5)
            buf2.load(path)
            assert buf2.seen_count == 100
            assert len(buf2) == 5

    def test_load_with_different_capacity_truncates(self):
        """Loading with a smaller capacity truncates the buffer."""
        buf = ReservoirBuffer(capacity=20)
        for i in range(20):
            buf.add(_make_sample(iteration=i))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trunc.npz")
            buf.save(path)

            buf2 = ReservoirBuffer(capacity=10)  # Smaller capacity
            buf2.load(path)
            assert len(buf2) == 10

    def test_load_with_larger_capacity(self):
        """Loading with a larger capacity keeps all samples."""
        buf = ReservoirBuffer(capacity=10)
        for i in range(10):
            buf.add(_make_sample(iteration=i))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "grow.npz")
            buf.save(path)

            buf2 = ReservoirBuffer(capacity=100)  # Larger capacity
            buf2.load(path)
            assert len(buf2) == 10
            assert buf2.capacity == 100

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "buf.npz")
            buf = ReservoirBuffer(capacity=10)
            buf.add(_make_sample())
            buf.save(path)
            assert os.path.exists(path)

    def test_load_adds_npz_extension(self):
        """Load handles paths without .npz extension."""
        buf = ReservoirBuffer(capacity=10)
        buf.add(_make_sample())
        with tempfile.TemporaryDirectory() as tmpdir:
            path_with_ext = os.path.join(tmpdir, "buf.npz")
            buf.save(path_with_ext)

            buf2 = ReservoirBuffer(capacity=10)
            # Load without .npz extension
            buf2.load(os.path.join(tmpdir, "buf"))
            assert len(buf2) == 1


# ===== ReservoirBuffer: Resize =====


class TestReservoirBufferResize:
    def test_resize_grow(self):
        buf = ReservoirBuffer(capacity=10)
        for i in range(10):
            buf.add(_make_sample())
        buf.resize(20)
        assert buf.capacity == 20
        assert len(buf) == 10  # Existing samples preserved

    def test_resize_shrink(self):
        buf = ReservoirBuffer(capacity=20)
        for i in range(20):
            buf.add(_make_sample(iteration=i))
        buf.resize(10)
        assert buf.capacity == 10
        assert len(buf) == 10

    def test_resize_to_same(self):
        buf = ReservoirBuffer(capacity=10)
        for i in range(5):
            buf.add(_make_sample())
        buf.resize(10)
        assert buf.capacity == 10
        assert len(buf) == 5

    def test_resize_shrink_below_current_size(self):
        buf = ReservoirBuffer(capacity=100)
        for i in range(50):
            buf.add(_make_sample())
        buf.resize(30)
        assert buf.capacity == 30
        assert len(buf) == 30


# ===== samples_to_tensors =====


class TestSamplesToTensors:
    def test_shapes(self):
        samples = [_make_sample_random() for _ in range(5)]
        features, targets, masks, iterations = samples_to_tensors(samples)
        assert features.shape == (5, INPUT_DIM)
        assert targets.shape == (5, NUM_ACTIONS)
        assert masks.shape == (5, NUM_ACTIONS)
        assert iterations.shape == (5,)

    def test_dtypes(self):
        samples = [_make_sample_random() for _ in range(3)]
        features, targets, masks, iterations = samples_to_tensors(samples)
        assert features.dtype == np.float32
        assert targets.dtype == np.float32
        assert masks.dtype == bool
        assert iterations.dtype == np.int64

    def test_empty_samples(self):
        features, targets, masks, iterations = samples_to_tensors([])
        assert features.shape == (0, INPUT_DIM)
        assert targets.shape == (0, NUM_ACTIONS)
        assert masks.shape == (0, NUM_ACTIONS)
        assert iterations.shape == (0,)

    def test_data_integrity(self):
        """Data in tensors matches the original samples."""
        s1 = _make_sample(iteration=10, value=1.0)
        s2 = _make_sample(iteration=20, value=2.0)
        features, targets, masks, iterations = samples_to_tensors([s1, s2])
        np.testing.assert_array_equal(features[0], s1.features)
        np.testing.assert_array_equal(features[1], s2.features)
        np.testing.assert_array_equal(targets[0], s1.target)
        np.testing.assert_array_equal(targets[1], s2.target)
        assert iterations[0] == 10
        assert iterations[1] == 20

    def test_single_sample(self):
        s = _make_sample_random(iteration=42)
        features, targets, masks, iterations = samples_to_tensors([s])
        assert features.shape == (1, INPUT_DIM)
        assert iterations[0] == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
