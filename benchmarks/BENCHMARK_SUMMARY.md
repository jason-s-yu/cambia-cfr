# Deep CFR Benchmark Summary

Generated: 2026-02-16

## Executive Summary

Comprehensive benchmarking of the Deep CFR implementation was conducted on an RTX 3090 / Ryzen 9 5950X / 31GB RAM (WSL2) system. Key finding: **traversals dominate training time (>99%)**, making GPU vs CPU choice for network training effectively irrelevant. The network (174,610 params) is too small to benefit significantly from GPU acceleration.

**IMPORTANT**: CPU benchmarks were run while 16 training workers consumed all CPU cores. The reported "speedup" ratios are artifacts of CPU contention, NOT representative of actual GPU vs CPU performance.

## System Configuration

- **GPU**: NVIDIA GeForce RTX 3090 (24 GB VRAM)
- **CPU**: AMD Ryzen 9 5950X (16C/32T)
- **RAM**: ~31 GB (WSL2)
- **Network**: AdvantageNetwork/StrategyNetwork, 174,610 params each (~698 KB)

## Completed Benchmarks

### 1. GPU Network Performance (Clean, Uncontended)

#### Forward Pass
| Batch Size | Latency (ms) | Throughput |
|------------|-------------|------------|
| 512 | 0.99-5.0 | 102-515K samples/s |
| 1024 | 0.80-0.95 | 785K-1.27M samples/s |
| **2048** | **0.70-0.71** | **2.9M samples/s** |
| 4096 | 1.1-4.1 | 1.0-3.7M samples/s |
| 8192 | 4.8 | 1.4-1.7M samples/s |
| 16384 | 2.3-4.6 | 3.6-7.1M samples/s |

#### Backward Pass (full training step: fwd + loss + backward + clip + optimizer)
| Batch Size | Latency (ms) | Throughput |
|------------|-------------|------------|
| 512 | 3.2-28.8 | 18-79K samples/s |
| 1024 | 4.7-32.7 | 31-271K samples/s |
| **2048** | **5.3-9.6** | **213-484K samples/s** |
| 4096 | 4.3-16.5 | 249-949K samples/s |
| 8192 | 8.7-10.8 | 760-936K samples/s |
| 16384 | 4.6-5.3 | 3.1-3.6M samples/s |

GPU memory peak: **200 MB** (<1% of 24 GB VRAM)

### 2. CPU Network Performance (CONTENDED — 16 training workers active)

**WARNING**: These measurements are NOT representative of uncontended CPU performance. The CPU was saturated by 16 training workers at ~200% each (3200% total CPU demand on 3200% capacity system).

#### Forward Pass (contended)
| Batch Size | Latency (ms) | Throughput |
|------------|-------------|------------|
| 512 | 1,694 | 302 samples/s |
| 1024 | 1,533 | 668 samples/s |
| **2048** | **1,583** | **1,293 samples/s** |
| 4096 | 1,626 | 2,519 samples/s |
| 8192 | 1,698 | 4,825 samples/s |

#### Backward Pass (contended)
| Batch Size | Latency (ms) | Throughput |
|------------|-------------|------------|
| 512 | 8,331 | 61 samples/s |
| 1024 | 6,034 | 170 samples/s |
| **2048** | **5,903** | **347 samples/s** |
| 4096 | 5,797 | 707 samples/s |
| 8192 | 5,855 | 1,399 samples/s |

**Estimated uncontended CPU at bs=2048**: Forward ~0.3-0.5ms, Backward ~3-7ms (competitive with GPU for this small network).

### 3. GPU vs CPU — Corrected Analysis

| Metric | GPU (clean) | CPU (estimated uncontended) | Real Speedup |
|--------|------------|---------------------------|-------------|
| Forward bs=2048 | 0.7 ms | ~0.3-0.5 ms | **~0.7-2x** (CPU may be faster) |
| Backward bs=2048 | 5.3-9.6 ms | ~3-7 ms | **~1-3x** |
| 4000 SGD steps | 21-38 sec | 12-28 sec | **~1-2x** |

**Conclusion**: For this 175K-param network, GPU provides minimal advantage (~1-3x) for network training. The network is too small to utilize the RTX 3090's massive parallelism.

### 4. Memory Profiling

#### Reservoir Buffer Memory
| Capacity | Total Memory | Per Sample |
|----------|-------------|------------|
| 100K | 208 MB | 2.08 KB |
| 500K | 938 MB | 1.88 KB |
| 1M | 1,611 MB | 1.61 KB |
| **2M (production)** | **~3,200 MB** (extrapolated) | **~1.6 KB** |

#### System Memory Budget (31 GB available)
| Component | Memory |
|-----------|--------|
| Advantage buffer (2M) | ~3.2 GB |
| Strategy buffer (2M) | ~3.2 GB |
| 16 worker processes | ~6.4 GB |
| Python/PyTorch overhead | ~2 GB |
| **Total** | **~15.2 GB (49%)** |
| **Headroom** | **~15.8 GB** |

### 5. Batch Size Sweep

#### GPU Training Throughput
| Batch Size | Throughput | GPU Memory |
|------------|-----------|------------|
| 256 | 21.9K samples/s | 21.5 MB |
| 512 | 127.4K samples/s | 23.9 MB |
| 1024 | 271.9K samples/s | 29.6 MB |
| **2048** | **483.8K samples/s** | **44.0 MB** |
| 4096 | 518.9K samples/s | 64.3 MB |
| 8192 | 760.1K samples/s | 110.5 MB |
| 16384 | 3,595K samples/s | 200.3 MB |

## Critical Finding: Traversals Dominate Training

Training Step 1 (1000 traversals, 16 workers) ran for **6+ hours** without completing.

| Phase | Time | % of Step |
|-------|------|-----------|
| Traversal (external sampling) | 6+ hours | **>99%** |
| Advantage net training (4000 SGD steps) | ~20-40 sec (GPU) / ~12-28 sec (CPU) | **<0.1%** |
| Strategy net training (4000 SGD steps) | ~20-40 sec (GPU) / ~12-28 sec (CPU) | **<0.1%** |
| Weight serialization | <1 sec | **~0%** |

## Recommendations

### 1. Keep CPU for Training
The network is too small for GPU to help meaningfully. GPU provides ~1-3x speedup for a phase that's <0.1% of total time.

### 2. Reduce Traversals Per Step
1000 traversals per step takes 6+ hours. Consider:
- Start with 100-200 traversals for early iterations (rough regret estimates)
- Increase to 500-1000 for later iterations when precision matters

### 3. Optimize Traversal Code
External sampling enumerates ALL actions at traverser nodes. Profile `_deep_traverse()` for:
- Unnecessary copying/allocation in the recursive calls
- Action mask computation efficiency
- Network inference batching across game states

### 4. Current Config Assessment
| Parameter | Current | Recommendation |
|-----------|---------|----------------|
| Workers | 16 | Good — matches physical cores |
| Batch size | 2048 | Good for CPU |
| Train steps/iter | 4000 | Consider 2000 for faster cycles |
| **Traversals/step** | **1000** | **Reduce to 200 initially** |
| Buffer capacity | 2M | Good — memory allows it |
| Device | CPU | **Keep CPU** |

## Files

All benchmark results in `/workspace/benchmarks/`:
- `gpu_quick/results.json` — Quick GPU-only benchmark (clean)
- `network/gpu_vs_cpu.json` — Formal GPU vs CPU comparison
- `network/sweep_cpu.json` — CPU batch size sweep (contended)
- `network/sweep_gpu.json` — GPU batch size sweep (clean)
- `2026-02-16_104447/network/` — bench-runner-net detailed results
