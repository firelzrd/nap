# Nap (Neural Adaptive Predictor) CPUIdle Governor

A Linux kernel CPUIdle governor that uses a small neural network to learn the optimal idle state for each CPU online.

## Overview

Traditional CPUIdle governors (`ladder`, `menu`, `teo`) rely on fixed heuristics to predict how long a CPU will sleep and select an idle state accordingly. These heuristics are effective for common patterns but struggle with irregular or shifting workloads.

Nap (Neural Adaptive Predictor) replaces the heuristic with a 16-16-10 multi-layer perceptron (MLP) that runs entirely in-kernel. Each CPU maintains its own network, learning online via backpropagation to select the idle state that best balances power savings against wakeup latency. SIMD-accelerated forward and backward passes (SSE2 / AVX2+FMA / AVX-512) keep inference overhead negligible.

## How It Works

### Neural Network Architecture

| Layer | Size | Activation |
|---|---|---|
| Input | 16 features | - |
| Hidden | 16 neurons | ReLU |
| Output | 10 neurons | Linear (argmax) |

Total parameters: 442 (1,768 bytes per CPU).

### Feature Groups

The 16 input features are organized into four groups:

- **Time prediction** (4): log2 sleep length, log2 last residency, ring buffer average & standard deviation
- **Pattern analysis** (4): log history min/max, trend, short idle ratio
- **State feedback** (4): above-target ratio, intercept rate, prediction error, log2 busy time
- **External signals** (4): tick stopped flag, I/O wait count, latency requirement ratio, log2 IRQ rate

### Online Learning

After each idle exit, the governor compares the selected state against the post-hoc ideal state derived from actual residency. Every `learn_interval` reflects (default: 16), a deferred backpropagation step updates the network weights using SGD with gradient clipping.

### Weight Initialization

- Hidden layer: Xavier uniform (deterministic PRNG, seed = 42)
- Output biases: seeded from idle-state exit latencies (`-0.1 * log2(exit_latency_ns)`)

This hardware-aware initialization provides sensible defaults before any learning occurs.

### Convergence

An exponential moving average (EMA) of prediction accuracy is tracked. After `warmup_threshold` learning steps (default: 64), the governor checks whether accuracy exceeds `convergence_thresh` (default: 75%). Until convergence, a latency-aware fallback heuristic supplements the network output.

### SIMD Dispatch

At governor enable time, Nap probes the CPU feature set and selects the fastest available implementation:

1. AVX-512F
2. AVX2 + FMA
3. SSE2 (baseline)

All FPU/SIMD code is compiled into separate translation units and wrapped in `kernel_fpu_begin()`/`kernel_fpu_end()` to prevent corruption of userspace FPU state.

## Tunables

Exposed under `/sys/devices/system/cpu/cpuidle/nap/`:

| Tunable | Default | Description |
|---|---|---|
| `version` | *(read-only)* | Governor version (0.1.0) |
| `simd` | *(read-only)* | Detected SIMD capability (`sse2` / `avx2` / `avx512`) |
| `converged` | *(read-only)* | Whether the network has converged |
| `stats` | *(read-only)* | Total selects, residency, undershoot rate, learn count |
| `ema_accuracy` | *(read-only)* | Exponential moving average of prediction accuracy |
| `learning_mode` | `1` | Enable (`1`) or disable (`0`) online learning |
| `learning_rate_millths` | `1` | Learning rate in thousandths (0.001) |
| `learn_interval` | `16` | Backpropagation frequency (every N reflects) |
| `max_grad_norm_millths` | `1000` | Gradient clipping threshold in thousandths (1.0) |
| `warmup_threshold` | `64` | Minimum learning steps before convergence check |
| `convergence_thresh` | `768` | Accuracy threshold (x1024 scale, 768 = 75%) |
| `reset_weights` | *(write-only)* | Trigger weight reinitialization |
| `reset_stats` | *(write-only)* | Reset statistics counters |

## Installation

Nap is delivered as a kernel patch. Apply it to the Linux 6.18.3 source tree and enable `CONFIG_CPU_IDLE_GOV_NAP=y`:

```sh
cd /path/to/linux
patch -p1 < /path/to/nap/patches/0001-6.18.3-nap-v0.1.0.patch
```

### Activate

Add the boot parameter:

```
cpuidle.governor=nap
```

Or switch at runtime:

```sh
echo nap | sudo tee /sys/devices/system/cpu/cpuidle/current_governor
```
