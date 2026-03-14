# Nap (Neural Adaptive Predictor) CPUIdle Governor

A Linux kernel CPUIdle governor that uses a MLP-based neural network to learn the optimal idle state for each CPU online.

<div align="center"><img width="416" height="369" alt="nap" src="https://github.com/user-attachments/assets/a2861a8d-15cf-4b0e-9c09-6c86001ebbb7" /></div>

## Overview

Traditional CPUIdle governors (`ladder`, `menu`, `teo`) rely on fixed heuristics to predict how long a CPU will sleep and select an idle state accordingly. These heuristics are effective for common patterns but struggle with irregular or shifting workloads.

Nap (Neural Adaptive Predictor) replaces the heuristic with a two-expert MoE regression model that runs entirely in-kernel. Each CPU maintains two 16-16-1 MLPs — one specializing in short sleeps, the other in deep sleeps — and selects the appropriate expert based on the predicted sleep length. The networks learn online via deferred backpropagation with an asymmetric overshoot loss, converging the overshoot probability to a configurable target (default 5%). SIMD-accelerated forward and backward passes (SSE2 / AVX2+FMA / AVX-512) keep inference overhead negligible.

## How It Works

### Neural Network Architecture

Each expert is a 16-16-1 multi-layer perceptron:

| Layer | Size | Activation |
|---|---|---|
| Input | 16 features | - |
| Hidden | 16 neurons | ReLU |
| Output | 1 neuron | Linear |

Parameters per expert: 289 (16x16 + 16 + 16 + 1). Total parameters: 578 (2 experts), active parameters per inference: 289.

The output is a scalar in log2 space representing the predicted sleep duration in nanoseconds. Idle state selection is performed by comparing this value against precomputed log2 cost thresholds (target residency + exit latency) for each state, choosing the deepest state whose cost does not exceed the prediction.

### Mixture of Experts

Two experts specialize on different workload regimes:

- **Expert 0** — short sleep predictions (log2(sleep_length) < threshold)
- **Expert 1** — deep sleep predictions (log2(sleep_length) >= threshold)

The expert boundary is set to the midpoint between the shallowest and deepest C-state costs in log2 space. On each idle entry, feature\[0\] (log2 of the next timer event) is compared against this threshold to select the active expert. Only the selected expert runs the forward pass and receives weight updates.

### Feature Groups

The 16 input features are organized into four groups:

- **Time prediction** (4): log2 sleep length, log2 last residency, ring buffer average & standard deviation
- **Pattern analysis** (4): log history min/max, trend, short idle ratio
- **State feedback** (4): above-target ratio, intercept rate, prediction error, log2 busy time
- **External signals** (4): tick stopped flag, I/O wait count, latency requirement ratio, log2 IRQ rate

### Online Learning

After each idle exit, the governor compares the selected state against the post-hoc ideal state derived from actual residency. Every `learn_interval` reflects (default: 4), a deferred backpropagation step updates the active expert's weights.

The loss function is a direct overshoot loss with asymmetric learning rates:

- **Overshoot** (selected state too deep for actual residency): gradient pushes the output down with learning rate `base_lr * (1 - alpha)`
- **No overshoot**: gradient pushes the output up with learning rate `base_lr * alpha`

where `alpha` is the target overshoot percentile (default: 0.05). At equilibrium, P(overshoot) converges to `alpha`. Gradients are element-wise clipped to `[-max_grad_norm, +max_grad_norm]`.

When the network output is clamped at the upper bound (prediction equals sleep length), non-overshoot gradients are suppressed to prevent unbounded weight growth in always-idle systems.

### Weight Initialization

- Hidden layer: Xavier uniform (deterministic PRNG, seed = 42)
- Output layer: uniform [-0.01, 0.01]
- All biases: zero
- **Neuron 0 pass-through**: `w_h1[0][0] = 1.0`, `w_out[0] = 1.0`, all other inputs to neuron 0 zeroed

The pass-through initialization ensures the initial output approximates `log2(sleep_length)`, providing sensible state selection before any learning occurs.

### SIMD Dispatch

At governor enable time, Nap probes the CPU feature set and selects the fastest available implementation:

1. AVX-512F (non-Intel only; Intel throttles AVX-512 with a heavy license penalty)
2. AVX2 + FMA
3. SSE2 (baseline)

All FPU/SIMD code is compiled into separate translation units and wrapped in `kernel_fpu_begin()`/`kernel_fpu_end()` to prevent corruption of userspace FPU state.

## Tunables

Exposed under `/sys/devices/system/cpu/nap/`:

| Tunable | Default | Description |
|---|---|---|
| `version` | *(read-only)* | Governor version |
| `simd` | *(read-only)* | Detected SIMD capability (`sse2` / `avx2` / `avx512`) |
| `stats` | *(read-only)* | Total selects, residency, overshoot count/rate, learn count |
| `learning_rate` | `1` | Learning rate in thousandths (1 = 0.001) |
| `learn_interval` | `4` | Backpropagation frequency (every N reflects) |
| `overshoot_pctl` | `50` | Target overshoot percentile in thousandths (50 = 5%) |
| `reset_weights` | *(write-only)* | Trigger weight reinitialization (`all` or cpulist e.g. `0-3,5,7`) |
| `reset_stats` | *(write-only)* | Reset statistics counters |

## Installation

Nap is delivered as a kernel patch. Apply it to the Linux 6.18.3 source tree and enable `CONFIG_CPU_IDLE_GOV_NAP=y`:

```sh
cd /path/to/linux
patch -p1 < /path/to/nap/patches/0001-6.18.3-nap-v0.2.1.patch
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
