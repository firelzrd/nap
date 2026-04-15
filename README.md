# Nap (Neural Adaptive Predictor) CPUIdle Governor

A Linux kernel CPUIdle governor that uses a MLP-based neural network to learn the optimal idle state for each CPU online.

<div align="center"><img width="416" height="369" alt="nap" src="https://github.com/user-attachments/assets/a2861a8d-15cf-4b0e-9c09-6c86001ebbb7" /></div>

## Overview

Traditional CPUIdle governors (`ladder`, `menu`, `teo`) rely on fixed heuristics to predict how long a CPU will sleep and select an idle state accordingly. These heuristics are effective for common patterns but struggle with irregular or shifting workloads.

Nap (Neural Adaptive Predictor) replaces the heuristic with a three-expert MoE regression model that runs entirely in-kernel. Each CPU maintains three 8-8-1 MLPs — one specializing in short (tick-bound) sleeps, one in intermediate (nohz) sleeps, and one in deep (deepest C-state) sleeps — and selects the appropriate expert based on the predicted sleep length. The networks learn online via deferred backpropagation with an asymmetric overshoot loss, converging the overshoot probability to a configurable target (default 8%). SIMD-accelerated forward and backward passes (SSE2 / AVX2+FMA) keep inference overhead negligible. A POLL short-circuit fast path bypasses NN inference entirely when the predicted sleep is too short for any C-state.

## How It Works

### Neural Network Architecture

Each expert is an 8-8-1 multi-layer perceptron:

| Layer | Size | Activation |
|---|---|---|
| Input | 8 features | - |
| Hidden | 8 neurons | ReLU |
| Output | 1 neuron | Linear |

Parameters per expert: 81 (8×8 + 8 + 8 + 1). Total parameters: 243 (3 experts), active parameters per inference: 81.

The output is a scalar in log2 space representing the predicted sleep duration in nanoseconds. Idle state selection is performed by comparing this value against precomputed log2 cost thresholds (`target_residency_ns` only; exit latency is a wakeup cost, not a factor in residency profitability) for each state, choosing the deepest state whose cost does not exceed the prediction.

### POLL Short-Circuit Fast Path

Before invoking the NN, `nap_select()` checks whether the predicted sleep length is shorter than the shallowest valid C-state's target residency. If so, POLL is returned immediately — no feature extraction, no inference, no history update. This eliminates NN overhead for very short idles.

- The shallowest valid C-state is cached per-CPU and invalidated when the PM QoS latency request changes or after `NAP_MIN_STATE_REFRESH_JIFFIES` (1 second).
- `poll_limit_ns` is set to `sleep_length + 1 µs` margin, clamped between 1 µs and the shallowest C-state's target residency.
- `nap_reflect()` skips history, learning, and all NN-related bookkeeping for short-circuited events, updating only the aggregate residency statistic. This prevents noisy POLL-duration samples from contaminating the NN's training distribution.

### Mixture of Experts

Three experts specialize on different workload regimes:

- **Expert 0 (short)** — tick-bound idles (log2(sleep_length) < `expert_mid`)
- **Expert 1 (long)** — nohz intermediate idles (`expert_mid` ≤ log2(sleep_length) < `expert_deep`)
- **Expert 2 (deep)** — deepest C-state idles (log2(sleep_length) ≥ `expert_deep`)

Two boundaries partition the sleep-length space:

1. **`expert_mid`** (short ↔ long) — tied to the tick period (`TICK_NSEC`): the first C-state whose target residency exceeds one jiffy marks the start of the "long" regime. This separates tick-bound idles (where measured residency is dominated by the next tick, producing noisy gradients) from nohz idles (where residency reflects the workload's true idle duration). If all states exceed one jiffy, the boundary is placed just below C1 so the short expert remains routable but unused.
2. **`expert_deep`** (long ↔ deep) — placed at the midpoint between the second-deepest and deepest C-state's log2(target_residency). The deepest C-state often has qualitatively different residency characteristics (package C-state, longer exit latency, power-gated domains) that warrant a dedicated expert to avoid gradient interference with intermediate states. On hardware with only 2 C-states, `expert_deep` collapses to `expert_mid`, effectively reducing to a 2-expert regime.

On each idle entry, feature\[0\] (log2 of the next timer event) is compared against both thresholds to select the active expert. Only the selected expert runs the forward pass and receives weight updates.

### Input Features

The 8 input features are selected via gradient-based importance analysis, retaining only those with significant contribution to selection quality:

| # | Feature | Description |
|---|---------|-------------|
| 0 | log2(sleep_length) | Next timer event (primary signal) |
| 1 | log2(last_residency) | Actual duration of last idle |
| 2 | log_hist avg | Average of recent idle durations (log2 ring buffer) |
| 3 | log_hist min | Shortest recent idle (overshoot guard) |
| 4 | log_hist max | Longest recent idle (deep state signal) |
| 5 | signed log2(\|pred_error\|+1) | Prediction feedback (sign-preserving) |
| 6 | log2(busy_ns) | Pre-idle busy duration |
| 7 | log2(lat_req) − log2(deepest_lat) | PM QoS headroom in log2 space |

### Online Learning

After each idle exit, the governor compares the selected state against the post-hoc ideal state derived from actual residency. Learning is governed by a dual gate: it fires only when both the reflect counter reaches `learn_interval` (default: 4) **and** at least `learn_jiffies_min` jiffies (default: 1) have elapsed since the last learning step. The time gate prevents sustained weight churn on workloads with very rapid idle bursts; setting it to 0 restores the original counter-only behavior.

The loss function is a direct overshoot loss with asymmetric learning rates:

- **Overshoot** (selected state too deep for actual residency): gradient pushes the output down with learning rate `base_lr * (1 - alpha)`
- **No overshoot**: gradient pushes the output up with learning rate `base_lr * alpha`

where `alpha` is the target overshoot percentile (default: 0.10). At equilibrium, P(overshoot) converges to `alpha`. Gradients are element-wise clipped to `[-max_grad_norm, +max_grad_norm]`.

When the network output is clamped at the upper bound (prediction equals sleep length), non-overshoot gradients are suppressed to prevent unbounded weight growth in always-idle systems.

### Weight Initialization

- Hidden layer: Xavier uniform (deterministic PRNG, seed = 42)
- Output layer: uniform [-0.01, 0.01]
- All biases: zero
- **Neuron 0 pass-through**: `w_h1[0][0] = 1.0`, `w_out[0] = 1.0`, all other inputs to neuron 0 zeroed

The pass-through initialization ensures the initial output approximates `log2(sleep_length)`, providing sensible state selection before any learning occurs.

### SIMD Dispatch

At governor enable time, Nap probes the CPU feature set and selects the fastest available implementation:

1. AVX2 + FMA (8 hidden neurons = 1 ymm register)
2. SSE2 (baseline; 8 hidden neurons = 2 xmm registers)

All FPU/SIMD code is compiled into separate translation units and wrapped in `kernel_fpu_begin()`/`kernel_fpu_end()` to prevent corruption of userspace FPU state.

## Tunables

Exposed under `/sys/devices/system/cpu/nap/`:

| Tunable | Default | Description |
|---|---|---|
| `version` | *(read-only)* | Governor version |
| `simd` | *(read-only)* | Detected SIMD capability (`sse2` / `avx2`) |
| `stats` | *(read-only)* | Total selects, residency, overshoot count/rate, learn count |
| `learning_rate` | `1` | Learning rate in thousandths (1 = 0.001) |
| `learn_interval` | `4` | Backpropagation frequency (every N reflects) |
| `learn_jiffies_min` | `1` | Minimum jiffies between learning steps (0 = disabled) |
| `overshoot_pctl` | `100` | Target overshoot percentile in thousandths (100 = 10%) |
| `reset_weights` | *(write-only)* | Trigger weight reinitialization (`all` or cpulist e.g. `0-3,5,7`) |
| `reset_stats` | *(write-only)* | Reset statistics counters |

## Benchmark: Overshoot Rate

Overshoot rate measures how often a governor selects a C-state deeper than the actual residency justifies. Lower is better.

| Governor | Overshoot Rate |
|----------|---------------|
| **nap**  | 4.2%         |
| teo      | 19.70%     |
| menu     | 47.27%       |

Measured with [this patch](https://github.com/firelzrd/nap/blob/main/tests/0002-menu-teo-overshoot-rate-sysfs.patch) applied on moderately idle desktop (10-second sample per governor, Linux 6.18, AMD Zen):

```sh
for gov in menu teo nap; do
  echo $gov | sudo tee /sys/devices/system/cpu/cpuidle/current_governor
  sleep 10
  grep -R . /sys/devices/system/cpu/$gov/stats
done
```

## Installation

Nap is delivered as a kernel patch. Apply it to the Linux 6.18.3 source tree and enable `CONFIG_CPU_IDLE_GOV_NAP=y`:

```sh
cd /path/to/linux
patch -p1 < /path/to/nap/patches/stable/0001-6.18.3-nap-v0.4.0.patch
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
