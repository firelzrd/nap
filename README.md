# Nap (Neural Adaptive Predictor) CPUIdle Governor

A Linux kernel CPUIdle governor that learns the optimal idle state for each CPU online with a small in-kernel neural network.

<div align="center"><img width="416" height="369" alt="nap" src="https://github.com/user-attachments/assets/a2861a8d-15cf-4b0e-9c09-6c86001ebbb7" /></div>

## Overview

Traditional CPUIdle governors (`ladder`, `menu`, `teo`) rely on fixed heuristics to predict how long a CPU will sleep and select an idle state accordingly. These heuristics are effective for common patterns but struggle with irregular or shifting workloads.

Nap (Neural Adaptive Predictor) replaces the heuristic with a single small MLP trunk feeding an **ordinal survival head**, running entirely in-kernel per CPU. Instead of regressing one point estimate of the sleep length, the network predicts, for each idle-state boundary, the probability that the upcoming idle lasts at least that state's `target_residency`. The decision layer blends this prediction with a per-CPU decayed histogram of recently observed idle depths (Beta-Binomial shrinkage) and picks the deepest feasible state whose calibrated survival probability still clears a configurable confidence level. The network learns online via deferred backpropagation; SIMD-accelerated forward and backward passes (SSE2 / AVX2+FMA) keep inference overhead negligible, and a POLL short-circuit fast path bypasses the NN entirely when the predicted sleep is too short for any C-state.

There is **no mixture of experts** — a single network handles every sleep-length regime. Multimodality and workload shifts are absorbed by the decayed-histogram floor in the decision layer, which also makes online learning robust against catastrophic forgetting.

## How It Works

### Neural Network Architecture

Each CPU owns a single 8-8-1 multi-layer perceptron trunk:

| Layer | Size | Activation |
|---|---|---|
| Input | 8 features | - |
| Hidden | 8 neurons | ReLU |
| Score | 1 scalar `s` | Linear |

Trunk parameters: 81 (8×8 + 8 + 8 + 1), plus one ordered threshold per idle-state boundary. The scalar score `s` is approximately the log2 of the predicted idle duration in nanoseconds; it is consumed by the ordinal survival head below rather than used directly as a sleep-length estimate.

### Ordinal Survival Head

For each idle-state boundary *k*, the network's survival probability is a proportional-odds model over the shared score:

```
q_nn_k = sigmoid(s − thr_ord[k−1])
```

with one ordered threshold `thr_ord[k−1]` per boundary, seeded at `log2(target_residency[k])`. This represents the idle-duration distribution at exactly the points the decision needs (the sufficient statistic), rather than a single point estimate.

### Decision Layer

The decision shrinks the NN prior toward observed data — a per-CPU decayed histogram `bin_count[]` of how often recent idles reached each depth — via Beta-Binomial shrinkage:

```
q_k = (K · q_nn_k + count(idle ≥ k)) / (K + total),   K = 16 pseudo-observations
```

The NN drives cold start; the floor takes over as samples accumulate. This shrinkage is what replaces a mixture of experts: it carries any multimodality of the idle distribution, and it keeps decisions stable even if the NN drifts. A running minimum enforces a monotone non-increasing survival curve, and the next timer event caps the reachable depth (`q_k = 0` for any state whose target residency lies past the next timer — it cannot be earned).

Nap then selects the **deepest feasible** state — enabled, and with `exit_latency_ns ≤` the PM QoS latency request — whose survival `q_k` still meets the confidence level (default 0.5).

### POLL Short-Circuit Fast Path

Before invoking the NN, `nap_select()` checks whether the predicted sleep length is shorter than the shallowest valid C-state's target residency. If so, POLL is returned immediately — no feature extraction, no inference, no history update. This eliminates NN overhead for very short idles.

- The shallowest valid C-state is cached per-CPU and invalidated when the PM QoS latency request changes or after `NAP_MIN_STATE_REFRESH_JIFFIES` (1 second).
- `poll_limit_ns` is set to `sleep_length + 1 µs` margin, clamped between 1 µs and the shallowest C-state's target residency.
- `nap_reflect()` skips history, learning, the floor sample, and all NN-related bookkeeping for short-circuited events, updating only the aggregate residency statistic. This prevents noisy POLL-duration samples from contaminating the NN's training distribution.

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

The decayed-histogram floor updates on **every** (non-short-circuited) idle, keeping the data term current. The trunk and ordinal thresholds update on a throttled schedule — a dual gate fires only when the reflect counter reaches `learn_interval` (default 4) **and** at least one jiffy has elapsed since the last step. The time gate caps weight churn on workloads with very rapid idle bursts; it has no effect on steady workloads.

The loss is the proportional-odds (ordinal) log-likelihood. Because all boundaries share the score `s`, the gradient with respect to `s` collapses to a single scalar:

```
g = Σ_k (q_k − y_k),   y_k = 1 if measured_residency ≥ target_residency[k] else 0
```

which drives the SIMD trunk backpropagation; each ordered threshold additionally moves by its own `(q_k − y_k)`. The loss is **symmetric** — all responsiveness lives in the decision-layer confidence dial, not in an asymmetric learning signal. Gradients are element-wise clipped to `[−max_grad_norm, +max_grad_norm]`.

### Weight Initialization

- Hidden layer: Xavier uniform (deterministic PRNG)
- Output (score) weights: near-zero, for ~0 initial learned contribution
- All biases: zero
- **Neuron 0 pass-through**: `w_h1[0][0] = 1.0`, `w_out[0] = 1.0`, all other inputs to neuron 0 zeroed

The pass-through initialization makes the initial score `s ≈ log2(sleep_length)`, and the thresholds are seeded at each boundary's `log2(target_residency)`, so before any learning the decision reproduces the classic "deepest state that fits the predicted sleep" behavior.

### SIMD Dispatch

At governor enable time, Nap probes the CPU feature set and selects the fastest available implementation:

1. AVX2 + FMA (8 hidden neurons = 1 ymm register)
2. SSE2 (baseline; 8 hidden neurons = 2 xmm registers)

AVX-512 is intentionally not used: with an 8-wide hidden layer a zmm register would run half-empty for the same instruction count, and on Intel parts the AVX-512 frequency license would hurt wakeup latency. All FPU/SIMD code is compiled into separate translation units and wrapped in `kernel_fpu_begin()`/`kernel_fpu_end()` to prevent corruption of userspace FPU state.

## Tunables

Exposed under `/sys/devices/system/cpu/nap/`:

| Tunable | Default | Description |
|---|---|---|
| `version` | *(read-only)* | Governor version |
| `simd` | *(read-only)* | Detected SIMD capability (`sse2` / `avx2`) |
| `stats` | *(read-only)* | Total selects, residency, overshoot count/rate, learn count |
| `learning_rate` | `1` | Learning rate in thousandths (1 = 0.001) |
| `learn_interval` | `4` | Backpropagation frequency (every N reflects) |
| `confidence` | `500` | Decision confidence in thousandths (500 = 0.5; range 1–999). A state is chosen only if its survival probability clears this. **Higher** = more conservative (shallower, lower wakeup-latency risk); **lower** = more aggressive (deeper, more energy savings). |
| `reset_weights` | *(write-only)* | Trigger weight reinitialization (`all` or cpulist, e.g. `0-3,5,7`) |
| `reset_stats` | *(write-only)* | Reset statistics counters |

## Benchmark: selection quality vs `teo`

Selection quality is measured with the governor-agnostic cpuidle `above`/`below`/`usage` counters (the kernel's own per-state miss accounting), so it is an oracle metric, not a governor self-report:

- **miss%** = (above + below) / usage — overall mis-selection (lower is better)
- **above%** = overshoot (too deep; a wakeup-latency cost)
- **below%** = undershoot (too shallow; an energy cost)

`teo` is **co-measured in the same boot** as a drift anchor — a single governor's absolute miss% drifts between boots with thermal/frequency state, so the meaningful signal is the `nap − teo` margin. Harness: `tests/` (16-CPU x86_64, AVX2, 5 reps, 3 s warmup + 10 s window).

### Synthetic sweep — miss% across fixed idle durations

| idle duration | **nap** | teo | nap advantage |
|---|---|---|---|
| 10 µs | **0.96** | 1.98 | 2.1× |
| 50 µs | **0.44** | 6.00 | **13.6×** |
| 200 µs | **0.99** | 7.97 | 8.1× |
| 1 ms | **4.87** | 18.11 | 3.7× (and lower power: 7.26 W vs 8.13 W) |
| 5 ms | **13.36** | 15.14 | 1.1× |

### Real workloads — miss%

| workload | **nap** | teo |
|---|---|---|
| pingpong (cross-CPU non-timer wakeups, intercept regime) | **3.44** | 7.02 |
| fio (bursty IO with think-time, iowait regime) | **1.22** | 1.88 |
| idle (desktop-idle mix) | **25.50** | 26.63 (nap also lower power: 6.77 W vs 6.86 W) |

nap selects more accurately than `teo` at every operating point measured — 3–13× fewer mis-selections on the synthetic sweep and ~2× on the intercept-heavy pingpong workload — at equal or lower package power. The reproduction harness, raw CSVs, and full methodology (including the boot-to-boot drift analysis) live in `tests/` and `v0.5.0-port-validation.md`.

## Installation

Nap is delivered as a kernel patch. Apply the patch for your kernel version from the `patches/` directory and enable `CONFIG_CPU_IDLE_GOV_NAP=y`:

```sh
cd /path/to/linux
patch -p1 < /path/to/nap/patches/stable/<kernel-version>-nap-v0.5.0.patch
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
