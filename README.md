# Nap (Neural Adaptive Predictor) CPUIdle Governor

A Linux kernel CPUIdle governor that predicts the distribution of the upcoming idle time with a small in-kernel neural network, learns online on every CPU, and picks the idle state that saves the most energy for a **budget of avoidable wakeup latency** you set.

<div align="center"><img width="416" height="369" alt="nap" src="https://github.com/user-attachments/assets/a2861a8d-15cf-4b0e-9c09-6c86001ebbb7" /></div>

## Why Nap

`ladder`, `menu` and `teo` choose an idle state with fixed heuristics: they estimate how long the CPU will sleep and go as deep as that allows. None of them has an explicit objective for what a wrong choice costs the user. Going too deep costs wakeup latency; going too shallow costs energy. The only dial is the PM QoS limit, which is all or nothing.

Nap makes the trade explicit:

- **It optimises what the user feels.** The cost of going deeper than the idle deserved is the *avoidable wakeup latency*: the exit latency of the state entered minus that of the state that fits the idle. Nap chooses, for every idle, the state (and whether to stop the tick) that minimises `energy + λ · avoidable latency`.
- **You set a budget, not a heuristic.** `budget_ppm` is the share of the wall-clock time a CPU may lose to avoidable wakeup latency. A per-CPU controller adapts λ to keep within it. A CPU that wakes rarely may take deep states with long exit latencies; one that wakes thousands of times a second is held to short ones.
- **It predicts a distribution, not a number.** The network outputs `P(idle ≥ t)` on a fixed time grid (0.5 µs … 16.8 ms, one point per octave), so nothing in it depends on the idle-state table. Any machine's `target_residency` / `exit_latency` are read off the grid at run time.
- **It learns, and what it learns can be kept.** Weights adapt online on every CPU. They can be dumped, loaded, averaged and restored across reboots (see [Keeping what it learned](#keeping-what-it-learned)).

### How it compares

Measured on one machine (Ryzen 7 7840HS, 16 threads, idle states POLL / C1 / C2 / C3), on an idle desktop and six synthetic loads: *media* (60 Hz frame + audio timers), *game*, *interactive*, *build* (many short compiler processes), *I/O*, *mixed*. 12 s per cell, 2 rounds. `teo` and `menu` are recorded with a transparent wrapper governor and scored with the same definition of avoidable latency on every idle; Nap v0.6.0 (measured as a module build of the same source) reports the same quantity itself.

**Avoidable wakeup latency per idle (µs), and the share of CPU time lost to it (ppm)**

| load | teo | menu | **Nap v0.6.0** (budget 4000 ppm) |
|---|---|---|---|
| idle | 73 / 18 068 | 75 / 12 597 | **10.8 / 3 196** |
| media | 78 / 14 286 | 79 / 14 716 | **10.6 / 3 378** |
| game | 100 / 20 102 | 100 / 18 646 | **10.9 / 3 510** |
| interactive | 82 / 13 548 | 78 / 12 203 | **11.2 / 3 175** |
| build | 25 / 33 416 | 30 / 32 758 | **3.6 / 4 776** |
| I/O | 94 / 31 199 | 110 / 41 297 | **9.9 / 4 941** |
| mixed | 105 / 45 998 | 125 / 50 289 | **10.3 / 5 137** |

`teo` and `menu` lose 1.2–5.0 % of the CPU time to avoidable wakeup latency; Nap v0.6.0 loses 0.3–0.5 %, 4 to 9 times less (and 7 to 10 times less per idle). `teo` and `menu` are indistinguishable from each other here.

> **v0.6.5.** These tables were measured with v0.6.0, whose network had 128 hidden units and 16 features. v0.6.5 uses 32 hidden units and 11 features (see [Network](#network) and [Cost](#cost)), which makes a prediction cheaper but a little less accurate: on held-out recordings the shortfall at a fixed overshoot rate is 16 % higher than with the v0.6.0 network. After online adaptation to a workload of another kind (pretrained on one recording, streamed on another) the difference was 0.2–0.8 points of misclassified idles. The latency and power comparisons above have not been repeated with v0.6.5.

**Measured wakeup latency** (cyclictest, mean over threads, µs; ranges over the measurement sessions)

| | teo | menu | **Nap v0.6.0** |
|---|---|---|---|
| 1 ms timer, with background load | 0–1 | 1–4 | **1** |
| 1.5 ms timer next to it | 8–29 | 8–17 | **4–14** |
| 4 threads, 500 µs | 1–31 | 1–29 | 1–35 (no difference) |

**Power** is within the measurement noise of `teo` (about ±2 %, interleaved runs): for example 9.4–9.6 W (Nap v0.6.0) against 9.5–10.1 W (`teo`) on the 4-thread cyclictest, 13.5 W against 13.2–13.3 W with the 1 ms timer plus load.

**What this does not show.**

- It is one machine. Nothing here says how Nap behaves on Intel parts with ten C-states or on ARM. The network is state-table independent by construction, but the pretrained prior and the default power ladder come from this machine.
- The synthetic loads were generated for this study, and the pretrained weights were trained on recordings of the same kinds of loads. Treat the gains as an upper bound for real use.
- Nap does not make fewer wrong choices on an idle desktop: an earlier build with a different cost function matched `teo`'s `above`/`below` fractions (about 18 % too deep, 9 % too shallow). v0.6.0 deliberately spends more time in states shallower than ideal (hundreds of ms per CPU-second against tens for `teo`) and enters more idles; that is the price of the lower latency, and it showed no measurable power cost here.
- Each cell is 12 s and two rounds; power differences below about 0.5 W are noise on this machine.
- On this CPU C2 and C3 draw about the same power at the measured wake rates: with C3 disabled the idle package power was not higher (8.6 W against 9.2 W, three rounds under `teo`). Deep states buy little here, which is part of why a latency-first budget costs no measurable power. On CPUs where deeper states save real power the trade will look different, and it is untested.

## How It Works

```
 raw signals ──► 11 features ──► network ──► P(idle ≥ t) on a time grid ──┐
 (timer, history, busy period,     (online-trained)                        │
  wake sources, tick)                                                      ▼
                    budget_ppm ──► per-CPU controller ──► λ ──► choose (state, stop tick)
                                                                  minimising E + λ·L
```

### Features (11)

| # | feature | kind |
|---|---|---|
| 0 | log2(time to next timer) | time |
| 1–5 | log2(last five idle lengths) | time |
| 6 | log2(length of the last busy period) | time |
| 7 | log2(next timer / next tick) | dense |
| 8 | busy CPUs | dense |
| 9–10 | wake sources of the last wakeup: device IRQ, local timer | dense |

The local-timer wake source is x86-only and meant to be fed as its training mean on other architectures (the weights were trained with it dropped out). Nap builds for x86-64 only today; this is the seam for a port.

v0.6.1 and earlier also used the sixth idle length, the number of interrupts during the busy period, call-function IPIs as a wake source, and the number of wakeups on all CPUs in the last 50 µs and 1 ms. They were left out in v0.6.5: the first three change the held-out shortfall by less than 0.5 %, and the system-wide wakeup counts (3 % together) needed a shared 128-entry timestamp ring that every CPU wrote and every select scanned, which cost 230–300 ns per idle on 8–16 CPUs. Only the number of idle CPUs is still shared (one counter).

### Network

The seven **time features** enter through a piecewise-linear embedding: 16 knots one octave apart (2^9 … 2^24 ns), the two knots around the value are mixed with linear weights. This lets the first layer represent sharp thresholds in time, which a plain linear layer cannot. The four dense features are standardised.

| Part | Shape | Parameters |
|---|---|---|
| Time-feature embedding tables | 7 features × 16 knots × 32 hidden | 3 584 |
| Dense-feature rows | 4 features × 32 hidden | 128 |
| Hidden bias | 32 (ReLU) | 32 |
| Output weights | 32 × 16 (sigmoid): `P(idle ≥ 2^(9+g) ns)` | 512 |
| Output bias | 16 | 16 |
| **Total** | | **4 272** |

That is 17 KB of float32 weights per CPU (each CPU learns its own copy), 18 KB with its state: it fits in an L1 data cache. An idle touches only 18 of the 116 first-layer rows, so a forward pass is about 1 100 multiply-adds and a learning step about 1 600. Forward pass ≈ 50 ns, learning step ≈ 80 ns (boot self-test, AVX2; 38 and 70 ns with AVX-512).

The open-ended last bin uses the training-data mean (65.5 ms) as its representative time. Idles are labelled by the grid thresholds they reach; labels the tick makes unknowable are masked.

### Decision

The output gives a distribution over the idle length. For each candidate (state, stop the tick or not) Nap integrates over it:

- **E**, the expected energy: relative per-state powers (`power_milli`, default a generic ladder), break-even from `target_residency`, and the cost of stopping and restarting the tick.
- **L**, the expected avoidable wakeup latency: for every idle length, `exit_latency` of the candidate minus `exit_latency` of the deepest state whose `target_residency` fits it (never below zero). With the tick kept, the last piece of the idle is judged.

It picks the candidate with the lowest `E + λ·L`. The tick is never stopped when the next timer is closer than one tick: nothing can be saved, and the slow tick restart delays the wakeup. PM QoS limits and disabled states are honoured as before. When no C-state can pay off within the predicted sleep, POLL is returned without running the network.

### Budget controller

Per CPU, the share of the time lost to avoidable latency (an EWMA of the latency over an EWMA of the cycle time, busy period plus idle) is compared with `budget_ppm`: above 105 % of it λ rises by 2 %, below it λ falls by 2 %.

### Learning

One SGD step per idle on the masked cross-entropy of the grid outputs (learning rate 0.02). The weights start from a pretrained prior (trained offline on ten recordings of different loads from one machine and bundled with the patch); `reset_weights` returns to it. After about 8 s (2 000 idles of a CPU) a good part of the mismatch between the prior and the actual workload is gone; it takes minutes to settle.

### SIMD

Scalar, SSE2, AVX2+FMA and AVX-512 versions share one template and run on the same data layout; `max_simd=scalar|sse2|avx2|avx512` caps the level. At boot each level is checked against the offline model and against the scalar reference, and a level that disagrees is never used. All floating-point code is compiled separately and wrapped in `kernel_fpu_begin()` / `kernel_fpu_end()`. Building needs GCC 12 or later, or clang.

### Cost

Measured on a Ryzen 7 7840HS, AVX2, with the governor's sources compiled for user space and run over recorded idles: one select that takes the network path (features, network, decision, and the learning step of the previous idle) takes about 0.32 µs, against about 1.0 µs for v0.6.1 (128 hidden units, 16 features, wake ring). Of the 0.32 µs, forward plus learning are 0.13 µs; the rest is the features, the sigmoids and the decision, which folds the bin probabilities into a few sums once so that every (state, tick) candidate costs O(states). Idles shorter than the shallowest sleep state's target residency skip all of it and return POLL. The only state shared between CPUs is the counter of idle CPUs: 12–45 ns per idle for 1–16 CPUs, against 120–300 ns for the former counters, ring and scan. Each select disturbs about 5 KB of cache lines instead of 22 KB.

## Tunables

Under `/sys/devices/system/cpu/nap/`:

| File | Default | Description |
|---|---|---|
| `version`, `simd`, `stats` | read-only | Version, active SIMD level, counters: selects, overshoot, `avoidable_wake_latency_ns` (also per select), learn count, stop-tick count, entries per state |
| `budget_ppm` | `4000` | Share of the time (ppm) a CPU may lose to avoidable wakeup latency. Lower = more responsive, more power |
| `max_simd` | highest supported | `scalar` / `sse2` / `avx2` / `avx512` |
| `learning_rate` | `20` | In thousandths (20 = 0.02) |
| `learn` | `1` | `0` freezes the weights |
| `power_milli` | `1000,500,200,50,30,20,14,10,7,5` | Relative power per state, shallowest first. Machine dependent |
| `tick_stop_cost_ns`, `tick_wake_cost_ns` | `3000`, `5000` | Energy cost of stopping the tick and of a tick wakeup |
| `weights`, `weights_cpu` | root only | Dump / load the weights of a CPU (or of all) |
| `reset_weights` | write-only | Back to the built-in model (`all` or a cpulist) |
| `reset_stats` | write-only | Clear the counters |

## Keeping what it learned

What the network has learned survives re-enabling a CPU (hotplug, suspend, governor switch) and can be saved across reboots:

- `/sys/devices/system/cpu/nap/weights` dumps (read) or loads (write) the weights of the CPU chosen in `weights_cpu`. The kernel rejects a blob with a wrong header, checksum or shape (dumps made by v0.6.1 and earlier have another shape: 16 features and 128 hidden units), or with a NaN, an infinity or a weight of 1024 or more.
- `tools/napctl` exports, imports, inspects and averages them and keeps them in `/var/lib/nap`; `nap-state.service` restores them at boot and saves them at shutdown, and a timer saves every 30 minutes. A state file from another machine (CPU model or idle-state table), or made by a nap with another network shape, is ignored with a message unless forced; the built-in model stays.
- `tools/napviz.html` is a single-file viewer: weight histograms, the embedding curves, predictions and decisions for conditions you set, and a side-by-side comparison of two dumps. Open it in a browser and drop a dump on it.

## Installation

Nap is delivered as a kernel patch. Apply the patch for your kernel version from the `patches/` directory and enable `CONFIG_CPU_IDLE_GOV_NAP=y`:

```sh
cd /path/to/linux
patch -p1 < /path/to/nap/patches/stable/<kernel-version>-nap-<version>.patch
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
