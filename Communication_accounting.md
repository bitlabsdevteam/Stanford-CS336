# 2.4 4D Parallelism: Communication Accounting

Assumptions used throughout:

- Model config XXL has `d_model = 16384`, `d_ff = 53248`, `num_blocks = 126`.
- Each block is exactly two linear layers:
  - `W_in : d_model -> d_ff`
  - `W_out : d_ff -> d_model`
- Attention, embeddings, output projection, and nonlinearities are ignored exactly as stated.
- Activations and gradient communication are in BF16 (`2` bytes/element).
- Master weights, accumulated gradients, and Adam optimizer state are in FP32 (`4` bytes/element).
- Adam stores two FP32 moment tensors per parameter.

Let

\[
D = 16384,\quad F = 53248,\quad L = 126.
\]

Each block has

\[
DF + FD = 2DF
\]

parameters, so the total parameter count is

\[
P = 2LDF
= 2 \cdot 126 \cdot 16384 \cdot 53248
= 219{,}848{,}638{,}464.
\]

## (a) Single-device memory

FP32 training state on one device:

- master weights: `4P` bytes
- accumulated gradients: `4P` bytes
- Adam moments: `8P` bytes

So

\[
M_{\text{state}}
= (4 + 4 + 8)P
= 16P
= 3{,}517{,}578{,}215{,}424 \text{ bytes}.
\]

This is:

- `3.5176 TB` in decimal units
- `3276 GiB`
- `3.20 TiB`

Saved-for-backward activations are batch-dependent. For one block we must save:

- input to `W_in`: shape `(B, D)`
- input to `W_out`: shape `(B, F)`

So total saved activations are

\[
A_{\text{bf16}}
= 2L B(D + F)
= 2 \cdot 126 \cdot B \cdot (16384 + 53248)
= 17{,}547{,}264 \, B \text{ bytes}.
\]

That is about:

- `17.55 MB` per token in decimal units
- `16.73 MiB` per token in binary units

Equivalent H100 80GB counts:

\[
\frac{M_{\text{state}}}{80 \cdot 10^9}
= \frac{3.5176 \cdot 10^{12}}{80 \cdot 10^9}
\approx 43.97.
\]

So the FP32 state alone is about **44 H100 80GB GPUs** worth of memory.

One-sentence response: storing the FP32 master weights, accumulated gradients, and Adam state takes about **3.52 TB**, the saved BF16 activations cost **17.55 MB per token of batch**, and the FP32 state alone is about **44 H100 80GB GPUs** worth of memory.

## (b) Per-device memory with FSDP sharding

Now shard the master weights, optimizer state, gradients, and half the saved activations across `N_FSDP` devices.

The sharded FP32 state per device is

\[
\frac{16P}{N_{\text{FSDP}}}.
\]

Half the activations remain unsharded and half are sharded, so activation memory per device is

\[
\frac{1}{2}A_{\text{bf16}} + \frac{1}{2}\frac{A_{\text{bf16}}}{N_{\text{FSDP}}}.
\]

Therefore

\[
M_{\text{per-device}}
= \frac{16P}{N_{\text{FSDP}}}
+ \frac{A_{\text{bf16}}}{2}
+ \frac{A_{\text{bf16}}}{2N_{\text{FSDP}}}.
\]

Substituting the constants:

\[
M_{\text{per-device}}
= \frac{3{,}517{,}578{,}215{,}424}{N_{\text{FSDP}}}
+ 8{,}773{,}632 \, B
+ \frac{8{,}773{,}632 \, B}{N_{\text{FSDP}}}
\quad \text{bytes}.
\]

Equivalently,

\[
M_{\text{per-device}}
= \frac{16P}{N_{\text{FSDP}}}
+ 126B(D+F)\left(1 + \frac{1}{N_{\text{FSDP}}}\right).
\]

To fit on one TPU v5p device we need

\[
M_{\text{per-device}} < 95 \cdot 10^9 \text{ bytes}.
\]

Because the activation term depends on `B`, the exact `N_FSDP` depends on the batch size. The hard lower bound from FP32 state memory alone is

\[
\frac{16P}{N_{\text{FSDP}}} < 95 \cdot 10^9
\Rightarrow
N_{\text{FSDP}} > \frac{3{,}517{,}578{,}215{,}424}{95 \cdot 10^9}
\approx 37.03.
\]

So we need at least

\[
N_{\text{FSDP}} = 38
\]

before even accounting for activations.

One-sentence response: the per-device memory is
\[
\frac{16P}{N_{\text{FSDP}}} + \frac{A_{\text{bf16}}}{2} + \frac{A_{\text{bf16}}}{2N_{\text{FSDP}}},
\]
and the parameter-state-only lower bound is **`N_FSDP >= 38`**, with the true requirement larger once the batch-dependent activation term is included.

## (c) Compute-bound batch size for mixed FSDP + TP

Use the Scaling Book formulas for mixed FSDP + TP forward-pass communication:

\[
T_{\text{FSDP comms}}(B, X, Y) = \frac{4DF}{Y W_{\text{ici}} M_X},
\]
\[
T_{\text{TP comms}}(B, X, Y) = \frac{4BD}{X W_{\text{ici}} M_Y},
\]
\[
T_{\text{math}} = \frac{4BDF}{N C},
\]

with

\[
W_{\text{ici}} = 2 \cdot 9 \cdot 10^{10} = 1.8 \cdot 10^{11},
\quad
C = 4.6 \cdot 10^{14},
\]
\[
M_X = 2,\quad M_Y = 1,\quad X = 16,\quad Y = 4,\quad N = XY = 64.
\]

We are compute bound when

\[
\max(T_{\text{FSDP comms}}, T_{\text{TP comms}}) < T_{\text{math}}.
\]

First compare FSDP comms to math:

\[
\frac{4DF}{Y W_{\text{ici}} M_X}
<
\frac{4BDF}{N C}.
\]

Cancel `4DF`:

\[
\frac{1}{Y W_{\text{ici}} M_X}
<
\frac{B}{N C}.
\]

So

\[
B > \frac{N C}{Y W_{\text{ici}} M_X}.
\]

Plugging in numbers:

\[
B > \frac{64 \cdot 4.6 \cdot 10^{14}}{4 \cdot 1.8 \cdot 10^{11} \cdot 2}
= 20{,}444.44.
\]

Now compare TP comms to math:

\[
\frac{4BD}{X W_{\text{ici}} M_Y}
<
\frac{4BDF}{N C}.
\]

Cancel `4BD`:

\[
\frac{1}{X W_{\text{ici}} M_Y}
<
\frac{F}{N C}.
\]

So

\[
F > \frac{N C}{X W_{\text{ici}} M_Y}.
\]

Plugging in numbers:

\[
\frac{N C}{X W_{\text{ici}} M_Y}
= \frac{64 \cdot 4.6 \cdot 10^{14}}{16 \cdot 1.8 \cdot 10^{11} \cdot 1}
= 10{,}222.22.
\]

Since `F = 53248 > 10222.22`, the TP condition is already satisfied, so the FSDP term is the bottleneck.

Thus the minimum total batch size is approximately

\[
B_{\min} \approx 20{,}445 \text{ tokens}.
\]

Because batch is sharded over the FSDP dimension `X = 16`, the per-device batch threshold is

\[
\frac{B}{X} > \frac{20{,}444.44}{16} \approx 1277.78.
\]

So the smallest integer per-device batch is

\[
1278 \text{ tokens/device},
\]

and the corresponding smallest integer overall batch divisible by `16` is

\[
B = 16 \cdot 1278 = 20{,}448 \text{ tokens}.
\]

One-sentence response: with `X = 16`, `Y = 4`, `M_X = 2`, and `M_Y = 1`, this model becomes compute bound at about **1278 tokens per device**, corresponding to an overall batch of about **20,448 tokens**.

## (d) How to reduce batch size while keeping throughput high

To reduce the required global batch size without falling into the communication-bound regime, the main tools are to add more efficient parallelism and improve overlap. Context parallelism helps because, for the MLP, tokens are interchangeable, so part of what would otherwise be pure batch parallelism can be moved onto the sequence dimension while still increasing effective data-parallel bandwidth. Sequence parallelism also reduces activation memory in tensor-parallel regions by sharding activations across the sequence dimension instead of gathering the full hidden state everywhere, which lets us run smaller local batches before memory becomes the limiter. Pipeline parallelism with microbatching helps keep devices busy at smaller microbatch sizes: with `p` pipeline stages and `m` microbatches, the bubble fraction scales like roughly `(p - 1) / m`, so increasing `m` and using schedules such as 1F1B or interleaving improves utilization without needing a single huge monolithic batch. In practice, the usual recipe is to keep TP and sequence-parallel groups within fast local interconnect domains, use FSDP across a larger dimension, add PP when the model is too deep or too large to fit efficiently otherwise, and overlap collectives with compute as aggressively as possible.

One-paragraph response: the main way to shrink batch size without sacrificing throughput is to combine better communication overlap with more memory-efficient parallelism, especially context parallelism, sequence parallelism, and pipeline parallelism with many microbatches; context/sequence parallelism effectively spreads tokens and activations across more devices, reducing local memory pressure, while pipelining reduces idle time because the bubble fraction scales roughly as `(p - 1) / m`, so larger `m` keeps utilization high even when each microbatch is small.
