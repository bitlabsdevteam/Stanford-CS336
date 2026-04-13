# Transformer LM Resource Accounting

Assumptions used throughout:

- Architecture matches the assignment implementation: token embedding -> `num_layers` pre-norm Transformer blocks -> final RMSNorm -> tied LM head.
- The LM head is tied to the token embedding matrix, so it adds no trainable parameters.
- FLOPs accounting includes only matrix multiplies, exactly as requested.
- Batch size is `1`.
- Sequence length is `T = context_length`.

Let:

- `V = vocab_size`
- `T = context_length`
- `L = num_layers`
- `d = d_model`
- `h = num_heads`
- `d_h = d / h`
- `d_ff =` feed-forward hidden dimension

## (a) GPT-2 XL parameter count and model memory

Configuration:

- `vocab_size = 50,257`
- `context_length = 1,024`
- `num_layers = 48`
- `d_model = 1,600`
- `num_heads = 25`
- `d_ff = 4,288`

### Parameter formula

The trainable parameter count is:

\[
Vd + L(4d^2 + 3dd_{ff} + 2d) + d
\]

because:

- token embedding: `Vd`
- per block attention projections: `4d^2`
- per block SwiGLU: `3dd_ff`
- per block two RMSNorm layers: `2d`
- final RMSNorm: `d`

### GPT-2 XL count

- token embedding: `50,257 * 1,600 = 80,411,200`
- per block: `4(1,600^2) + 3(1,600)(4,288) + 2(1,600) = 30,825,600`
- 48 blocks: `48 * 30,825,600 = 1,479,628,800`
- final RMSNorm: `1,600`

Total trainable parameters:

\[
80,411,200 + 1,479,628,800 + 1,600 = 1,560,041,600
\]

So the model has **1,560,041,600 trainable parameters**.

If each parameter is stored in fp32, memory to load the model is:

\[
1,560,041,600 \times 4 = 6,240,166,400 \text{ bytes}
\]

which is about **6.24 GB**, or about **5.81 GiB**.

## (b) GPT-2 XL forward-pass matrix multiplies and total FLOPs

We use the rule:

\[
(m \times n)(n \times p) \Rightarrow 2mnp \text{ FLOPs}
\]

### Matrix multiplies in one Transformer block

For one block, with input activations of shape `(T, d)`:

1. Query projection
   - `(T x d) @ (d x d)`
   - FLOPs: `2Td^2`

2. Key projection
   - `(T x d) @ (d x d)`
   - FLOPs: `2Td^2`

3. Value projection
   - `(T x d) @ (d x d)`
   - FLOPs: `2Td^2`

4. Output projection
   - `(T x d) @ (d x d)`
   - FLOPs: `2Td^2`

5. Attention score matrix `QK^T`
   - per head: `(T x d_h) @ (d_h x T)`
   - across all heads: `2T^2d`

6. Attention application `AV`
   - per head: `(T x T) @ (T x d_h)`
   - across all heads: `2T^2d`

7. SwiGLU `W1`
   - `(T x d) @ (d x d_ff)`
   - FLOPs: `2Tdd_ff`

8. SwiGLU `W3`
   - `(T x d) @ (d x d_ff)`
   - FLOPs: `2Tdd_ff`

9. SwiGLU `W2`
   - `(T x d_ff) @ (d_ff x d)`
   - FLOPs: `2Tdd_ff`

So one block requires:

\[
8Td^2 + 4T^2d + 6Tdd_{ff}
\]

### Final LM head

After the last block and final RMSNorm, the tied LM head computes:

- `(T x d) @ (d x V)`
- FLOPs: `2TdV`

### GPT-2 XL numerical values

Use:

- `T = 1,024`
- `L = 48`
- `d = 1,600`
- `d_ff = 4,288`
- `V = 50,257`

#### Attention projections over all 48 blocks

\[
L \cdot 8Td^2
= 48 \cdot 8 \cdot 1,024 \cdot 1,600^2
= 1,006,632,960,000
\]

#### Attention score matmuls over all 48 blocks

\[
L \cdot 2T^2d
= 48 \cdot 2 \cdot 1,024^2 \cdot 1,600
= 161,061,273,600
\]

#### Attention value matmuls over all 48 blocks

\[
L \cdot 2T^2d
= 161,061,273,600
\]

#### SwiGLU matmuls over all 48 blocks

\[
L \cdot 6Tdd_{ff}
= 48 \cdot 6 \cdot 1,024 \cdot 1,600 \cdot 4,288
= 2,023,332,249,600
\]

#### Final tied LM head

\[
2TdV
= 2 \cdot 1,024 \cdot 1,600 \cdot 50,257
= 164,682,137,600
\]

### Total FLOPs

\[
1,006,632,960,000
+ 161,061,273,600
+ 161,061,273,600
+ 2,023,332,249,600
+ 164,682,137,600
= 3,516,769,894,400
\]

So the total matrix-multiply cost of one forward pass is:

\[
\boxed{3,516,769,894,400 \text{ FLOPs}}
\]

which is about **3.52 TFLOPs**.

## (c) Which parts require the most FLOPs?

For GPT-2 XL at `T = 1,024`, the **SwiGLU feed-forward network** is the largest contributor, accounting for about **57.53%** of the total FLOPs. The next largest cost is the set of four attention projections, at about **28.62%**, while the quadratic attention score/value matmuls are much smaller at this context length.

## (d) GPT-2 small, medium, large, and XL FLOPs breakdown

For the assignment architecture, use:

- GPT-2 small: `L = 12`, `d = 768`, `h = 12`, `d_ff = 2,048`
- GPT-2 medium: `L = 24`, `d = 1,024`, `h = 16`, `d_ff = 2,752`
- GPT-2 large: `L = 36`, `d = 1,280`, `h = 20`, `d_ff = 3,392`
- GPT-2 XL: `L = 48`, `d = 1,600`, `h = 25`, `d_ff = 4,288`

All numbers below assume `T = 1,024`.

### GPT-2 small

- Total FLOPs: `291,648,307,200`
- Attention projections: `57,982,058,496` = **19.88%**
- Attention scores: `19,327,352,832` = **6.63%**
- Attention apply: `19,327,352,832` = **6.63%**
- FFN: `115,964,116,992` = **39.76%**
- LM head: `79,047,426,048` = **27.10%**

### GPT-2 medium

- Total FLOPs: `830,172,299,264`
- Attention projections: `206,158,430,208` = **24.83%**
- Attention scores: `51,539,607,552` = **6.21%**
- Attention apply: `51,539,607,552` = **6.21%**
- FFN: `415,538,085,888` = **50.05%**
- LM head: `105,396,568,064` = **12.70%**

### GPT-2 large

- Total FLOPs: `1,768,530,903,040`
- Attention projections: `483,183,820,800` = **27.32%**
- Attention scores: `96,636,764,160` = **5.46%**
- Attention apply: `96,636,764,160` = **5.46%**
- FFN: `960,327,843,840` = **54.30%**
- LM head: `131,745,710,080` = **7.45%**

### GPT-2 XL

- Total FLOPs: `3,516,769,894,400`
- Attention projections: `1,006,632,960,000` = **28.62%**
- Attention scores: `161,061,273,600` = **4.58%**
- Attention apply: `161,061,273,600` = **4.58%**
- FFN: `2,023,332,249,600` = **57.53%**
- LM head: `164,682,137,600` = **4.68%**

### How the proportions change with model size

As model size increases at fixed context length, the **FFN** and **attention projection** terms take up proportionally more of the total FLOPs, because they scale like `LTd^2` or `LTdd_ff`, and `d_ff` grows roughly linearly with `d`. In contrast, the **LM head** takes up proportionally less, and the quadratic attention score/value terms also shrink proportionally, because they scale only like `LT^2d`, which grows more slowly in `d` than the `d^2` terms.

## (e) GPT-2 XL with context length 16,384

Keep GPT-2 XL the same, but change:

- `T = 16,384`

## Topic 4.3: AdamW resource accounting

Use the shorthand:

- `B = batch_size`
- `V = vocab_size`
- `T = context_length`
- `L = num_layers`
- `d = d_model`
- `h = num_heads`
- `d_ff = 8d / 3`

All tensors are assumed to be float32, so each stored scalar takes 4 bytes.

### (a) Peak memory for training with AdamW

With tied input/output embeddings, the trainable parameter count is

\[
P = Vd + L(4d^2 + 3dd_{ff} + 2d) + d
= Vd + L(12d^2 + 2d) + d.
\]

So the parameter memory is

\[
M_{\text{params}} = 4P.
\]

The gradient tensor has the same shape as the parameters, so

\[
M_{\text{grads}} = 4P.
\]

AdamW keeps first- and second-moment tensors for every parameter, so the optimizer
state is

\[
M_{\text{opt}} = 8P.
\]

For activations, count only the components requested in the assignment.

Per Transformer block:

- two RMSNorm outputs: `2BTd`
- attention activations:
  - Q, K, V projections: `3BTd`
  - QK^T scores: `BhT^2`
  - softmax output: `BhT^2`
  - weighted value sum: `BTd`
  - output projection: `BTd`
- SwiGLU activations:
  - gate projection: `BTd_ff`
  - SiLU output on gate branch: `BTd_ff`
  - value projection: `BTd_ff`
  - element-wise product: `BTd_ff`
  - output projection: `BTd`

That gives

\[
A_{\text{block}} = 2BTd + 5BTd + 4BTd_{ff} + 2BhT^2 + BTd
= 8BTd + 4BTd_{ff} + 2BhT^2.
\]

Substituting `d_ff = 8d/3`,

\[
A_{\text{block}} = \frac{56}{3}BTd + 2BhT^2.
\]

Outside the blocks:

- final RMSNorm: `BTd`
- output embedding / logits: `BTV`
- cross-entropy on logits: `BTV`

So total activation storage is

\[
A = L\left(8BTd + 4BTd_{ff} + 2BhT^2\right) + BTd + 2BTV
\]

or, after substituting `d_ff = 8d/3`,

\[
A = B\left(\left(\frac{56}{3}L + 1\right)Td + 2LhT^2 + 2TV\right).
\]

The activation memory is

\[
M_{\text{acts}} = 4A.
\]

Putting everything together,

\[
M_{\text{peak}} = M_{\text{params}} + M_{\text{grads}} + M_{\text{opt}} + M_{\text{acts}}
= 16P + 4A \text{ bytes}.
\]

### (b) GPT-2 XL instantiation and max batch size on 80 GB

For GPT-2 XL, use:

- `V = 50,257`
- `T = 1,024`
- `L = 48`
- `d = 1,600`
- `h = 25`
- `d_ff = 8d/3`

Parameter count:

\[
P = 50{,}257(1{,}600) + 48(12(1{,}600)^2 + 2(1{,}600)) + 1{,}600
= 1{,}555{,}126{,}400.
\]

So:

\[
M_{\text{params}} = 6{,}220{,}505{,}600 \text{ bytes}
\]

\[
M_{\text{grads}} = 6{,}220{,}505{,}600 \text{ bytes}
\]

\[
M_{\text{opt}} = 12{,}441{,}011{,}200 \text{ bytes}.
\]

The activation term becomes

\[
A = B\left(\left(\frac{56}{3}\cdot 48 + 1\right)(1{,}024)(1{,}600) + 2(48)(25)(1{,}024)^2 + 2(1{,}024)(50{,}257)\right)
\]

\[
= 4{,}089{,}153{,}536 \, B \text{ elements},
\]

so

\[
M_{\text{acts}} = 16{,}356{,}614{,}144 \, B \text{ bytes}.
\]

Therefore the total peak memory is

\[
M_{\text{peak}}(B) = 16{,}356{,}614{,}144\, B + 24{,}882{,}022{,}400 \text{ bytes}.
\]

In decimal GB this is approximately

\[
16.36 \cdot B + 24.88 \text{ GB}.
\]

Under an 80 GB limit,

\[
B_{\max} = \left\lfloor\frac{80{,}000{,}000{,}000 - 24{,}882{,}022{,}400}{16{,}356{,}614{,}144}\right\rfloor = 3.
\]

So the largest batch size that fits is **3**.

### (c) FLOPs for one AdamW step

Let `P` be the number of trainable parameters.

For each parameter element, AdamW does approximately:

- weight decay: 2 FLOPs
- first moment update: 3 FLOPs
- second moment update: 4 FLOPs
- parameter update using `sqrt`, division, multiply, and subtraction: 5 FLOPs

So one AdamW step takes

\[
F_{\text{AdamW}} \approx 14P
\]

FLOPs, ignoring the `O(1)` scalar bias-correction work.

For this model family,

\[
F_{\text{AdamW}} \approx 14\left(Vd + L(12d^2 + 2d) + d\right).
\]

For GPT-2 XL specifically,

\[
F_{\text{AdamW}} \approx 14(1{,}555{,}126{,}400) = 21{,}771{,}769{,}600
\]

FLOPs.

### (d) Training time on one H100 at 50% MFU

For one forward pass with batch size `B`, the model FLOPs are

\[
F_{\text{fwd}}(B) = B\left(L(8Td^2 + 4T^2d + 6Tdd_{ff}) + 2TdV\right).
\]

With `d_ff = 8d/3`, this is

\[
F_{\text{fwd}}(B) = B\left(L(24Td^2 + 4T^2d) + 2TdV\right).
\]

For GPT-2 XL, a forward pass at `B = 1024` is

\[
F_{\text{fwd}}(1024) = 3.5908644503552 \times 10^{15}
\]

FLOPs.

Using the rule "backward = 2 x forward", one full training step is

\[
F_{\text{step}} = 3F_{\text{fwd}}(1024) + F_{\text{AdamW}}
\approx 1.07726151203352 \times 10^{16}
\]

FLOPs.

An H100 at 50% MFU delivers

\[
0.5 \times 495 \times 10^{12} = 2.475 \times 10^{14}
\]

FLOPs/s.

So the time per step is

\[
t_{\text{step}} = \frac{F_{\text{step}}}{2.475 \times 10^{14}} \approx 43.53 \text{ s}.
\]

For `400,000` steps,

\[
t_{\text{train}} = 400{,}000 \cdot 43.53 \text{ s}
= 1.741 \times 10^{7} \text{ s}
\approx 4{,}836 \text{ hours}.
\]

So the training run would take about **4,836 hours**, or about **201.5 days**.

This uses `batch_size = 1024` in the same sense as part (b), namely 1024 sequences of
length 1024 per step. That is a compute-only estimate; it does not fit in 80 GB from
part (b).

Then the total forward-pass FLOPs become:

\[
133,577,729,638,400
\]

which is about **133.58 TFLOPs**. Compared with the `T = 1,024` case, this is about a **37.98x** increase.

The relative FLOPs breakdown changes to:

- Attention projections: **12.06%**
- Attention scores: **30.87%**
- Attention apply: **30.87%**
- FFN: **24.24%**
- LM head: **1.97%**

So at very long context length, the quadratic attention score and value matmuls dominate the computation, together accounting for about **61.73%** of the total FLOPs.
