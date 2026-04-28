Yes, the scales (and sometimes zero-points) do need to be stored, but they are stored **per-group** (or per-channel), not per-weight, so the overhead is negligible compared to the savings from compressing the weights themselves.

### How it is stored (Group-wise Quantization)

For the common case of **int8 weight-only quantization** (GPTQ, AWQ, AutoRound, Quanto, etc.), weights are quantized in **groups** (e.g., every 64, 128, or 256 consecutive weights share one scale).

| Component | Storage per parameter | Explanation |
|-----------|---------------------|-------------|
| **Quantized weights** | **1 byte** (int8) | One int8 value per weight |
| **Scales** | **2 bytes every *g* weights** | One fp16 (or fp32) scale per group of size *g* |
| **Zero-points** (if asymmetric) | **1-2 bytes every *g* weights** | Optional, for asymmetric schemes |

### Memory Math (Concrete Example)

Take a linear layer: **in_features=4,096**, **out_features=4,096** (16.7M parameters)

| Format | Weights | Scales | Total | Compression |
|--------|---------|--------|-------|-------------|
| **FP16 (baseline)** | 32.0 MB | — | **32.0 MB** | 1× |
| **Int8 per-channel** (group size = 4,096) | 16.0 MB | 8 KB (4,096 × 2 bytes) | **~16.0 MB** | 2× |
| **Int8 group-size=128** | 16.0 MB | 256 KB (4,096×4096/128 × 2 bytes) | **~16.3 MB** | **1.96×** |
| **Int4 group-size=128** | 8.0 MB | 256 KB | **~8.3 MB** | **3.86×** |

**Key insight**: The scale storage is amortized over the entire group. With a typical group size of 128, you pay only **2 bytes of overhead for every 128 bytes of weight data** (≈1.5% overhead), while cutting the weight storage by **50% or 75%**.

### De-quantization on the fly

During the forward pass, the kernel:
1. Loads the int8 weight block (128 values = 128 bytes)
2. Loads the single fp16 scale for that block (2 bytes)
3. De-quantizes to fp16/bf16 **in registers/shared memory**: $W_{fp} = \text{scale} \times W_{int8}$
4. Immediately multiplies with the fp16 activation

The scale is never expanded into a full matrix; it is applied "fused" during the GEMM, keeping the memory bandwidth low.

### Special Cases

- **LLM.int8() (BitsAndBytes)**: Uses a *mixed* scheme where most weights are int8 with a per-row scale, but outlier values (> threshold 6.0) are kept in fp16. This requires storing an additional sparse mask, but still yields ~1.5×–2× memory savings.
- **NF4/FP4 (QLoRA)**: Uses *double quantization*—the scales themselves are quantized (usually to 8-bit), so the overhead is even smaller (~0.5%).
- **GPTQ/AWQ**: Often store scales in fp16 and sometimes zero_points in int8, but with group sizes of 128 or 256, the overhead remains <2%.

**Bottom line**: The scales add a tiny fraction of memory (kilobytes to low megabytes) compared to the gigabytes saved by quantizing the weights from fp16→int8 or int4.



**AWQ** (Activation-aware Weight Quantization) is a quantization algorithm that recognizes **not all weights are equally important**—and protects the "salient" ones by looking at the magnitude of activations that multiply those weights.

While **normal group quantization** treats every weight in a group equally (applying the same scale to minimize MSE across the whole group), **AWQ scales weights differently based on how much their corresponding activation channels contribute to the output**.

---

### The Core Insight

In a linear layer: $Y = X \cdot W^T$

- Some **input channels** of $X$ have consistently larger magnitudes than others (e.g., certain feature dimensions in hidden states).
- The **columns** of $W$ corresponding to these "active" channels have outsized impact on the output.
- If you quantize these important columns aggressively, you get high error. If you quantize unimportant columns aggressively, you don't care.

**AWQ's trick**: Before quantizing, **scale up the important weight columns** by a factor $s > 1$, quantize them, then **scale down the corresponding activation channels by $1/s$**. Mathematically:
$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W^T) = X \cdot W^T$$

- The weights get "stretched" → they use more of the int4/int8 dynamic range → less quantization error on important channels.
- The activations get "squashed" → this is free at runtime (just a element-wise multiply).
- The scaling factor $s$ is fused into the group scale, so inference requires **no extra memory** compared to normal group quantization.

---

### Key Differences from Normal Group Quantization

| Feature | Normal Group Quantization (e.g., basic GPTQ, RTN) | AWQ |
|---------|-----------------------------------------------|-----|
| **What guides quantization** | Weight distribution only (minimize $\|W - \hat{W}\|$) | **Activation magnitudes** (protect channels where $\|X_{\cdot,j}\|$ is large) |
| **Pre-processing** | None (direct rounding) | **Per-channel scaling** of weights before group quantization |
| **Calibration data** | Optional (for GPTQ) | Required (to find which activation channels are "salient") |
| **Protection of outliers** | None uniform scale for whole group | **Non-uniform**: important channels get better effective precision |
| **Hardware compatibility** | Standard | Optimized for fused GPU kernels (auto-awq library) |
| **Quantization speed** | Fast (RTN) or Slow (GPTQ optimization) | Medium (grid search for optimal scaling factors) |

---

### How the Memory Layout Differs (It Doesn't Much)

Both store:
- **Int4/Int8 quantized weights** (grouped, e.g., 128 values per group)
- **Group scales** (fp16/fp32)

**AWQ adds**:
- **Per-channel scaling factors** ($s$ in the diagram above), but these are **fused into the group scales** during the search phase. The final model stores the exact same tensors as normal group quantization—weights + scales—because the per-channel scaling has been mathematically absorbed into the quantized weights and group scales.

**Visual Example** (simplified):

```
Weight Matrix W:
┌─────────────────────────────────┐
│  w1  w2  w3  w4  ...            │  ← Row 1
│  ...                            │
└─────────────────────────────────┘

Normal Group Quant (Group size 2):
  Group 1: [w1, w2] → Scale s_g1 (same for both)
  Group 2: [w3, w4] → Scale s_g2

AWQ (Group size 2):
  Channel 2 has large activations → scale w2 and w4 up by 1.5× before quant
  Group 1: [w1, 1.5*w2] → Quantize → stored as int8, scale absorbs the 1.5
  Group 2: [w3, 1.5*w4] → Quantize → same
  
  At runtime: 
  X → [x1, x2/1.5, x3, x4/1.5] → Multiply with quantized weights
```

---

### Practical Implications

1. **Better accuracy at 4-bit**: AWQ generally achieves lower perplexity than basic round-to-nearest (RTN) group quantization and is competitive with GPTQ but **much faster to compute** (no heavy Hessian-based reconstruction).

2. **Smaller group sizes work better**: Because AWQ protects salient channels via scaling, you can use smaller group sizes (e.g., 64 or 128) without the normal accuracy degradation, saving more memory.

3. **Inference**: AWQ requires the `auto-awq` library (or `transformers` with `AwqConfig`), which implements the fused dequantization + activation scaling kernel. You cannot run AWQ models with standard group-quantized kernels.

**In summary**: AWQ is "smarter" group quantization that uses activation statistics to pre-scale weight channels, giving more precision to weights that matter most, while keeping the same memory footprint as naive group quantization.

SmoothQuant is **not a weight-only quantization method**—it is a **joint activation-and-weight quantization** technique whose key idea is to **mathematically migrate the "outlier problem" from activations to weights**, where it is easier to handle, and then quantize both tensors to **INT8** without accuracy loss.

---

### The Problem SmoothQuant Solves

In large Transformer models, **activations contain severe outliers** (a few channels with very large magnitudes), while weights are relatively smooth.  
Normal quantization schemes try to quantize activations directly, but:
- A single outlier forces the whole tensor’s scale to be huge → 99% of values quantize to 0 → huge error.
- Keeping activations in FP16 defeats the purpose of INT8 acceleration.

SmoothQuant asks: *“Can we redistribute the outlier energy into weights, so both tensors become quantization-friendly?”*

---

### The Core Math Trick

For a linear layer $Y = X \cdot W^T$, introduce a **per-channel migration vector** $s \in \mathbb{R}^{C_{in}}$:

$$
Y = \underbrace{(X \cdot \text{diag}(s)^{-1})}_{\text{smoother } X'} \cdot \underbrace{(\text{diag}(s) \cdot W^T)}_{\text{larger } W'^T}
$$

- **Activations** $X$ are divided channel-wise by $s$ → outliers shrink → easy INT8.
- **Weights** $W$ are multiplied channel-wise by $s$ → the “outlier energy” is absorbed → weights still quantize well because they start small.
- **Output is unchanged** (mathematically identical).

The vector $s$ is chosen so that the **quantization error on both tensors is minimized**.  
A common closed-form choice is:

$$
s_j = \max(|X_{\cdot,j}|)^{\alpha} \; / \; \max(|W_{j,\cdot}|)^{1-\alpha}, \quad \alpha \in [0,1]
$$

with $\alpha=0.5$ as the default “50-50” balance.

---

### Workflow in One Sentence

1. **Offline calibration** → compute $s$ once per layer.  
2. **Fuse** $\text{diag}(s)$ into $W$ (weights are stored **once**, already scaled).  
3. **At runtime** → quantize activations to INT8 **on the fly** (smooth, no outliers), quantize the pre-scaled weights to INT8, execute INT8 GEMM, dequantize output back to FP16/BF16.

---

### Memory & Compute Footprint

| Tensor | Stored As | Bytes/elem | Notes |
|--------|-----------|------------|-------|
| Weights | **INT8** | 1 | Pre-smoothed, quantized offline |
| Activations | **INT8** | 1 | Quantized **online** each forward |
| Scales $s$ | **Absorbed** | 0 | Fused into weights, no extra tensor |
| Output | **FP16/BF16** | 2 | Dequantized after INT8 GEMM |

→ **4× weight memory reduction**, **2× activation bandwidth reduction**, **INT8 tensor-core throughput**.

---

### Comparison with “Normal” Quantization

| Aspect | Normal INT8 (weight-only or static) | SmoothQuant |
|--------|-------------------------------------|-------------|
| **What is quantized** | Weights only, **or** weights + activations with difficult outliers | **Both weights & activations to INT8** |
| **Outlier handling** | Keep activations in FP16, or accept large error | **Mathematically migrate outliers to weights** |
| **Calibration data** | Optional for weights, required for activations | Required (once) to compute $s$ |
| **Extra runtime cost** | None (weight-only) or high (activation outliers) | **One cheap per-channel multiply** before INT8 GEMM |
| **Accuracy** | Degrades quickly on ≥ 6 B models when activations are quantized | **Maintains FP16 accuracy** on 176 B OPT / 530 B MT-NLG |
| **Hardware** | Any INT8 kernel | Needs **SmoothQuant-enabled kernel** (cuBLAS, Torch, FasterTransformer, etc.) |

---

### Practical Take-away

SmoothQuant lets you run **entire transformer layers in INT8 arithmetic** (weights + activations + GEMM) without the accuracy collapse that normally happens when you try to quantize activation outliers.  
It does so by **pre-scaling weights** once offline, making activations smooth enough for trivial per-tensor INT8 quantization, while keeping the memory footprint identical to vanilla INT8 weight-only schemes.


AWQ and SmoothQuant both **pre-scale** weights and **counter-scale** activations with the same per-channel vector `s`, but they **solve two different problems**, **operate at different granularities**, and **target different deployment modes**.

| Aspect | AWQ (Activation-aware **Weight** Quantization) | SmoothQuant (Smooth **Activation** Quantization) |
|--------|-----------------------------------------------|--------------------------------------------------|
| **Goal** | Make **weights** quantizable to **INT4/INT8** **without hurting accuracy** | Make **activations** quantizable to **INT8** **without hurting accuracy** |
| **Pain point addressed** | Some **weight columns** are more important (large activation magnitudes) | **Activation outliers** (a few channels with huge values) |
| **What is stored** | **INT4/INT8 weights** only; activations stay in **FP16/BF16 at runtime** | **INT8 weights** **and** **INT8 activations** at runtime |
| **Where `s` is applied** | **Offline**: scale **weights up**, store them already scaled; **Online**: divide **activations down** | **Offline**: scale **weights up**, store them already scaled; **Online**: divide **activations down** |
| **Granularity of `s`** | **Per-channel** (one value per input channel) | **Per-channel** (same math) |
| **Calibration data** | Needed to find “salient” channels | Needed to find outlier channels |
| **Runtime tensor types** | `X=FP16, W=INT4/8, Y=FP16` | `X=INT8, W=INT8, Y=FP16/BF16` |
| **Memory bandwidth** | Same as weight-only INT4/8 | **½ activation BW**, **¼ weight BW** vs FP16 |
| **Arithmetic** | GEMM in **FP16** (weights de-quant on the fly) | GEMM in **INT8** (tensor-core) |
| **Hardware target** | GPUs with fast INT4/INT8 de-quant (Ada, Hopper, etc.) | Any GPU with INT8 tensor cores (Turing+) |
| **Typical use-case** | **4-bit inference** when you can’t afford FP16 activations | **Full INT8 inference** when you want **both weights & activations** in INT8 |

---

One-sentence summary  
**AWQ** keeps activations in FP16 and uses scaling to squeeze **4-bit weights**; **SmoothQuant** keeps **both tensors in INT8** by using the **same scaling trick** to neutralize **activation outliers**.


QAT = **Quantization-Aware Training**  
It is **not a new quantization format**; it is a **training strategy** that **inserts fake-quantization operations into the forward pass** so the network learns **weight values that are robust to the rounding error** that will occur when the model is **actually quantized to INT8/INT4** after training.

---

### Key points in one sentence each

1. **Fake quant op**  
   During forward: `w_q = round( clamp(w, min, max) / scale ) * scale`  
   Gradients flow through the **straight-through estimator** (treat round as identity).

2. **When it happens**  
   **Before deployment**—you train/fine-tune with QAT, then **export to standard INT8/INT4 weights** (TFLite, ONNX, torch, AWQ, GPTQ, etc.).

3. **What changes permanently**  
   Only the **weight values** (and maybe activation statistics); the **bit-width, scales, zero-points** are chosen by the exporter exactly as in **post-training quantization**.

4. **Memory at inference**  
   Same as ordinary quantized model: **1 byte per weight** for INT8, **0.5 byte** for INT4, etc. QAT itself is **training-time only**.

5. **Accuracy benefit**  
   Typically **1–3 % BLEU / top-1 gain** over post-training quantization, especially for **tiny models, 4-bit, or tasks sensitive to outliers**.

6. **Compute cost**  
   **~1.2–1.5× training slowdown** (extra clamp/round ops), but **no runtime cost** once exported.

---

### Minimal PyTorch snippet

```python
from torch.quantization import QuantStub, DeQuantStub, fake_quantize_per_tensor

class QATLinear(nn.Module):
    def __init__(self, in_f, out_f, qbits=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f))
        self.scale_w = nn.Parameter(torch.tensor(0.01), requires_grad=False)
        self.qbits = qbits

    def forward(self, x):
        # fake-quantize weights
        w_q = fake_quantize_per_tensor(self.weight, self.scale_w, 0,
                                       -128, 127)          # INT8 example
        return F.linear(x, w_q)
```

After convergence you call `torch.quantization.convert(model)` and get a **real INT8 module** with the **learned weights**.

---

### Relationship to other schemes

| Scheme | QAT? | Learns weights? | Runtime dtype |
|--------|------|-----------------|---------------|
| **GPTQ / AWQ** | ❌ (PTQ) | ❌ | INT4/8 weights, FP16 activations |
| **SmoothQuant** | ❌ (PTQ) | ❌ | INT8 weights + INT8 activations |
| **QAT** | ✅ | ✅ | Whatever you fake-trained (INT8, INT4, FP8, etc.) |

---

### TL;DR

QAT = **“teach the model to live with rounded numbers”** by **pretending to quantize during training**; after that you **throw away the fake ops** and ship a **regular low-bit model** that now **just works better**.


AWQ does **not** use an analytic formula like SmoothQuant.  
Instead it **searches** for the per-channel scale vector `s` that **minimizes the quantization error of the *weights*** while **keeping the activations in FP16** (so no activation-error term is needed).  
The search is cheap, layer-by-layer, **once on calibration data**, and the criterion is purely **weight reconstruction MSE**.

---

### AWQ scale-search algorithm (high-level)

1. **Collect** a few hundred calibration samples → get input activations `X` (FP16).  
2. **Estimate channel importance**  
   `imp_j = mean(|X_{:,j}|)` across tokens → large `imp_j` ⇒ salient channel.  
3. **Define candidate scales**  
   `s_j = imp_j ^ α` with `α` in a small grid (e.g. `{0.5, 0.6, … 1.0}`).  
   (Only **one scalar α per layer** is tuned, not a full vector.)  
4. **Grid search**  
   For each α  
   - scale weights: `W' = diag(s) · W`  
   - quantize / de-quantize `W'` to target bits (INT4 or INT8)  
   - compute `MSE(W' - dequant(W'))`  
   Pick the α with **lowest MSE**.  
5. **Absorb** the chosen `s` into the **quantized weights** and store them; **discard `s`**.

---

### Key differences vs SmoothQuant

| Aspect | AWQ | SmoothQuant |
|--------|-----|-------------|
| **Objective** | Minimize **weight reconstruction error** | Minimize **activation+weight error** |
| **Formula** | **No closed form**; grid search over `α` in `s_j = imp_j^α` | **Closed-form**: `s_j = max(|X|)^α / max(|W|)^{1-α}` |
| **Activation dtype at runtime** | **FP16** (activations **not** quantized) | **INT8** (activations **are** quantized) |
| **Search space** | **1 scalar α per layer** (10–20 candidates) | **0 scalars** (analytic) |
| **Calibration cost** | **Seconds** (lightweight) | **Milliseconds** (formula) |

---

### Intuition

AWQ only needs the **relative importance** of channels to decide **which weights deserve more effective precision**; a single exponent `α` is enough to stretch them appropriately.  
SmoothQuant needs an **exact balance** between shrinking outliers **and** keeping weights quantizable, hence the analytic two-term ratio.

So:  
**SmoothQuant → closed-form balance** for **INT8 activations + weights**.  
**AWQ → tiny grid search** to protect **important weight columns** while **activations stay in FP16**.