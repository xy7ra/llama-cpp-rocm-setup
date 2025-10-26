# Multi-GPU Setup: How Tensor Parallelism Actually Works

**The most important guide in this repo.** This is how you run 72B models on consumer hardware.

---

## The Problem

You have 2× GPUs with 24GB VRAM each = 48GB total.

You want to run a 40GB model.

**Most people do this wrong.**

---

## ❌ The Wrong Way: Model Duplication

### What Ollama Does (Don't Do This)
```bash
# Loads ENTIRE model on GPU 0: 40GB
# Loads ENTIRE model on GPU 1: 40GB
# Total needed: 80GB
# Your hardware: 48GB
# Result: CRASH 💥
```

This is called **model parallelism** or **pipeline parallelism**. Each GPU gets a full copy of the model.

**Why it's bad:**
- Wastes VRAM (duplicate data)
- Can't run big models
- Only helps with multiple requests (not single user)

---

## ✅ The Right Way: Tensor Parallelism

### What llama.cpp Does (Do This)
```bash
/opt/llama.cpp-rocm/bin/llama-server \
  --model model-40GB.gguf \
  --n-gpu-layers 999 \
  --tensor-split 0.5,0.5 \    # ← THE MAGIC FLAG
  --port 8080
```

**What happens:**
- Model tensors are **split** across GPUs
- GPU 0 holds: ~20GB of model
- GPU 1 holds: ~20GB of model
- Total: 40GB (fits in 48GB!)

**Both GPUs work together on EVERY forward pass.**

---

## How Tensor Splitting Works (ELI5)

Imagine the model is a giant spreadsheet with billions of numbers.

### Model Duplication (Wrong)
```
GPU 0: [Complete spreadsheet copy]  ← 40GB
GPU 1: [Complete spreadsheet copy]  ← 40GB
Total: 80GB needed ❌
```

### Tensor Splitting (Right)
```
GPU 0: [Columns 1-50 of spreadsheet]  ← 20GB
GPU 1: [Columns 51-100 of spreadsheet] ← 20GB
Total: 40GB needed ✅
```

When the model needs to do math:
1. **Split the work** across both GPUs
2. Each GPU processes its columns
3. **Combine results** and continue

Both GPUs are always working. No duplication. Maximum efficiency.

---

## The Tensor Split Flag Explained

### Format
```bash
--tensor-split RATIO_GPU0,RATIO_GPU1[,RATIO_GPU2,...]
```

### Examples

**Equal split (most common):**
```bash
--tensor-split 0.5,0.5
```
- GPU 0 gets 50% of tensors
- GPU 1 gets 50% of tensors

**Asymmetric split:**
```bash
--tensor-split 0.6,0.4
```
- GPU 0 gets 60% (maybe it has more free VRAM)
- GPU 1 gets 40%

**Three GPUs:**
```bash
--tensor-split 0.33,0.33,0.34
```
- GPU 0: 33%
- GPU 1: 33%
- GPU 2: 34% (slightly more to handle rounding)

**Four GPUs:**
```bash
--tensor-split 0.25,0.25,0.25,0.25
```

---

## Real Example: Qwen2.5-72B (30GB model)

### Command
```bash
/opt/llama.cpp-rocm/bin/llama-server \
  --model Qwen2.5-72B-Instruct.i1-IQ3_XXS.gguf \
  --n-gpu-layers 999 \
  --tensor-split 0.5,0.5 \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 32768
```

### VRAM Usage (from rocm-smi)
```
GPU 0 (0000:03:00.0): 19.8GB / 24GB used
GPU 1 (0000:07:00.0): 20.2GB / 24GB used
Total: 40GB (30GB model + 10GB context)
```

### Performance
- **Prompt processing**: 347 tokens/sec
- **Generation**: 15 tokens/sec
- **Both GPUs at ~95% utilization**

---

## Why This Is Better

### VRAM Efficiency
| Setup | VRAM Needed | Can Run 72B? |
|-------|-------------|--------------|
| No GPU (CPU only) | 0 VRAM | ✅ (but 0.5 tok/s) |
| Single GPU | 40GB+ | ❌ (24GB available) |
| Model duplication | 80GB+ | ❌ (48GB available) |
| **Tensor split** | **40GB** | **✅ (48GB available)** |

### Speed Comparison
| Setup | Tokens/Sec |
|-------|------------|
| CPU (64 cores) | ~0.5 |
| Single GPU (all on GPU 0) | Would crash |
| **Tensor split (2 GPUs)** | **347 (prompt), 15 (gen)** |

---

## Common Mistakes

### Mistake 1: Not Using Tensor Split
```bash
# ❌ WRONG: Tries to fit 40GB model on one 24GB GPU
llama-server -m model-40GB.gguf -ngl 999
```

```bash
# ✅ RIGHT: Splits across both GPUs
llama-server -m model-40GB.gguf -ngl 999 -ts 0.5,0.5
```

### Mistake 2: Wrong Ratios
```bash
# ❌ WRONG: Ratios must sum to 1.0
--tensor-split 0.5,0.5,0.5  # = 1.5, invalid
```

```bash
# ✅ RIGHT: Ratios sum to 1.0
--tensor-split 0.33,0.33,0.34  # = 1.0, valid
```

### Mistake 3: Too Few GPU Layers
```bash
# ❌ WRONG: Only offloads 10 layers to GPU
llama-server -m model.gguf -ngl 10 -ts 0.5,0.5
# Result: Most of model still on CPU, slow!
```

```bash
# ✅ RIGHT: Offloads ALL layers
llama-server -m model.gguf -ngl 999 -ts 0.5,0.5
# Result: Entire model on GPUs, fast!
```

---

## How to Calculate VRAM Needs

### Formula
```
Total VRAM = Model Size + Context Memory + Overhead

Context Memory ≈ (context_size × 0.0001 × model_params_billions) GB
Overhead ≈ 500MB - 1GB
```

### Example: Qwen2.5-72B IQ3_XXS
```
Model: 30GB
Context (32k × 0.0001 × 72): ~10GB
Overhead: ~1GB
Total: ~41GB

Available: 48GB (2× 24GB GPUs)
Headroom: 7GB ✅ Safe!
```

### Example: Qwen2.5-72B Q4_K_M
```
Model: 42GB
Context (8k × 0.0001 × 72): ~2.5GB
Overhead: ~1GB
Total: ~45.5GB

Available: 48GB
Headroom: 2.5GB ✅ Tight but workable
```

---

## Monitoring Multi-GPU Usage

### Check VRAM in Real-Time
```bash
# Watch both GPUs update every 1 second
watch -n 1 /opt/rocm/bin/rocm-smi

# Output shows:
# GPU 0: 19.8GB / 24GB (83%)
# GPU 1: 20.2GB / 24GB (84%)
```

### Check GPU Utilization
```bash
# Look for "GPU Activity" in rocm-smi
/opt/rocm/bin/rocm-smi --showuse

# Should see both GPUs at 90-100% during inference
```

### Check Temperature
```bash
/opt/rocm/bin/rocm-smi --showtemp

# RX 7900 XTX safe temps: <90°C
# Typical under load: 70-85°C
```

---

## Troubleshooting Multi-GPU

### "Only GPU 0 is being used"
**Fix:** Add the tensor-split flag
```bash
--tensor-split 0.5,0.5
```

### "VRAM usage uneven"
**Normal!** Small differences (±10%) are fine due to:
- Context cache location
- Tensor size rounding
- Memory fragmentation

### "Second GPU not detected"
```bash
# Check ROCm sees both GPUs
/opt/rocm/bin/rocminfo | grep -i "marketing"

# Should see TWO entries:
# Marketing Name: AMD Radeon RX 7900 XTX
# Marketing Name: AMD Radeon RX 7900 XTX
```

**If only one GPU shows:**
1. Check PCIe connection
2. Check power cables
3. Check BIOS settings (Above 4G Decoding, Resizable BAR)
4. Reboot

---

## Advanced: Optimal Tensor Split

### When to Use Asymmetric Split

**Scenario 1: Different GPU Models**
```bash
# GPU 0: RX 7900 XTX (24GB)
# GPU 1: RX 6900 XT (16GB)
# Total: 40GB

# Give more work to bigger GPU
--tensor-split 0.6,0.4
```

**Scenario 2: One GPU Running Display**
```bash
# GPU 0: Running monitors (2GB used)
# GPU 1: Dedicated to AI (24GB free)

# Give more work to dedicated GPU
--tensor-split 0.4,0.6
```

**Scenario 3: Three Unequal GPUs**
```bash
# GPU 0: 24GB
# GPU 1: 16GB
# GPU 2: 12GB
# Total: 52GB

# Split proportionally
--tensor-split 0.46,0.31,0.23  # (24/52, 16/52, 12/52)
```

---

## Performance Impact

### Latency
- **Single GPU**: Baseline
- **Tensor split (2 GPUs)**: +5-10% latency (negligible)
- **Tensor split (4 GPUs)**: +15-20% latency

**Why:** GPUs must communicate and sync results (PCIe bandwidth)

### Throughput
- **Single GPU**: Baseline
- **Tensor split (2 GPUs)**: ~1.8× throughput (not quite 2×)
- **Tensor split (4 GPUs)**: ~3.2× throughput

**Why:** Communication overhead and synchronization

### Is It Worth It?
**Absolutely.** For big models:
- 5-10% slower ≫ than crashing from OOM
- 5-10% slower ≫ than 100× slower CPU inference

---

## The Bottom Line

**Tensor parallelism is how you run big models on consumer hardware.**

Key points:
1. ✅ Use `--tensor-split` to split model across GPUs
2. ✅ Ratios should sum to 1.0 (e.g., 0.5,0.5 or 0.33,0.33,0.34)
3. ✅ Use `-ngl 999` to offload all layers
4. ✅ Monitor VRAM with `rocm-smi`
5. ✅ Both GPUs should be 90%+ utilized

**With 2× RX 7900 XTX (48GB total), you can run:**
- 7B models: Easily (any quant)
- 32B models: Easily (Q4/Q5/Q8)
- 70B models: Yes! (Q4, IQ3)
- 110B models: No (need ~63GB)

---

**Questions?** Open an issue. Seriously, ask. This stuff is confusing.

**Working?** Star the repo and help someone else. Pay it forward.
