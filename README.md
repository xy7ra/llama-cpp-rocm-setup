# 🚀 llama.cpp Multi-GPU Setup for AMD ROCm (RX 7900 XTX)

**Run 70B+ models locally on consumer AMD GPUs. No cloud. No subscriptions. Just you and your hardware.**

This repo documents a **proven working setup** for running large language models (up to 72B parameters) on 2× AMD Radeon RX 7900 XTX GPUs using llama.cpp with ROCm acceleration.

If you're tired of paying for API credits, want to keep your data private, or just want to tinker with AI on your own terms - this guide is for you.

---

## ⚠️ CRITICAL: PCIe Configuration Required

### 🚨 YOU MUST USE PCIe BIFURCATION ON CPU LANES 🚨

**Before you do ANYTHING else, configure your BIOS properly or this WILL NOT WORK.**

### The Problem
If your GPUs are connected asymmetrically (one on CPU lanes, one on chipset lanes), tensor parallelism will **fail spectacularly**. You'll get:
- Slow performance (10× slower than expected)
- Mysterious crashes
- GPU communication timeouts
- One GPU sitting idle while the other works

### The Solution
**Both GPUs MUST be on CPU PCIe lanes, NOT chipset lanes.**

### How to Fix This

**1. Check Your Motherboard Manual**
- Find which PCIe slots connect directly to CPU
- Usually: First 2 x16 slots = CPU lanes
- Lower slots = Chipset lanes (avoid these!)

**2. BIOS Settings (CRITICAL)**
```
Advanced → PCI Subsystem Settings
├── PCIe Bifurcation: Enabled
├── First x16 Slot: x8/x8 (or x8/x8/x8/x8)
├── Above 4G Decoding: Enabled
└── Resizable BAR: Enabled
```

**3. Physical Installation**
- GPU 0 in **first** x16 slot (runs at x8)
- GPU 1 in **second** x16 slot (runs at x8)
- Leave other slots empty (or use for non-GPU cards)

**4. Verify After Boot**
```bash
# Both should show x8 (not x16, not x4)
lspci | grep -i vga

# Should see something like:
# 03:00.0 VGA compatible controller: AMD ... [Radeon RX 7900 XTX] (rev c8)
# 07:00.0 VGA compatible controller: AMD ... [Radeon RX 7900 XTX] (rev c8)
```

### Why x8/x8 Instead of x16/x16?
- **x16 electrical**: Full 16 lanes to one GPU
- **x8/x8 bifurcation**: Splits 16 lanes → two GPUs get 8 lanes each
- PCIe 4.0 x8 = 128 GB/s bandwidth (plenty for LLM inference)
- Allows both GPUs on CPU lanes = symmetric, fast communication

### What Happens If You Don't Do This?
💀 **Pain.** Don't skip this step.

---

## 📊 What This Setup Can Do

| Model | Size | Speed | VRAM Used | Context |
|-------|------|-------|-----------|---------|
| **Qwen2.5-72B** (IQ3_XXS) | 30GB | 347 tok/s prompt, 15 tok/s gen | 40GB | 32k tokens |
| **Qwen2.5-Coder-7B** (Q4_K_M) | 4.4GB | 328 tok/s prompt, 89 tok/s gen | 4GB | 4k tokens |

**That's a 72 billion parameter model running locally, faster than you can read.**

---

## 💪 Why llama.cpp?

After trying vLLM, MLC-LLM, Ollama, and others, here's why llama.cpp won:

### ✅ What Works
- **Actually supports AMD consumer GPUs** (RX 7900 XTX) without hacks
- **Simple multi-GPU tensor parallelism** - one flag, it just works
- **Mature GGUF quantization** - proven, stable, excellent quality
- **Small binary, no Python bloat** - C++ executable you can understand
- **Active development** - bugs get fixed, new models get supported

### ❌ Why Not Others?

**vLLM**:
- "ROCm support" means MI200/MI300 data center cards only
- RX 7900 XTX treated as second-class citizen
- Flash attention broken on gfx1100
- Crashes, segfaults, mysterious errors

**MLC-LLM**:
- Compilation is a black art
- Architecture-specific binaries, no portability
- When it breaks, good luck debugging TVM

**Ollama**:
- Great for beginners, but limited control
- No tensor parallelism (duplicates model per GPU)
- Wastes VRAM, can't run 72B models

**llama.cpp**:
- Build it once, runs forever
- One config flag per feature
- When something breaks, you can actually fix it
- Community writes guides like this one

---

## 🎯 The Secret Sauce: Tensor Parallelism

**This is the key to running big models on consumer hardware.**

### What Most People Do Wrong (Model Duplication)
```bash
# ❌ WRONG: Ollama style - loads model on EACH GPU
# 30GB model × 2 GPUs = 60GB VRAM needed (won't fit!)
```

### What We Do Right (Tensor Splitting)
```bash
# ✅ RIGHT: Split tensors across GPUs
--tensor-split 0.5,0.5

# 30GB model ÷ 2 GPUs = 15GB per GPU (fits easily!)
```

**Tensor parallelism** means:
- **One model** split across multiple GPUs
- Both GPUs work on **each forward pass** together
- Not duplication - actual parallel processing
- 72B model fits in 48GB total VRAM (24GB × 2)

---

## 🛠️ Hardware Requirements

### Minimum Setup
- **GPU**: 1× AMD Radeon RX 7900 XTX (24GB VRAM)
- **CPU**: Any modern CPU (8+ cores recommended)
- **RAM**: 32GB system RAM
- **Storage**: 100GB free space for models
- **OS**: Ubuntu 22.04 or 24.04

### My Setup (Recommended for 70B+ models)
- **GPU**: 2× AMD Radeon RX 7900 XTX (48GB VRAM total)
- **CPU**: Ryzen/Threadripper (16+ cores)
- **RAM**: 64GB
- **Storage**: 500GB+ SSD for models
- **OS**: Ubuntu 24.04

### Can I Use One GPU?
**Yes!** You can run:
- 7B models easily (any quantization)
- 13B models comfortably (Q4/Q5)
- 32B models with tight VRAM (Q4)

---

## 📦 Quick Start

### 1. Install ROCm
```bash
# Ubuntu 22.04/24.04
sudo apt-get update
sudo apt-get install -y rocm-hip-sdk rocm-opencl-sdk rocm-smi-lib

# Add user to render group
sudo usermod -a -G render $USER

# Reboot required!
sudo reboot
```

### 2. Build llama.cpp
```bash
# Clone repo
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Create build directory
mkdir build && cd build

# Configure for AMD RX 7900 XTX (gfx1100)
cmake -DGGML_HIP=ON \
      -DAMDGPU_TARGETS=gfx1100 \
      -DCMAKE_BUILD_TYPE=Release \
      ..

# Build (use all cores)
cmake --build . -j$(nproc)

# Install binaries
sudo mkdir -p /opt/llama.cpp-rocm/bin
sudo install -m 0755 ./bin/llama-server /opt/llama.cpp-rocm/bin/
sudo install -m 0755 ./bin/llama-quantize /opt/llama.cpp-rocm/bin/
sudo install -m 0755 ./bin/llama-cli /opt/llama.cpp-rocm/bin/
```

### 3. Download a Model

**Option A: Pre-quantized GGUF (easiest)**
```bash
# Using huggingface-cli
pip install huggingface-hub

# Download Qwen2.5-72B (IQ3_XXS, 30GB)
huggingface-cli download mradermacher/Qwen2.5-72B-Instruct-GGUF \
  Qwen2.5-72B-Instruct.i1-IQ3_XXS.gguf \
  --local-dir ~/models/
```

**Option B: Quantize Yourself**
```bash
# Download HuggingFace model
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
  --local-dir ~/models/Qwen2.5-Coder-7B-Instruct

# Convert to F16 GGUF
python3 llama.cpp/convert_hf_to_gguf.py \
  ~/models/Qwen2.5-Coder-7B-Instruct \
  --outfile ~/models/qwen2.5-coder-7b-f16.gguf \
  --outtype f16

# Quantize to Q4_K_M
/opt/llama.cpp-rocm/bin/llama-quantize \
  ~/models/qwen2.5-coder-7b-f16.gguf \
  ~/models/qwen2.5-coder-7b-Q4_K_M.gguf \
  Q4_K_M
```

### 4. Run the Server

**Single GPU:**
```bash
/opt/llama.cpp-rocm/bin/llama-server \
  --model ~/models/qwen2.5-coder-7b-Q4_K_M.gguf \
  --n-gpu-layers 99 \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 4096
```

**Multi-GPU (2 GPUs):**
```bash
/opt/llama.cpp-rocm/bin/llama-server \
  --model ~/models/Qwen2.5-72B-Instruct.i1-IQ3_XXS.gguf \
  --n-gpu-layers 999 \
  --tensor-split 0.5,0.5 \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 32768
```

### 5. Test It!
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms"}],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

---

## 🧠 Understanding Quantization

**Quantization = Making models smaller without losing much quality**

### Quantization Types (Best to Worst)

| Type | Bits | 7B Size | 72B Size | Quality | When to Use |
|------|------|---------|----------|---------|-------------|
| **Q8_0** | 8.5 | 7.5GB | 75GB | ⭐⭐⭐⭐⭐ | Max quality, lots of VRAM |
| **Q5_K_M** | 5.5 | 5.2GB | 52GB | ⭐⭐⭐⭐ | Good balance |
| **Q4_K_M** | 4.5 | 4.4GB | 42GB | ⭐⭐⭐⭐ | **← Recommended** |
| **IQ4_XS** | 4.25 | 4.1GB | 40GB | ⭐⭐⭐⭐ | Excellent compromise |
| **IQ3_XXS** | 3.06 | 2.8GB | 30GB | ⭐⭐⭐ | Maximum compression |

### My Recommendation
- **First time?** Start with **Q4_K_M** - proven, reliable, great quality
- **Want bigger models?** Try **IQ4_XS** - nearly same quality, 10% smaller
- **VRAM constrained?** Use **IQ3_XXS** - still very usable

---

## 📚 Step-by-Step Guides

### [Convert HuggingFace Model to GGUF](docs/convert-hf-to-gguf.md)
How to take any HuggingFace model and make it work with llama.cpp

### [Quantization Guide](docs/quantization-guide.md)
Detailed explanation of quantization types and when to use each

### [Multi-GPU Setup](docs/multi-gpu-setup.md)
**The most important guide** - how tensor parallelism actually works

### [Troubleshooting](docs/troubleshooting.md)
Common issues and how to fix them

### [Performance Tuning](docs/performance-tuning.md)
Squeeze every last token/sec out of your hardware

---

## 🎓 Key Concepts Explained Simply

### What is GGUF?
- File format for quantized models
- Makes models smaller and faster
- Created by the llama.cpp project
- Widely supported, easy to use

### What is ROCm?
- AMD's answer to NVIDIA CUDA
- Lets AMD GPUs accelerate AI workloads
- Works great on RX 7900 XTX (with proper setup)
- Sometimes finicky, but worth it

### What is HIPBlas?
- ROCm's math library
- Makes matrix operations fast on AMD GPUs
- Required for good performance
- Automatically used when you build with `-DGGML_HIP=ON`

### Why gfx1100?
- Architecture code for RX 7900 XTX
- Tells ROCm exactly what GPU you have
- **Critical** - wrong architecture = slow or broken
- RX 7900 XT = also gfx1100, RX 6900 XT = gfx1030

---

## ⚙️ Advanced Topics

### Running Multiple Models
Use different ports for each model:
```bash
# Terminal 1: Coding model on port 8080
llama-server -m qwen-coder-7b.gguf --port 8080 -ts 0.5,0.5

# Terminal 2: Chat model on port 8081
llama-server -m qwen-chat-72b.gguf --port 8081 -ts 0.5,0.5
```

### Asymmetric GPU Split
If one GPU has more VRAM free:
```bash
# GPU 0 gets 60%, GPU 1 gets 40%
--tensor-split 0.6,0.4
```

### Temperature Monitoring
```bash
# Watch GPU temps in real-time
watch -n 1 /opt/rocm/bin/rocm-smi
```

### Background Server
```bash
# Run as systemd service (recommended for production)
# See docs/systemd-service.md
```

---

## 🐛 Common Issues

### "error: no GPUs detected"
```bash
# Fix: Check ROCm installation
/opt/rocm/bin/rocminfo | grep -i "marketing name"

# Should see: "Marketing Name: AMD Radeon RX 7900 XTX"
```

### "out of memory" errors
```bash
# Fix 1: Reduce context size
--ctx-size 2048  # Instead of 4096

# Fix 2: Use smaller quantization
# Q8 → Q5 → Q4 → IQ3

# Fix 3: Check VRAM usage
/opt/rocm/bin/rocm-smi
```

### Slow generation speed
```bash
# Check: Are all layers on GPU?
# Look for "offloaded: 80/80" (or similar) in server output

# Fix: Increase GPU layers
--n-gpu-layers 999  # Forces all layers to GPU
```

### MoE models fail
```bash
# Known issue: Mixtral and other MoE models don't work
# Reason: GGUF format changed, old files incompatible
# Solution: Use dense models (better for single-user anyway)
```

---

## 🌟 Real Talk: Why I Made This

I spent **weeks** fighting with vLLM, reading cryptic error messages, recompiling kernels, and banging my head against "unsupported architecture" errors.

The AI community loves to gatekeep. "Just use NVIDIA" they say. "AMD isn't supported" they claim.

**Bullshit.**

AMD cards work great. You just need someone to show you how.

I'm a parent. I work full time. I don't have unlimited hours to debug segfaults. This guide is what I wish existed when I started.

If this helped you, **pay it forward**:
- Help someone on Reddit/Discord who's stuck
- Write a blog post about your setup
- Contribute to open source projects
- Be kind to beginners

**The best engineers aren't the smartest - they're the ones who help others get better.**

---

## 🤝 Contributing

Found a mistake? Have a better way? **Please share!**

- Open an issue
- Submit a PR
- Join discussions

This guide gets better when we all contribute.

---

## 📜 License

**MIT License** - Do whatever you want with this. Copy it, remix it, sell it. I don't care.

Just help people. That's all I ask.

---

## 🙏 Acknowledgments

- **llama.cpp team** - For making this possible
- **AMD ROCm team** - For actually caring about consumer GPUs
- **mradermacher** - For GGUF quantizations that actually work
- **My family** - For tolerating the GPU fan noise
- **You** - For reading this far. Now go build something cool.

---

## 📞 Questions?

Open an issue. I'll help if I can.

But remember: **Google first, ask second.** The error message usually tells you what's wrong.

---

**Built with ❤️ and too much coffee**

**If this saved you time, spend that time with someone you love.**

**Be the change. Help others. Make the world a little bit better.**

---

*Last updated: 2025-10-26*
*Hardware: 2× AMD Radeon RX 7900 XTX, ROCm 6.4.0*
*Working llama.cpp commit: a80ff183*
