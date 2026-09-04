# OncoBridge AI — Troubleshooting Guide

This guide helps diagnose and fix common issues when running `python main.py`.

---

## Issue 1: "ModuleNotFoundError: No module named 'webview'"

**What happened:**
```
ModuleNotFoundError: No module named 'webview'
```

**Solution:**
This is expected and harmless. The app will automatically fall back to browser mode.

**To enable desktop window (optional):**
```bash
pip install pywebview
```

On Ubuntu, you may also need:
```bash
sudo apt-get install python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.0
```

**Note:** Browser mode works perfectly fine for the demo. Desktop window is purely cosmetic.

---

## Issue 2: Analysis Hangs — Progress Shows But No Results

**Symptoms:**
- Click "Run OncoBridge Analysis"
- Progress box shows: `[imaging_agent] Imaging agent: interpreting...`
- Waits for minutes with no results
- No error messages in UI

**Likely causes:**

### A. Gemma 4 model not loaded or path wrong

**Check your terminal output when you started `main.py`:**

**BAD (model not found):**
```
[INFO] Loading Gemma 4 26B-A4B from: ./models/gemma-4-26b-a4b
FileNotFoundError: Model folder validation failed:
  • Missing required file: config.json
```

**FIX:**
```bash
# Set correct path in config.yaml
nano configs/config.yaml

# Change this line to your actual Kaggle folder:
local_model_dir: "/full/path/to/gemma-4-26b-a4b"

# OR use environment variable:
export ONCOBRIDGE_MODEL_DIR=/full/path/to/gemma-4-26b-a4b
python main.py
```

### B. Model loaded but generation timing out

**Check terminal for:**
```
[ERROR] ImagingAgent: LLM generation timeout after 120s
```

**This means Gemma 4 is taking >2 minutes per generation.**

**Solutions:**

1. **Use INT4 precision (fastest):**
```bash
export ONCOBRIDGE_PRECISION=int4
python main.py
```

2. **Increase timeout (if you have slow CPU):**
Edit `agents/pipeline.py` line 75:
```python
result_container.append(str(result))
                except Exception as e:
                    error_container.append(e)
            
            thread = threading.Thread(target=generate_with_error_capture, daemon=True)
            thread.start()
            thread.join(timeout=timeout_seconds)  # ← change to 300 for 5min timeout
```

3. **Use demo mode to test UI without model:**
```bash
python main.py --no-llm
```
This should return results in 2-3 seconds.

### C. ChromaDB or RAG failing silently

**Check terminal for:**
```
[ERROR] Literature agent failed: ...
```

**FIX:**
```bash
# Rebuild knowledge base
rm -rf data/chroma_db
python -c "from rag.knowledge_base import KnowledgeBase; KnowledgeBase({'rag': {'chroma_dir': './data/chroma_db', 'embedding_model': 'BAAI/bge-m3', 'embedding_device': 'CPU', 'top_k': 5}})"
```

---

## Issue 3: Gemma 4 Loads But Generation Produces Garbage

**Symptoms:**
- Terminal shows: `✓ Gemma 4 26B-A4B loaded in 45s`
- Analysis completes
- Report shows random tokens or malformed text

**Likely cause:** Wrong model files or corrupted download

**FIX:**

1. **Validate your Kaggle download:**
```bash
python models/gemma_engine.py --model-dir /path/to/gemma-4-26b-a4b --check
```

Expected output:
```
Model folder: /path/to/gemma-4-26b-a4b
✓ Folder is valid
✓ Weight format: safetensors_sharded
✓ Shards found: 2
```

If you see errors, re-download from Kaggle.

2. **Test generation directly:**
```bash
python models/gemma_engine.py --model-dir /path/to/gemma-4-26b-a4b --test
```

You should see coherent clinical text. If not, the model weights are corrupted.

---

## Issue 4: Out of Memory (OOM) Error

**Symptoms:**
```
RuntimeError: [enforce fail at alloc_cpu.cpp:64] . DefaultCPUAllocator: can't allocate memory
```

**Your system doesn't have enough RAM.**

**Solutions in order of preference:**

1. **Force INT4 (uses ~15 GB RAM):**
```bash
export ONCOBRIDGE_PRECISION=int4
python main.py
```

2. **Close other applications:**
```bash
# Check RAM usage
free -h
# Kill memory hogs before running
```

3. **Use a smaller model (if you have <16 GB RAM):**

You'll need to download `gemma-4-4b-it` instead of 26B from Kaggle, then:
```bash
export ONCOBRIDGE_MODEL_DIR=/path/to/gemma-4-4b-it
python main.py
```

---

## Issue 5: "No Results" But No Errors

**Check the terminal output carefully.** The issue is logged there even if the UI doesn't show it.

**Common patterns:**

### Pattern A: Silent RAG failure
```
[WARNING] ChromaDB not available — using in-memory search
```
**FIX:** `pip install chromadb`

### Pattern B: Missing dependencies
```
[WARNING] PyRadiomics not available — using mock features
```
**FIX:** `pip install pyradiomics`

### Pattern C: Agent produced empty output
```
[WARNING] ImagingAgent: No result generated
```
**This means the LLM returned an empty string.**

**FIX:**
```bash
# Test the model directly
python models/gemma_engine.py --model-dir /path/to/model --test

# If this also returns nothing, your model files are corrupted
# Re-download from Kaggle
```

---

## Issue 6: Slow Performance (Each Analysis Takes >5 Minutes)

**Expected times:**
- **BF16 on 64 GB Xeon 6:** 30-60s total
- **INT8 on 32 GB Xeon 6:** 45-90s total
- **INT4 on 16 GB system:** 90-120s total

**If you're seeing >5 minutes:**

1. **Check you're not in demo mode:**
```bash
# Don't use --no-llm for real analysis
python main.py  # not: python main.py --no-llm
```

2. **Check precision:**
```bash
# In terminal output, look for:
[INFO] Precision: BF16 | Xeon 6 optimised  ← if this says BF16 but you only have 16 GB RAM, 
                                               you'll be swapping to disk (very slow)
```

**FIX:** Force appropriate precision:
```bash
export ONCOBRIDGE_PRECISION=int4  # for 16 GB RAM
python main.py
```

3. **Check CPU utilization:**
```bash
# In another terminal while analysis runs:
htop
```

If you see only 10-20% CPU usage on a many-core system, PyTorch isn't using all cores.

**FIX:**
```bash
export OMP_NUM_THREADS=32  # set to your physical core count
python main.py
```

---

## Issue 7: Results Show But Report Is Truncated

**Symptoms:**
- Imaging analysis shows: "PRIMARY FINDING..."
- Then cuts off mid-sentence

**Cause:** `max_new_tokens` too low

**FIX:**
Edit `configs/config.yaml`:
```yaml
model:
  max_new_tokens: 4096  # increase from 2048
```

---

## Debugging Commands

### Quick test without model (2 second startup):
```bash
python main.py --no-llm
```

### Test model loading only:
```bash
python models/gemma_engine.py --model-dir /path/to/model --benchmark
```

### Test full pipeline in terminal (no UI):
```bash
python test_demo.py
```

### View live logs:
```bash
python main.py 2>&1 | tee oncobridge.log
```

---

## Getting Help

If none of these solutions work:

1. **Capture full log:**
```bash
python main.py --no-llm 2>&1 | tee debug.log
# Run an analysis
# Send debug.log
```

2. **Capture system info:**
```bash
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().total/1e9:.1f} GB'); print(f'CPU cores: {psutil.cpu_count(logical=False)}')"
```

3. **Check versions:**
```bash
pip list | grep -E "(transformers|torch|gradio|openvino)"
```

---

## Known Issues

### Ubuntu 20.04 / Python 3.8
- `bitsandbytes` may not compile on Python 3.8
- **FIX:** Use Python 3.10+

### ARM64 (Apple Silicon, AWS Graviton)
- Intel-specific optimizations won't work
- **FIX:** Use `ONCOBRIDGE_PRECISION=bf16` (no bitsandbytes)

### Windows WSL1
- File I/O can be very slow
- **FIX:** Use WSL2 or native Windows Python

---

## Performance Benchmarks (For Comparison)

**Expected performance on Intel Xeon 6 (32 cores, 64 GB RAM):**

| Precision | RAM Used | Tokens/sec | Full Analysis Time |
|-----------|----------|------------|-------------------|
| BF16      | 52 GB    | 8-12 tok/s | 35-50s |
| INT8      | 27 GB    | 12-18 tok/s| 25-40s |
| INT4      | 16 GB    | 15-25 tok/s| 20-35s |

If your times are 3-5× slower, something is wrong (see Issue 6 above).
