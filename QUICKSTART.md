# OncoBridge AI — Quick Start Guide

Get OncoBridge running in 5 minutes.

---

## Prerequisites

- **OS:** Ubuntu 20.04+, macOS, or Windows WSL2
- **RAM:** 16 GB minimum, 32 GB+ recommended
- **CPU:** Intel Xeon 6 (ideal) or any modern x86-64 CPU with AVX2
- **Disk:** 40 GB free space (for model + dependencies)

---

## Step 1: Install Python Dependencies

```bash
cd oncobridge
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

**This will take 5-10 minutes** (downloads PyTorch, Transformers, Gradio, etc.)

---

## Step 2: Point to Your Kaggle Model

You should have already downloaded **gemma-4-26b-a4b** from Kaggle. Now tell OncoBridge where it is:

### Option A: Edit config file
```bash
nano configs/config.yaml
```

Find this line and change it to your actual path:
```yaml
local_model_dir: "./models/gemma-4-26b-a4b"   # ← EDIT THIS
```

### Option B: Use environment variable (no file editing)
```bash
export ONCOBRIDGE_MODEL_DIR=/full/path/to/gemma-4-26b-a4b
```

---

## Step 3: Validate Your Model Folder

Before loading the heavy model, check that your Kaggle download is valid:

```bash
python models/gemma_engine.py --model-dir /path/to/gemma-4-26b-a4b --check
```

**Expected output:**
```
============================================================
Model folder: /path/to/gemma-4-26b-a4b
============================================================
✓ Folder is valid
✓ Weight format: safetensors_sharded
✓ Shards found: 2
```

If you see errors, re-download the model from Kaggle.

---

## Step 4: Test Without Loading Model (Optional)

Test the UI instantly without waiting for Gemma 4 to load:

```bash
python main.py --no-llm
```

**What happens:**
- UI opens in browser at http://localhost:7860 (or desktop window if pywebview installed)
- Click "Run OncoBridge Analysis" 
- Results appear in 2-3 seconds (using mock LLM responses)
- All tabs populate with demo data

**This proves the UI works.** Now test with the real model:

---

## Step 5: Run With Real Model

```bash
python main.py
```

**What happens:**

1. **Model loading (30-90 seconds):**
```
╔══════════════════════════════════════════════════════════════╗
║           OncoBridge AI — Precision Oncology DSS            ║
╚══════════════════════════════════════════════════════════════╝

[INFO] Initializing OncoBridge AI pipeline...
[INFO] Model folder OK — 2 weight shards, format: safetensors_sharded
[INFO] Xeon 6: 32 physical cores configured for PyTorch
[INFO] Loading Gemma 4 26B-A4B from: /path/to/gemma-4-26b-a4b
[INFO] Precision: INT8 | Xeon 6 optimised  ← auto-selected based on your RAM
[INFO] ✓ Gemma 4 26B-A4B loaded in 45.2s | precision: INT8 | device: cpu
[INFO] ✓ Radiomics engine ready
[INFO] ✓ Knowledge base ready
[INFO] ✓ OncoBridge pipeline ready

Running on local URL:  http://127.0.0.1:7860
```

2. **UI opens:**
- Desktop window (if pywebview installed)
- OR browser at http://localhost:7860

3. **Demo case is pre-loaded:** Patient P001 (Lung Adenocarcinoma)

4. **Click "Run OncoBridge Analysis"**

5. **Watch progress updates (30-60 seconds):**
```
[radiomics] Extracting radiomics features from CT...
[genomics_parse] Parsing genomic variant file...
[imaging_agent] Imaging agent: interpreting radiological findings...
[genomics_agent] Genomics agent: annotating molecular profile...
[literature_agent] Literature agent: retrieving evidence...
[synthesis_agent] Synthesis agent: generating tumor board report...
[complete] Analysis complete in 42.3s
```

6. **Results appear in all 5 tabs**

---

## What You Should See (Expected Results)

### Tab 1: Patient Input
Shows the progress log and case details.

### Tab 2: Imaging Analysis
```
IMAGING ANALYSIS (CT)

PRIMARY FINDING
A 2.4 cm part-solid nodule in the right upper lobe with significant 
ground-glass opacity component (approximately 60% GGO)...

QUANTITATIVE RADIOMICS:
  firstorder Entropy: 5.234
  shape Sphericity: 0.82
  glcm Homogeneity: 0.78
  ...
```

### Tab 3: Genomics
```
MOLECULAR DIAGNOSIS
Primary driver: EGFR exon 19 deletion (p.Glu746_Ala750del)
Allele frequency: 42%

ACTIONABILITY ASSESSMENT
Tier 1 — FDA-approved targeted therapy:
• EGFR exon 19 deletion → Osimertinib 80mg daily
  Evidence: FLAURA trial (NEJM 2018)
  Outcome: mPFS 18.9 vs 10.2 months
  ...
```

### Tab 4: Tumor Board Report ⭐ (THE MAIN DELIVERABLE)
```
PATIENT SUMMARY & CLINICAL STAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Patient P001: 68-year-old Asian female, lifelong non-smoker
Clinical stage: IA3 (T2aN0M0)

IMAGING FINDINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CT Chest: 2.4 cm part-solid nodule, RUL, peripheral...

MOLECULAR DIAGNOSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EGFR exon 19 deletion (p.Glu746_Ala750del) — AF 42%
TP53 R175H mutation — AF 38%

TREATMENT RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIMARY RECOMMENDATION:
Surgical resection (lobectomy) followed by adjuvant 
osimertinib 80mg daily × 3 years

Rationale:
ADAURA trial: Adjuvant osimertinib reduced recurrence 
risk by 83% (3-year DFS: 90% vs 44%, HR 0.17)
Evidence level: 1A (NCCN Category 1)
...

[Full 9-section structured report continues]
```

### Tab 5: System
Shows live metrics (RAM usage, CPU %, model info).

---

## Try All Three Demo Cases

OncoBridge ships with 3 pre-configured RSNA demo cases:

### Case P001 — Lung Adenocarcinoma (EGFR+)
- Expected: "Osimertinib first-line"
- Concordance: ✓ Imaging phenotype matches EGFR mutation

### Case P002 — Glioblastoma (IDH-wildtype, MGMT+)
- Expected: "Stupp protocol: TMZ + RT + TTFields"
- Concordance: ✓ Ring enhancement consistent with IDH-wt GBM

### Case P003 — Breast Cancer (HER2 discordance)
- Expected: ⚠️ "HOLD — Repeat HER2 testing required"
- Concordance: ✗ **DISCORDANCE ALERT** — MRI shows HER2+ phenotype but IHC HER2-

**To switch cases:**
1. Select a different case from the radio buttons
2. Click "Load Selected Case"
3. Click "Run OncoBridge Analysis"

---

## Common First-Run Issues

### Issue: Analysis hangs forever
**Check terminal output.** If you see:
```
[ERROR] ImagingAgent: LLM generation timeout after 120s
```

Your CPU is too slow for BF16. **FIX:**
```bash
export ONCOBRIDGE_PRECISION=int4
python main.py
```

### Issue: Out of memory error
**Your system needs more RAM.** **FIX:**
```bash
export ONCOBRIDGE_PRECISION=int4  # reduces from 52 GB to 15 GB
python main.py
```

### Issue: Model folder errors
**FIX:**
```bash
# Verify folder contents
ls /path/to/gemma-4-26b-a4b

# Should see:
# config.json
# tokenizer.json
# model-00001-of-00002.safetensors
# model-00002-of-00002.safetensors
# (and other files)

# If missing files, re-download from Kaggle
```

### Issue: Browser doesn't open
**Manual access:**
Open your browser and go to: `http://127.0.0.1:7860`

---

## Next Steps

### For RSNA Demo
- Test all 3 demo cases
- Practice explaining the imaging-genomic discordance alert (Case P003)
- Note the performance metrics in Tab 5 to quote during demo

### For Kaggle Submission
```bash
python main.py --share
```
This creates a public Gradio link you can submit to Kaggle.

### For Real Patient Data (TCGA Validation)
See `VALIDATION.md` for instructions on:
- Downloading TCGA-LUAD imaging + genomics
- Running OncoBridge on real paired data
- Computing validation metrics (AUC, precision, recall)

---

## Performance Tuning

### If analysis is too slow (>2 minutes):
```bash
export ONCOBRIDGE_PRECISION=int4
export OMP_NUM_THREADS=32  # set to your physical core count
python main.py
```

### If you want highest quality (and have 64 GB RAM):
```bash
export ONCOBRIDGE_PRECISION=bf16
python main.py
```

### If you want to see verbose debugging:
```bash
python main.py 2>&1 | tee oncobridge.log
```

---

## Stopping the App

- **Desktop window:** Click the X button
- **Browser mode:** Press Ctrl+C in the terminal
- **Graceful shutdown:** The app saves no state, so you can just kill it

---

## File Locations

After running, you'll find:
- **Downloaded reports:** `~/Downloads/oncobridge_report_*.txt`
- **Logs:** Terminal output (redirect to file if needed)
- **Data:** `data/chroma_db` (RAG vector store)
- **Outputs:** No persistent outputs (everything is in-memory)

---

## Getting Help

If something doesn't work:
1. Check `TROUBLESHOOTING.md`
2. Run `python test_demo.py` to diagnose
3. Capture full log: `python main.py 2>&1 | tee debug.log`

---

## You're Ready!

You now have a working OncoBridge demo that:
- ✅ Loads Gemma 4 26B-A4B from your Kaggle download
- ✅ Runs full 5-agent pipeline (Imaging, Genomics, Literature, Synthesis, Uncertainty)
- ✅ Generates tumor board reports in 30-60 seconds
- ✅ Demonstrates imaging-genomic concordance checking
- ✅ Matches clinical trials in real-time
- ✅ Runs 100% locally on your Xeon 6 (no cloud calls)

Perfect for RSNA 2027 demo and Kaggle Gemma 4 Good Hackathon submission!
