# OncoBridge AI — Multimodal Precision Oncology Decision Support
### RSNA 2027 Demo |Potential Reference Application for HLS AI Suite

**Powered by:** Gemma 4 (26B MoE) · OpenVINO 2026.3 · Intel Xeon 6 · RAG · Radiogenomics

---

## What It Does

OncoBridge fuses CT/MR/PET/X-ray imaging data with genomic sequencing (VCF/CNV) to produce:
- Quantitative radiomics feature extraction (OpenVINO-accelerated)
- Genomic variant annotation and molecular subtyping
- Evidence-based recommendations via RAG over PubMed + NCCN guidelines
- Real-time clinical trial matching (ClinicalTrials.gov)
- Structured tumor board report in < 60 seconds
- Imaging-genomic discordance alerts

All inference runs 100% locally on Intel Xeon 6 — no cloud required.

---

## Hardware Requirements

- Intel Xeon 6 (Granite Rapids / Sierra Forest) or Intel Xeon 4th/5th Gen
- 64 GB+ RAM recommended for Gemma 4 26B MoE (INT4)
- 32 GB RAM minimum for Gemma 4 E4B fallback
- SSD with 40 GB free space

---

## Project Structure

```
oncobridge/
├── main.py                    # Entry point — Gradio demo UI
├── configs/
│   └── config.yaml            # All runtime configuration
├── agents/
│   ├── imaging_agent.py       # Radiomics interpretation agent
│   ├── genomics_agent.py      # Variant annotation agent
│   ├── literature_agent.py    # RAG retrieval agent
│   ├── synthesis_agent.py     # Final report generation agent
│   └── uncertainty_agent.py   # Confidence scoring agent
├── imaging/
│   ├── radiomics_engine.py    # PyRadiomics + OpenVINO segmentation
│   └── dicom_loader.py        # DICOM/NIfTI ingestion
├── genomics/
│   └── variant_parser.py      # VCF parsing + annotation
├── rag/
│   ├── knowledge_base.py      # ChromaDB vector store + embeddings
│   └── trial_matcher.py       # ClinicalTrials.gov API integration
├── models/
│   └── gemma_engine.py        # Gemma 4 + OpenVINO inference engine
├── utils/
│   ├── report_generator.py    # Structured tumor board report
│   └── radiogenomics.py       # Imaging-genomic correlation engine
└── data/
    └── sample_cases/          # Three de-identified demo patient cases
```

---

## Installation

```bash
# 1. Clone and set up environment
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download Gemma 4 model (requires HuggingFace token)
export HF_TOKEN=your_token_here
python models/gemma_engine.py --download

# 4. Build knowledge base (one-time, ~10 min)
python rag/knowledge_base.py --build

# 5. Launch demo
python main.py
```

---

---

## Demo Cases

| Case | Cancer Type | Imaging | Genomics | Key Finding |
|------|-------------|---------|----------|-------------|
| P001 | Lung Adenocarcinoma | CT Chest | EGFR exon 19 del | Imaging-genomic concordance — osimertinib candidate |
| P002 | Glioblastoma | MR Brain | IDH-wildtype, MGMT methylated | TMZ + RT — trial NCT match found |
| P003 | Breast Cancer | PET + MR | PIK3CA H1047R, HER2+ | Imaging-genomic discordance alert — re-biopsy recommended |
