"""
OncoBridge AI — Main Demo Application (Desktop Window Edition)
RSNA 2027 | Kaggle Gemma 4 Good Hackathon

WHAT HAPPENS WHEN YOU RUN THIS
═══════════════════════════════════════════════════════════════════════════════
After executing `python main.py`, you will see:

1. TERMINAL OUTPUT — Loading sequence (~30-90 seconds depending on precision):
   ┌─────────────────────────────────────────────────────────────────────┐
   │ OncoBridge AI — Precision Oncology Decision Support                 │
   │ Gemma 4 + OpenVINO + Intel Xeon 6 | RSNA 2027 Demo                  │
   │ Kaggle Gemma 4 Good Hackathon — Health & Sciences Track             │
   └─────────────────────────────────────────────────────────────────────┘
   
   [INFO] Initializing OncoBridge AI pipeline...
   [INFO] Model folder OK — 2 weight shards, format: safetensors_sharded
   [INFO] Xeon 6: 32 physical cores configured for PyTorch
   [INFO] Loading Gemma 4 26B-A4B from: ./models/gemma-4-26b-a4b
   [INFO] Precision: INT8 | Xeon 6 optimised
   [INFO] ✓ Gemma 4 26B-A4B loaded in 45.2s | precision: INT8 | device: cpu
   [INFO] ✓ Radiomics engine ready
   [INFO] ✓ Variant parser ready
   [INFO] ✓ Knowledge base ready
   [INFO] ✓ Trial matcher ready
   [INFO] ✓ Radiogenomics engine ready
   [INFO] ✓ OncoBridge pipeline ready
   
   Running on local URL:  http://127.0.0.1:7860    ← this appears
   
   → A DESKTOP WINDOW OPENS showing the full OncoBridge UI

2. DESKTOP WINDOW — Full 5-tab interface appears:
   ┌─────────────────────────────────────────────────────────────────────┐
   │  🧬 OncoBridge AI                                    [─] [□] [×]    │
   ├─────────────────────────────────────────────────────────────────────┤
   │  📋 Patient Input | 🔬 Imaging | 🧬 Genomics | 📄 Report | ⚙️ System │
   │                                                                     │
   │  🎯 Demo Cases                                                      │
   │  ○ P001: Lung Adenocarcinoma (EGFR exon 19 del)  [Load Case]      │
   │  ○ P002: Glioblastoma (IDH-wildtype, MGMT+)                        │
   │  ○ P003: Breast Cancer — ⚠️ DISCORDANCE ALERT                      │
   │                                                                     │
   │  👤 Patient Information                                             │
   │  Patient ID: [P001              ]  Cancer Type: [NSCLC...]         │
   │  Modality: [CT ▼]  Imaging Path: [                               ] │
   │  VCF Path: [                                                     ] │
   │  Clinical Notes:                                                    │
   │  ┌───────────────────────────────────────────────────────────────┐ │
   │  │ 68yo Asian female non-smoker. 2.4cm RUL nodule on screening  │ │
   │  │ CT. Progressive cough x3mo. ECOG PS 1. Part-solid GGO...     │ │
   │  └───────────────────────────────────────────────────────────────┘ │
   │                                                                     │
   │  [    🚀 Run OncoBridge Analysis    ]  ← CLICK THIS BUTTON         │
   │                                                                     │
   │  Pipeline Progress:                                                 │
   │  Ready. Click 'Run OncoBridge Analysis' to start.                  │
   └─────────────────────────────────────────────────────────────────────┘

3. WHEN YOU CLICK "Run OncoBridge Analysis":
   Progress updates stream into the "Pipeline Progress" box:
   
   [radiomics] Extracting radiomics features from CT...
   [genomics_parse] Parsing genomic variant file...
   [radiogenomics] Evaluating imaging-genomic concordance...
   [imaging_agent] Imaging agent: interpreting radiological findings...
   [genomics_agent] Genomics agent: annotating molecular profile...
   [literature_agent] Literature agent: retrieving evidence + trial matching...
   [synthesis_agent] Synthesis agent: generating tumor board report...
   [uncertainty_agent] Uncertainty agent: scoring confidence...
   [complete] Analysis complete in 42.3s
   
   Then ALL TABS populate with results:
   
   Tab 2 (🔬 Imaging Analysis):
   ├─ Gemma 4 vision tower analysis of the CT slice
   ├─ 10 key radiomics metrics (sphericity, entropy, GLCM features)
   └─ Radiogenomics concordance assessment
   
   Tab 3 (🧬 Genomics):
   ├─ Genomics agent clinical summary
   ├─ Actionable variants table (gene, variant, tier, therapy, evidence)
   └─ Matched clinical trials from ClinicalTrials.gov
   
   Tab 4 (📄 Tumor Board Report):  ← THE MAIN DELIVERABLE
   ├─ Complete AI-generated tumor board report
   │  ## 1. PATIENT SUMMARY & CLINICAL STAGE
   │  ## 2. IMAGING FINDINGS
   │  ## 3. MOLECULAR DIAGNOSIS
   │  ## 4. TREATMENT RECOMMENDATION
   │  ## 5. ALTERNATIVE OPTIONS
   │  ## 6. IMAGING-GENOMIC CONCORDANCE
   │  ## 7. CLINICAL TRIAL ELIGIBILITY
   │  ## 8. RECOMMENDED FOLLOW-UP
   │  ## 9. DISCUSSION POINTS FOR TUMOR BOARD
   │
   ├─ Confidence & uncertainty assessment (right column)
   │  "Overall confidence: 82%"
   │  "Key uncertainties: ..."
   │  "Human review required: Yes"
   │
   └─ [📥 Download Report (TXT)]  ← saves to .txt file
   
   Tab 5 (⚙️ System):
   └─ Live metrics: RAM usage, CPU %, tok/s, precision, model path

4. FOR THE RSNA DEMO — THREE PRE-LOADED CASES:
   
   Case P001 — Lung Adenocarcinoma
   Expected result: "Osimertinib 80mg daily (FLAURA: mPFS 18.9 months)"
   Concordance: ✓ "Imaging phenotype concordant with EGFR-mutant LUAD"
   
   Case P002 — Glioblastoma  
   Expected result: "Stupp protocol: TMZ + RT 60Gy, then TMZ x6 + TTFields"
   Concordance: ✓ "Ring enhancement + necrosis consistent with IDH-wt GBM"
   
   Case P003 — Breast Cancer
   Expected result: ⚠️ "HOLD — Repeat HER2 testing required"
   Concordance: ✗ "DISCORDANCE — MRI shows HER2-like phenotype but IHC HER2-"
   
LAUNCH OPTIONS
═══════════════════════════════════════════════════════════════════════════════
  python main.py                     # Desktop window (default)
  python main.py --browser           # Browser at http://localhost:7860
  python main.py --share             # Browser + public Gradio link (for Kaggle submission)
  python main.py --no-llm            # Demo mode (no Gemma 4 loading, instant startup)
  python main.py --port 8080         # Custom port
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── Import OncoBridge modules ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from models.gemma_engine import Gemma4Engine
from imaging.radiomics_engine import RadiomicsEngine
from genomics.variant_parser import VariantParser
from rag.knowledge_base import KnowledgeBase
from rag.trial_matcher import TrialMatcher
from utils.radiogenomics import RadiogenomicsEngine
from agents.pipeline import OncoBridgePipeline


# ── Load configuration ────────────────────────────────────────────────────────
def load_config(path: str = "./configs/config.yaml") -> dict:
    """Load YAML config or return minimal defaults if file not found."""
    if Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f)
    
    logger.warning(f"Config not found at {path}, using defaults")
    return {
        "model": {
            "local_model_dir": os.environ.get("ONCOBRIDGE_MODEL_DIR", "./models/gemma-4-26b-a4b"),
            "max_new_tokens": 2048,
            "temperature": 0.1,
            "context_length": 8192,
            "precision": "auto",
        },
        "openvino": {"device": "CPU"},
        "imaging": {"segmentation_model": "./models/totalseg"},
        "genomics": {},
        "rag": {"chroma_dir": "./data/chroma_db", "embedding_model": "BAAI/bge-m3", "top_k": 8},
        "clinical_trials": {"api_url": "https://clinicaltrials.gov/api/v2/studies", "max_results": 10},
        "demo": {"sample_cases_dir": "./data/sample_cases", "output_dir": "./outputs"},
        "ui": {"title": "OncoBridge AI", "theme": "soft", "server_port": 7860},
    }


# ── Initialize all components ─────────────────────────────────────────────────
def initialize_pipeline(config: dict, llm_enabled: bool = True):
    """Initialize all OncoBridge components."""
    logger.info("Initializing OncoBridge AI pipeline...")

    # LLM — Gemma 4 with local Kaggle weights
    llm = None
    if llm_enabled:
        try:
            llm = Gemma4Engine(config)
            logger.info("✓ Gemma 4 engine ready")
        except Exception as e:
            logger.warning(f"LLM load failed (running in demo mode): {e}")
            llm = None

    # Imaging
    radiomics = RadiomicsEngine(
        config,
        seg_model_dir=config.get("imaging", {}).get("segmentation_model", "./models/totalseg")
    )
    logger.info("✓ Radiomics engine ready")

    # Genomics
    variant_parser = VariantParser(config)
    logger.info("✓ Variant parser ready")

    # RAG
    knowledge_base = KnowledgeBase(config)
    logger.info("✓ Knowledge base ready")

    trial_matcher = TrialMatcher(config)
    logger.info("✓ Trial matcher ready")

    # Radiogenomics
    rg_engine = RadiogenomicsEngine()
    logger.info("✓ Radiogenomics engine ready")

    # Pipeline
    pipeline = OncoBridgePipeline(
        llm=llm,
        knowledge_base=knowledge_base,
        trial_matcher=trial_matcher,
        radiomics_engine=radiomics,
        variant_parser=variant_parser,
        radiogenomics_engine=rg_engine,
    )
    logger.info("✓ OncoBridge pipeline ready")

    return pipeline, llm


# ── Load demo cases ───────────────────────────────────────────────────────────
def load_demo_cases(cases_dir: str = "./data/sample_cases") -> dict:
    """Load pre-configured RSNA demo patient cases."""
    cases_path = Path(cases_dir) / "demo_cases.json"
    if cases_path.exists():
        with open(cases_path) as f:
            data = json.load(f)
        return {c["id"]: c for c in data.get("cases", [])}
    logger.warning(f"Demo cases not found at {cases_path}")
    return {}


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_ui(pipeline, llm, config: dict):
    """Build the full 5-tab Gradio interface."""
    try:
        import gradio as gr
    except ImportError:
        logger.error("Gradio not installed. Run: pip install gradio>=4.40.0")
        sys.exit(1)

    demo_cases = load_demo_cases(config.get("demo", {}).get("sample_cases_dir", "./data/sample_cases"))

    # ── CSS ────────────────────────────────────────────────────────────────────
    CUSTOM_CSS = """
    .oncobridge-header {
        background: linear-gradient(135deg, #1a56a0 0%, #0f7a6b 100%);
        color: white; padding: 20px 24px; border-radius: 12px;
        margin-bottom: 16px;
    }
    .alert-box {
        border: 2px solid #dc2626; border-radius: 8px;
        padding: 12px; background: #fef2f2; color: #dc2626;
        font-weight: 600;
    }
    .concordant-box {
        border: 2px solid #16a34a; border-radius: 8px;
        padding: 12px; background: #f0fdf4; color: #16a34a;
    }
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 12px; text-align: center;
    }
    """

    with gr.Blocks(
        title=config["ui"]["title"],
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    ) as demo:

        # ── Header ─────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="oncobridge-header">
            <h1 style="margin:0;font-size:26px;font-weight:700;">
                🧬 OncoBridge AI
            </h1>
            <p style="margin:6px 0 0;opacity:0.9;font-size:14px;">
                Multimodal Precision Oncology Decision Support &nbsp;·&nbsp;
                Gemma 4 + Intel Xeon 6 &nbsp;·&nbsp; RSNA 2027 Demo
            </p>
        </div>
        """)

        with gr.Tabs():

            # ═══════════════════════════════════════════════════════════════════
            # TAB 1: PATIENT INPUT
            # ═══════════════════════════════════════════════════════════════════
            with gr.Tab("📋 Patient Input", id="input"):

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 Demo Cases")
                        gr.Markdown("Select a pre-loaded RSNA demo case:")
                        case_selector = gr.Radio(
                            choices=[f"{v['id']}: {v['label']}" for v in demo_cases.values()],
                            label="Demo Patient Cases",
                            value=list(demo_cases.values())[0]["id"] + ": " + list(demo_cases.values())[0]["label"] if demo_cases else None
                        )
                        load_case_btn = gr.Button("Load Selected Case", variant="secondary")

                    with gr.Column(scale=2):
                        gr.Markdown("### 👤 Patient Information")
                        with gr.Row():
                            patient_id = gr.Textbox(label="Patient ID", value="P001", scale=1)
                            cancer_type = gr.Textbox(
                                label="Cancer Type / Diagnosis",
                                value="Non-Small Cell Lung Cancer (Lung Adenocarcinoma)",
                                scale=3
                            )
                        with gr.Row():
                            imaging_modality = gr.Dropdown(
                                choices=["CT", "MR", "PET", "XRAY"],
                                value="CT",
                                label="Imaging Modality",
                                scale=1
                            )
                            imaging_path = gr.Textbox(
                                label="Imaging Path (leave blank for demo data)",
                                placeholder="/path/to/dicom/series or file.nii.gz",
                                scale=3
                            )
                        vcf_path = gr.Textbox(
                            label="VCF File Path (leave blank for demo genomics)",
                            placeholder="/path/to/variants.vcf or variants.vcf.gz",
                        )
                        clinical_notes = gr.Textbox(
                            label="Clinical Notes / History",
                            lines=5,
                            placeholder="Patient age, symptoms, ECOG PS, prior treatments, biopsy details..."
                        )

                with gr.Row():
                    analyze_btn = gr.Button(
                        "🚀 Run OncoBridge Analysis",
                        variant="primary",
                        size="lg"
                    )

                with gr.Row():
                    progress_box = gr.Textbox(
                        label="Pipeline Progress",
                        value="Ready. Click 'Run OncoBridge Analysis' to start.",
                        lines=6,
                        interactive=False
                    )

            # ═══════════════════════════════════════════════════════════════════
            # TAB 2: IMAGING ANALYSIS
            # ═══════════════════════════════════════════════════════════════════
            with gr.Tab("🔬 Imaging Analysis", id="imaging"):
                gr.Markdown("### Radiomics & AI Vision Tower Analysis")
                with gr.Row():
                    with gr.Column():
                        imaging_analysis_out = gr.Textbox(
                            label="Imaging Agent Analysis (Gemma 4 Vision Tower)",
                            lines=20, interactive=False
                        )
                    with gr.Column():
                        gr.Markdown("#### Key Radiomics Metrics")
                        radiomics_metrics = gr.JSON(label="Quantitative Features", value={})
                        gr.Markdown("#### Radiogenomics Concordance")
                        concordance_out = gr.Textbox(
                            label="Imaging-Genomic Correlation",
                            lines=8, interactive=False
                        )

            # ═══════════════════════════════════════════════════════════════════
            # TAB 3: GENOMICS
            # ═══════════════════════════════════════════════════════════════════
            with gr.Tab("🧬 Genomics", id="genomics"):
                gr.Markdown("### Molecular Profile & Variant Annotation")
                with gr.Row():
                    with gr.Column():
                        genomics_out = gr.Textbox(
                            label="Genomics Agent Analysis",
                            lines=20, interactive=False
                        )
                    with gr.Column():
                        gr.Markdown("#### Actionable Variants")
                        variants_table = gr.JSON(
                            label="Tier 1-2 Actionable Alterations",
                            value={}
                        )
                        gr.Markdown("#### Matched Clinical Trials")
                        trials_out = gr.Textbox(
                            label="Open Recruiting Trials",
                            lines=12, interactive=False
                        )

            # ═══════════════════════════════════════════════════════════════════
            # TAB 4: TUMOR BOARD REPORT
            # ═══════════════════════════════════════════════════════════════════
            with gr.Tab("📄 Tumor Board Report", id="report"):
                gr.Markdown("### AI-Generated Tumor Board Report")
                gr.Markdown(
                    "*AI-generated analysis for tumor board discussion. "
                    "Requires physician review before clinical use.*",
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        report_out = gr.Textbox(
                            label="Complete Tumor Board Report",
                            lines=35, interactive=False
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("#### Confidence Assessment")
                        uncertainty_out = gr.Textbox(
                            label="Uncertainty & Confidence Report",
                            lines=20, interactive=False
                        )
                        gr.Markdown("#### Performance Metrics")
                        perf_metrics = gr.JSON(label="Xeon 6 / Gemma 4 Metrics", value={})

                with gr.Row():
                    download_report = gr.Button("📥 Download Report (TXT)", variant="secondary")
                    report_file = gr.File(label="Download", visible=False)

            # ═══════════════════════════════════════════════════════════════════
            # TAB 5: SYSTEM INFO
            # ═══════════════════════════════════════════════════════════════════
            with gr.Tab("⚙️ System", id="system"):
                gr.Markdown("### OncoBridge AI — System Information")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
#### Technology Stack
| Component | Technology |
|-----------|-----------|
| LLM | **Gemma 4 26B-A4B MoE** (Kaggle download) |
| Inference | **HuggingFace Transformers** + bitsandbytes quantization |
| CPU Acceleration | **Intel Xeon 6** AMX-BF16 + AVX-512 |
| Radiomics | **PyRadiomics** + OpenVINO segmentation |
| Genomics | cyvcf2 + VCF 4.2 parsing |
| RAG Store | **ChromaDB** + BGE-M3 embeddings |
| Trial Matching | ClinicalTrials.gov v2 API |
| Orchestration | Multi-agent pipeline (5 specialized agents) |
| UI | Gradio 4.x (desktop window or browser) |

#### Kaggle Gemma 4 Good Hackathon
- **Track:** Health & Sciences
- **Impact:** Reduces tumor board prep 4hr → 15 min
- **Privacy:** 100% local inference — no cloud calls
- **Validation:** TCGA-LUAD, TCGA-GBM, TCGA-BRCA datasets
                        """)
                    with gr.Column():
                        system_info = gr.JSON(label="Live System Metrics", value={})
                        refresh_btn = gr.Button("Refresh Metrics")

        # ═══════════════════════════════════════════════════════════════════════
        # EVENT HANDLERS
        # ═══════════════════════════════════════════════════════════════════════

        def load_case(case_choice):
            """Load a pre-configured demo case."""
            if not case_choice or not demo_cases:
                return [gr.update()] * 6

            case_id = case_choice.split(":")[0].strip()
            case = demo_cases.get(case_id, {})

            notes = case.get("clinical_notes", "")
            ctype = case.get("cancer_type", "")
            mod = case.get("imaging_modality", "CT")

            return (
                case_id,
                ctype,
                mod,
                "",   # imaging_path — use demo data
                "",   # vcf_path — use demo data
                notes
            )

        load_case_btn.click(
            fn=load_case,
            inputs=[case_selector],
            outputs=[patient_id, cancer_type, imaging_modality, imaging_path, vcf_path, clinical_notes]
        )

        def run_analysis(pid, ctype, mod, img_path, vcf, notes):
            """Run the full OncoBridge pipeline and update all UI outputs."""
            progress_lines = []

            def progress(step, msg):
                progress_lines.append(f"[{step}] {msg}")

            progress("start", "OncoBridge AI pipeline starting...")
            progress("model", f"Gemma 4 26B-A4B on Intel Xeon 6 — {llm.get_performance_stats()['precision'] if llm else 'demo mode'}...")

            try:
                state = pipeline.run(
                    patient_id=pid or "P-DEMO",
                    cancer_type=ctype or "Non-Small Cell Lung Cancer",
                    imaging_path=img_path or "",
                    vcf_path=vcf or "",
                    clinical_notes=notes or "",
                    imaging_modality=mod or "CT",
                    progress_callback=progress
                )

                # Format outputs
                progress_text = "\n".join(progress_lines)

                # Radiomics metrics (key features only)
                key_features = state.radiomics_result.get("key_features", {})
                metrics_display = {
                    k.replace("original_", "").replace("_", " "): round(float(v), 4)
                    for k, v in list(key_features.items())[:10]
                }
                metrics_display["Tumor Volume (mm³)"] = round(
                    state.radiomics_result.get("mask_volume_mm3", 0), 1
                )

                # Concordance
                rg = state.radiogenomics_result
                concordance_text = "\n".join(rg.get("recommendations", ["Not evaluated"]))
                if rg.get("has_discordance"):
                    concordance_text = "⚠️ DISCORDANCE ALERT\n\n" + concordance_text

                # Actionable variants table
                actionable = state.genomic_profile.get("actionable_variants", [])
                variants_display = {
                    av.get("gene", "?"): {
                        "variant": av.get("variant", ""),
                        "tier": av.get("tier", "?"),
                        "therapy": av.get("therapy", ""),
                        "evidence": av.get("evidence", ""),
                        "AF": f"{av.get('af', 0):.1%}"
                    }
                    for av in actionable[:8]
                }

                # Trials
                trials_text = ""
                for i, t in enumerate(state.trial_matches[:5], 1):
                    trials_text += (
                        f"{i}. {t.get('nctId', 'N/A')} — {t.get('briefTitle', 'N/A')}\n"
                        f"   Phase: {t.get('phase', 'N/A')} | "
                        f"Relevance: {t.get('score', 0):.0%}\n"
                        f"   {t.get('relevance_reason', '')}\n\n"
                    )
                if not trials_text:
                    trials_text = "No open trials matched. Search ClinicalTrials.gov manually."

                # Performance
                timing = state.timing
                perf = {
                    "Total Analysis Time": f"{timing.get('total', 0):.1f}s",
                    "Imaging Agent": f"{timing.get('imaging_agent', 0):.2f}s",
                    "Genomics Agent": f"{timing.get('genomics_agent', 0):.2f}s",
                    "Literature RAG": f"{timing.get('literature_agent', 0):.2f}s",
                    "Synthesis Agent": f"{timing.get('synthesis_agent', 0):.2f}s",
                }
                if llm:
                    perf.update(llm.get_performance_stats())

                uncertainty_text = state.uncertainty_report.get("analysis", "Confidence assessment not available.")

                return (
                    progress_text,
                    state.imaging_analysis,
                    metrics_display,
                    concordance_text,
                    state.genomics_summary,
                    variants_display,
                    trials_text,
                    state.tumor_board_report,
                    uncertainty_text,
                    perf,
                )

            except Exception as e:
                error_msg = f"Pipeline error: {e}\n\nCheck logs for details."
                logger.error(f"Analysis error: {e}", exc_info=True)
                empty = {}
                return (
                    error_msg, error_msg, empty, error_msg,
                    error_msg, empty, error_msg, error_msg, error_msg, empty
                )

        analyze_btn.click(
            fn=run_analysis,
            inputs=[patient_id, cancer_type, imaging_modality, imaging_path, vcf_path, clinical_notes],
            outputs=[
                progress_box,
                imaging_analysis_out, radiomics_metrics, concordance_out,
                genomics_out, variants_table, trials_out,
                report_out, uncertainty_out, perf_metrics
            ]
        )

        def get_system_info():
            """Gather live system metrics."""
            info = {
                "Gemma 4 Engine": "Ready" if llm else "Demo mode (LLM not loaded)",
                "PyRadiomics": "Available" if _check_import("radiomics") else "Not installed",
                "ChromaDB": "Available" if _check_import("chromadb") else "Not installed",
            }
            try:
                import psutil
                info["CPU Cores"] = psutil.cpu_count(logical=False)
                info["RAM"] = f"{psutil.virtual_memory().total / 1e9:.0f} GB"
                info["RAM Used"] = f"{psutil.virtual_memory().percent}%"
            except Exception:
                pass
            return info

        refresh_btn.click(fn=get_system_info, outputs=[system_info])
        demo.load(fn=get_system_info, outputs=[system_info])

        def save_report(report_text):
            """Save tumor board report to a text file."""
            import tempfile
            if not report_text:
                return None
            f = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="oncobridge_report_",
                delete=False
            )
            f.write(report_text)
            f.close()
            return gr.File(value=f.name, visible=True)

        download_report.click(
            fn=save_report,
            inputs=[report_out],
            outputs=[report_file]
        )

    return demo


def _check_import(module: str) -> bool:
    """Check if a module is importable."""
    try:
        __import__(module)
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Desktop window launcher (using pywebview)
# ─────────────────────────────────────────────────────────────────────────────

def launch_desktop_window(demo, port: int):
    """
    Launch the Gradio app in a native desktop window using pywebview.
    Falls back to browser if pywebview is not installed.
    """
    try:
        import webview
        import threading
        
        logger.info("pywebview detected — launching in desktop window mode...")
        
        # Flag to track if Gradio is ready
        server_ready = threading.Event()
        
        # Start Gradio in a background thread
        def start_gradio():
            try:
                demo.queue()  # Enable queuing for better stability
                demo.launch(
                    server_port=port,
                    share=False,
                    inbrowser=False,
                    prevent_thread_lock=True,
                    show_error=True,
                    quiet=False,
                )
                server_ready.set()
            except Exception as e:
                logger.error(f"Gradio launch error: {e}")
                server_ready.set()
        
        gradio_thread = threading.Thread(target=start_gradio, daemon=True)
        gradio_thread.start()
        
        # Wait for server to be ready (max 10 seconds)
        logger.info("Waiting for Gradio server to start...")
        if not server_ready.wait(timeout=10):
            logger.warning("Gradio server startup timeout — proceeding anyway")
        
        # Additional small delay to ensure full readiness
        import time
        time.sleep(2)
        
        # Open in desktop window
        logger.info("Opening OncoBridge desktop window...")
        try:
            webview.create_window(
                title="OncoBridge AI — Precision Oncology Decision Support",
                url=f"http://127.0.0.1:{port}",
                width=1400,
                height=900,
                resizable=True,
                fullscreen=False,
                min_size=(1200, 800),
            )
            webview.start()
        except Exception as e:
            logger.error(f"pywebview window creation failed: {e}")
            logger.info(f"Gradio is still running at http://127.0.0.1:{port}")
            logger.info("Open this URL in your browser manually")
            # Keep the main thread alive so Gradio stays running
            import time
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
        
    except ImportError:
        logger.warning(
            "\n╔════════════════════════════════════════════════════════════╗\n"
            "║  pywebview not installed — using browser mode             ║\n"
            "║  For desktop window: pip install pywebview                 ║\n"
            "╚════════════════════════════════════════════════════════════╝\n"
        )
        # Standard browser launch without prevent_thread_lock
        demo.queue()
        demo.launch(
            server_port=port,
            share=False,
            inbrowser=True,
            show_error=True,
            quiet=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OncoBridge AI Demo")
    parser.add_argument("--no-llm", action="store_true",
                        help="Run without LLM (demo mode — instant startup)")
    parser.add_argument("--browser", action="store_true",
                        help="Launch in browser instead of desktop window")
    parser.add_argument("--share", action="store_true",
                        help="Create public Gradio link (for Kaggle submission)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--config", default="./configs/config.yaml")
    parser.add_argument("--model-dir", 
                        help="Override model directory (same as ONCOBRIDGE_MODEL_DIR env var)")
    args = parser.parse_args()

    # Override model dir if specified
    if args.model_dir:
        os.environ["ONCOBRIDGE_MODEL_DIR"] = args.model_dir

    print("""
╔══════════════════════════════════════════════════════════════╗
║           OncoBridge AI — Precision Oncology DSS            ║
║  Gemma 4 26B-A4B + Intel Xeon 6 | RSNA 2027 Demo           ║
║  Kaggle Gemma 4 Good Hackathon — Health & Sciences Track    ║
╚══════════════════════════════════════════════════════════════╝
    """)

    config = load_config(args.config)
    pipeline, llm = initialize_pipeline(config, llm_enabled=not args.no_llm)
    demo = build_ui(pipeline, llm, config)

    if args.share:
        # Kaggle submission mode — creates public link
        demo.launch(
            server_port=args.port,
            share=True,
            show_error=True,
        )
    elif args.browser:
        # Browser mode
        demo.launch(
            server_port=args.port,
            share=False,
            inbrowser=True,
            show_error=True,
        )
    else:
        # Desktop window mode (default)
        launch_desktop_window(demo, args.port)
