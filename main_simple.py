#!/usr/bin/env python3
"""
OncoBridge AI — Simplified Main UI (Working Version)
Precision Oncology Decision Support System
RSNA 2027 Demo | Kaggle Gemma 4 Good Hackathon
"""

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

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Import OncoBridge modules
from models.gemma_engine import Gemma4Engine
from imaging.radiomics_engine import RadiomicsEngine
from genomics.variant_parser import VariantParser
from rag.knowledge_base import KnowledgeBase
from rag.trial_matcher import TrialMatcher
from utils.radiogenomics import RadiogenomicsEngine
from agents.pipeline import OncoBridgePipeline


# ──────────────────────────────────────────────────────────────────────────────
# Configuration & Initialization
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "./configs/config.yaml") -> dict:
    """Load YAML config or return defaults."""
    if Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f)
    logger.warning(f"Config not found at {path}, using defaults")
    return {"model": {}, "rag": {}, "imaging": {}, "genomics": {}, "clinical_trials": {}, "demo": {}}


def initialize_pipeline(config: dict, llm_enabled: bool = True):
    """Initialize all OncoBridge components."""
    logger.info("Initializing OncoBridge AI pipeline...")

    # LLM — Gemma 4
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


def load_demo_cases(cases_dir: str = "./data/sample_cases") -> dict:
    """Load pre-configured demo cases."""
    cases_path = Path(cases_dir) / "demo_cases.json"
    if cases_path.exists():
        with open(cases_path) as f:
            data = json.load(f)
        return {c["id"]: c for c in data.get("cases", [])}
    logger.warning(f"Demo cases not found at {cases_path}")
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────────────────────────────────────

def build_ui(pipeline, llm, config: dict):
    """Build Gradio interface."""
    try:
        import gradio as gr
    except ImportError:
        logger.error("Gradio not installed. Run: pip install gradio>=4.40.0")
        sys.exit(1)

    demo_cases = load_demo_cases(config.get("demo", {}).get("sample_cases_dir", "./data/sample_cases"))

    # Custom CSS
    CUSTOM_CSS = """
    .header {
        background: linear-gradient(135deg, #1a56a0 0%, #0f7a6b 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .alert { border: 2px solid #dc2626; border-radius: 8px; padding: 12px; background: #fef2f2; color: #dc2626; }
    .success { border: 2px solid #16a34a; border-radius: 8px; padding: 12px; background: #f0fdf4; color: #16a34a; }
    """

    with gr.Blocks(title="OncoBridge AI", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:

        # Header
        gr.HTML("""
        <div class="header">
            <h1 style="margin:0;">🧬 OncoBridge AI</h1>
            <p style="margin:6px 0 0;opacity:0.9;font-size:14px;">
                Multimodal Precision Oncology Decision Support | Gemma 4 + Intel Xeon 6 | RSNA 2027
            </p>
        </div>
        """)

        with gr.Tabs():

            # ───────────────────────────────────────────────────────────────
            # TAB 1: Patient Input
            # ───────────────────────────────────────────────────────────────
            with gr.Tab("📋 Patient Input"):

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 Demo Cases")
                        case_selector = gr.Radio(
                            choices=[f"{v['id']}: {v['label']}" for v in demo_cases.values()] if demo_cases else ["No demo cases"],
                            label="Select Case",
                            value=list(demo_cases.values())[0]["id"] + ": " + list(demo_cases.values())[0]["label"] if demo_cases else "No demo cases"
                        )
                        load_case_btn = gr.Button("Load Selected Case", variant="secondary")

                    with gr.Column(scale=2):
                        gr.Markdown("### 👤 Patient Information")
                        with gr.Row():
                            patient_id = gr.Textbox(label="Patient ID", value="P001", scale=1)
                            cancer_type = gr.Textbox(label="Cancer Type", value="Non-Small Cell Lung Cancer", scale=2)
                        
                        imaging_modality = gr.Dropdown(
                            choices=["CT", "MR", "PET", "XRAY"],
                            value="CT",
                            label="Imaging Modality",
                            scale=1
                        )
                        
                        clinical_notes = gr.Textbox(
                            label="Clinical Notes",
                            lines=4,
                            placeholder="Patient age, symptoms, ECOG PS, prior treatments...",
                        )

                with gr.Row():
                    run_btn = gr.Button("🚀 Run OncoBridge Analysis", variant="primary", size="lg")

                with gr.Row():
                    progress_box = gr.Textbox(
                        label="Pipeline Progress",
                        value="Ready. Click 'Run OncoBridge Analysis' to start.",
                        lines=8,
                        interactive=False
                    )

            # ───────────────────────────────────────────────────────────────
            # TAB 2: Imaging Analysis
            # ───────────────────────────────────────────────────────────────
            with gr.Tab("🔬 Imaging Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        imaging_out = gr.Textbox(
                            label="Imaging Analysis",
                            lines=20,
                            interactive=False
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("#### Radiomics Metrics")
                        metrics_out = gr.JSON(label="Features", value={})

            # ───────────────────────────────────────────────────────────────
            # TAB 3: Genomics
            # ───────────────────────────────────────────────────────────────
            with gr.Tab("🧬 Genomics"):
                with gr.Row():
                    with gr.Column(scale=1):
                        genomics_out = gr.Textbox(
                            label="Genomics Summary",
                            lines=20,
                            interactive=False
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("#### Actionable Variants")
                        variants_out = gr.JSON(label="Tier 1-2 Alterations", value={})

            # ───────────────────────────────────────────────────────────────
            # TAB 4: Tumor Board Report
            # ───────────────────────────────────────────────────────────────
            with gr.Tab("📄 Tumor Board Report"):
                gr.Markdown("*AI-generated analysis for tumor board discussion. Requires physician review.*")
                with gr.Row():
                    with gr.Column(scale=2):
                        report_out = gr.Textbox(
                            label="Complete Tumor Board Report",
                            lines=40,
                            interactive=False
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("#### Confidence")
                        uncertainty_out = gr.Textbox(
                            label="Uncertainty Report",
                            lines=15,
                            interactive=False
                        )

            # ───────────────────────────────────────────────────────────────
            # TAB 5: System Info
            # ───────────────────────────────────────────────────────────────
            with gr.Tab("⚙️ System"):
                gr.Markdown("""
#### Technology Stack
- **LLM:** Gemma 4 26B-A4B (MoE)
- **Inference:** HuggingFace Transformers + Bitsandbytes
- **CPU:** Intel Xeon 6 with AVX-512 + AMX-BF16
- **Radiomics:** PyRadiomics (mock mode)
- **RAG:** ChromaDB + BGE-M3
- **Trial Matching:** ClinicalTrials.gov API
- **UI:** Gradio 4.x

#### Demo Cases
- **P001:** Lung Adenocarcinoma (EGFR exon 19 del)
- **P002:** Glioblastoma (IDH-wt, MGMT+)
- **P003:** Breast Cancer (HER2 discordance alert)
                """)
                perf_out = gr.JSON(label="Performance Metrics", value={})

        # ───────────────────────────────────────────────────────────────────
        # EVENT HANDLERS
        # ───────────────────────────────────────────────────────────────────

        def load_case(case_choice):
            """Load a demo case."""
            if not case_choice or not demo_cases:
                return [""] * 6
            
            case_id = case_choice.split(":")[0].strip()
            case = demo_cases.get(case_id, {})
            
            return (
                case_id,
                case.get("cancer_type", ""),
                case.get("imaging_modality", "CT"),
                case.get("clinical_notes", ""),
            )

        load_case_btn.click(
            fn=load_case,
            inputs=[case_selector],
            outputs=[patient_id, cancer_type, imaging_modality, clinical_notes]
        )

        def run_analysis(pid, ctype, mod, notes):
            """Run the full OncoBridge pipeline."""
            print("\n" + "="*80)
            print(f"[ANALYSIS START] Patient: {pid}, Cancer: {ctype}, Modality: {mod}")
            print("="*80, flush=True)
            
            progress_lines = []
            t0 = time.time()

            def progress(step, msg):
                line = f"[{step}] {msg}"
                progress_lines.append(line)
                print(f"[PROGRESS] {line}", flush=True)

            try:
                progress("init", "Starting OncoBridge analysis...")
                
                state = pipeline.run(
                    patient_id=pid or "P-DEMO",
                    cancer_type=ctype or "NSCLC",
                    imaging_path="",
                    vcf_path="",
                    clinical_notes=notes or "",
                    imaging_modality=mod or "CT",
                    progress_callback=progress
                )

                progress("complete", f"Analysis complete in {time.time()-t0:.1f}s")
                
                # Extract results
                rad = state.radiomics_result
                metrics = {k.replace("original_", "").replace("_", " "): round(float(v), 4)
                          for k, v in list(rad.get("key_features", {}).items())[:10]}
                
                rg = state.radiogenomics_result
                concordance = "\n".join(rg.get("recommendations", ["N/A"]))
                if rg.get("has_discordance"):
                    concordance = "⚠️ DISCORDANCE ALERT\n\n" + concordance
                
                actionable = {av.get("gene", "?"): {
                    "variant": av.get("variant", ""),
                    "tier": av.get("tier", "?"),
                    "therapy": av.get("therapy", "")
                } for av in state.genomic_profile.get("actionable_variants", [])[:5]}
                
                perf = {
                    "Total Time": f"{state.timing.get('total', 0):.1f}s",
                    "Imaging": f"{state.timing.get('imaging_agent', 0):.2f}s",
                    "Genomics": f"{state.timing.get('genomics_agent', 0):.2f}s",
                }
                if llm:
                    perf.update(llm.get_performance_stats() if hasattr(llm, 'get_performance_stats') else {})
                
                print("[ANALYSIS COMPLETE]", flush=True)
                
                return (
                    "\n".join(progress_lines),
                    state.imaging_analysis or "No imaging analysis",
                    metrics or {},
                    concordance or "N/A",
                    state.genomics_summary or "No genomics analysis",
                    actionable or {},
                    state.tumor_board_report or "No report generated",
                    state.uncertainty_report.get("analysis", "N/A") if hasattr(state.uncertainty_report, 'get') else "N/A",
                    perf or {},
                )

            except Exception as e:
                print(f"[ERROR] {e}", flush=True)
                import traceback
                traceback.print_exc()
                error = f"ERROR: {str(e)[:500]}"
                return (error,) * 9

        run_btn.click(
            fn=run_analysis,
            inputs=[patient_id, cancer_type, imaging_modality, clinical_notes],
            outputs=[
                progress_box,
                imaging_out, metrics_out,
                imaging_out,  # concordance placeholder
                genomics_out, variants_out,
                report_out, uncertainty_out,
                perf_out
            ]
        )

    return demo


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OncoBridge AI Demo")
    parser.add_argument("--no-llm", action="store_true", help="Run without LLM (demo mode)")
    parser.add_argument("--browser", action="store_true", help="Launch in browser")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--config", default="./configs/config.yaml")
    parser.add_argument("--model-dir", help="Override model directory")
    args = parser.parse_args()

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

    logger.info(f"Launching UI on http://127.0.0.1:{args.port}")
    demo.launch(server_port=args.port, share=False, inbrowser=not args.browser)
