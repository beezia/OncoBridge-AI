"""
OncoBridge AI — Quick validation test (no GPU, no model download required)
Tests all modules with mock data to verify the pipeline runs end-to-end.

Usage: python test_demo.py
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

print("=" * 60)
print("OncoBridge AI — Pipeline Validation Test")
print("=" * 60)

# ── 1. Config ──────────────────────────────────────────────────
print("\n[1] Loading configuration...")
import yaml
config_path = Path("./configs/config.yaml")
if config_path.exists():
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print("    ✓ Config loaded")
else:
    config = {"model": {"name": "google/gemma-4-27b-it", "openvino_model_dir": "./models/gemma4_ov",
                        "quantization": "INT4", "max_new_tokens": 512, "temperature": 0.1, "context_length": 8192},
              "openvino": {"device": "CPU", "cpu_threads": 0, "enable_amx": True},
              "imaging": {"segmentation_model": "./models/totalseg"},
              "genomics": {"clinvar_db": "", "tcga_signatures": ""},
              "rag": {"chroma_dir": "./data/chroma_db", "embedding_model": "BAAI/bge-m3",
                      "embedding_device": "CPU", "top_k": 5},
              "clinical_trials": {"api_url": "https://clinicaltrials.gov/api/v2/studies",
                                  "max_results": 5, "cache_ttl_hours": 24}}
    print("    ℹ Using default config (config.yaml not found)")

# ── 2. Radiomics ───────────────────────────────────────────────
print("\n[2] Testing Radiomics Engine...")
from imaging.radiomics_engine import RadiomicsEngine
radiomics = RadiomicsEngine(config, seg_model_dir="./models/totalseg_nonexistent")
result = radiomics._mock_features("CT")
assert "features" in result
assert "key_features" in result
assert "representative_slice" in result
print(f"    ✓ Radiomics: {len(result['features'])} features extracted (mock)")
print(f"    ✓ Tumor volume: {result['mask_volume_mm3']:.0f} mm³")
summary = radiomics.summarize_features(result, "CT")
print(f"    ✓ Summary: {summary[:80]}...")

# ── 3. Genomics ────────────────────────────────────────────────
print("\n[3] Testing Variant Parser...")
from genomics.variant_parser import VariantParser
vparser = VariantParser(config)
profile = vparser._demo_profile()
assert profile["tier1_count"] >= 1
assert profile["tmb_score"] > 0
print(f"    ✓ Variants: {profile['variant_count']} total, {profile['tier1_count']} Tier 1")
print(f"    ✓ TMB: {profile['tmb_score']} mut/Mb | MSI: {profile['msi_status']}")
print(f"    ✓ Actionable: {profile['actionable_variants'][0]['gene']} → {profile['actionable_variants'][0]['therapy']}")

# ── 4. RAG Knowledge Base ─────────────────────────────────────
print("\n[4] Testing Knowledge Base + RAG...")
from rag.knowledge_base import KnowledgeBase
kb = KnowledgeBase(config)
results = kb.retrieve("EGFR mutation osimertinib treatment NSCLC", top_k=3)
assert len(results) > 0
print(f"    ✓ Retrieved {len(results)} passages")
print(f"    ✓ Top result (score {results[0]['score']:.3f}): {results[0]['text'][:60]}...")

# ── 5. Clinical Trial Matcher ─────────────────────────────────
print("\n[5] Testing Clinical Trial Matcher...")
from rag.trial_matcher import TrialMatcher
tmatcher = TrialMatcher(config)
trials = tmatcher.find_trials(
    cancer_type="Non-Small Cell Lung Cancer",
    biomarkers=["EGFR exon 19 deletion"]
)
print(f"    ✓ Matched {len(trials)} trials")
if trials:
    print(f"    ✓ Top trial: {trials[0].get('nctId')} (score {trials[0].get('score', 0):.0%})")

# ── 6. Radiogenomics Engine ───────────────────────────────────
print("\n[6] Testing Radiogenomics Correlation Engine...")
from utils.radiogenomics import RadiogenomicsEngine
rg = RadiogenomicsEngine()
eval_result = rg.evaluate(
    imaging_features={"sphericity": 0.82, "glcm_homogeneity": 0.78, "glcm_contrast": 0.15},
    genomic_markers=["EGFR:exon19del", "TP53:R175H"],
    cancer_type="LUAD",
)
print(f"    ✓ Concordance score: {eval_result.get('overall_concordance_score', 0):.0%}")
print(f"    ✓ Discordance detected: {eval_result.get('has_discordance', False)}")
print(f"    ✓ Recommendation: {eval_result['recommendations'][0][:80]}...")

# ── 7. Full Pipeline (no LLM) ─────────────────────────────────
print("\n[7] Testing Full Pipeline (demo mode — no LLM)...")
from agents.pipeline import OncoBridgePipeline
from utils.radiogenomics import RadiogenomicsEngine

pipeline = OncoBridgePipeline(
    llm=None,              # No LLM in test mode
    knowledge_base=kb,
    trial_matcher=tmatcher,
    radiomics_engine=radiomics,
    variant_parser=vparser,
    radiogenomics_engine=rg,
)

state = pipeline.run(
    patient_id="TEST-001",
    cancer_type="Non-Small Cell Lung Cancer",
    imaging_path="",
    vcf_path="",
    clinical_notes="Test case — 65yo female, EGFR-mutant LUAD, stage IIIA",
    imaging_modality="CT",
)

assert state.patient_id == "TEST-001"
assert state.radiomics_result is not None
assert state.genomic_profile is not None
assert len(state.evidence_passages) > 0
assert len(state.trial_matches) > 0
print(f"    ✓ Pipeline completed in {state.timing.get('total', 0):.1f}s")
print(f"    ✓ Evidence passages: {len(state.evidence_passages)}")
print(f"    ✓ Trials matched: {len(state.trial_matches)}")
score = state.radiogenomics_result.get('overall_concordance_score') or 0
print(f"    ✓ Radiogenomics: concordance {score:.0%}")
print(f"    ✓ Errors: {state.errors or 'None'}")

# ── 8. Demo Cases ─────────────────────────────────────────────
print("\n[8] Validating demo patient cases...")
import json
cases_path = Path("./data/sample_cases/demo_cases.json")
if cases_path.exists():
    with open(cases_path) as f:
        cases_data = json.load(f)
    cases = cases_data.get("cases", [])
    print(f"    ✓ {len(cases)} demo cases loaded")
    for c in cases:
        print(f"    ✓ {c['id']}: {c['label'][:50]}")
else:
    print("    ℹ demo_cases.json not found (run from oncobridge/ directory)")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED — OncoBridge AI pipeline is functional")
print("=" * 60)
print("""
Next steps:
  1. Install dependencies:   pip install -r requirements.txt
  2. Download Gemma 4 model: python models/gemma_engine.py --download
  3. Launch demo:            python main.py
  4. For Kaggle submission:  python main.py --share
  
For RSNA demo (pre-loaded cases):
  python main.py --port 7860
  → Open browser: http://localhost:7860
""")
