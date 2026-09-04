"""
OncoBridge AI — Multi-Agent Orchestrator
Five specialized agents powered by Gemma 4 with native function calling,
orchestrated via LangGraph-style sequential pipeline with shared state.

Agents:
  1. ImagingAgent    — Interprets radiomics + vision tower analysis
  2. GenomicsAgent   — Annotates variants and molecular subtype
  3. LiteratureAgent — RAG retrieval over PubMed + NCCN
  4. SynthesisAgent  — Integrates all inputs → clinical report
  5. UncertaintyAgent — Scores confidence and flags gaps
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Generator
import signal

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared Pipeline State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    """Shared state passed through the agent pipeline."""
    patient_id: str = ""
    cancer_type: str = ""

    # Input data
    imaging_path: str = ""
    vcf_path: str = ""
    clinical_notes: str = ""
    imaging_modality: str = "CT"

    # Agent outputs
    radiomics_result: dict = field(default_factory=dict)
    imaging_analysis: str = ""
    genomic_profile: dict = field(default_factory=dict)
    genomics_summary: str = ""
    evidence_passages: list = field(default_factory=list)
    literature_summary: str = ""
    trial_matches: list = field(default_factory=list)
    radiogenomics_result: dict = field(default_factory=dict)
    synthesis: str = ""
    uncertainty_report: dict = field(default_factory=dict)

    # Final report
    tumor_board_report: str = ""
    treatment_recommendation: str = ""

    # Metadata
    timing: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Base Agent
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent:
    def __init__(self, name: str, llm, system_prompt: str):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt

    def run(self, state: PipelineState) -> PipelineState:
        raise NotImplementedError

    def _generate(self, prompt: str, stream: bool = True, timeout_seconds: int = 120) -> str:
        """
        Generate text with timeout protection.
        
        Args:
            prompt: Input prompt
            stream: Whether to stream tokens
            timeout_seconds: Max seconds to wait for generation (default: 120)
            
        Returns:
            Generated text or error message
        """
        if self.llm is None:
            logger.debug(f"{self.name}: Using mock response (LLM not loaded)")
            return self._mock_response(prompt)
        
        try:
            logger.debug(f"{self.name}: Calling LLM with {len(prompt)} char prompt")
            
            # Use threading to implement timeout
            import threading
            result_container = []
            error_container = []
            
            def generate_with_error_capture():
                try:
                    result = self.llm.generate(
                        prompt=prompt,
                        system_prompt=self.system_prompt,
                        stream=stream
                    )
                    if hasattr(result, "__iter__") and not isinstance(result, str):
                        result_container.append("".join(result))
                    else:
                        result_container.append(str(result))
                except Exception as e:
                    error_container.append(e)
            
            thread = threading.Thread(target=generate_with_error_capture, daemon=True)
            thread.start()
            thread.join(timeout=6000)     #100minutes
            
            if thread.is_alive():
                logger.error(f"{self.name}: LLM generation timeout after {timeout_seconds}s")
                return f"[{self.name} TIMEOUT — generation exceeded {timeout_seconds}s. Check Gemma 4 engine logs.]"
            
            if error_container:
                raise error_container[0]
            
            if result_container:
                logger.debug(f"{self.name}: Generated {len(result_container[0])} chars")
                return result_container[0]
            
            logger.warning(f"{self.name}: No result generated")
            return f"[{self.name} — no output generated]"
            
        except Exception as e:
            logger.error(f"{self.name}: Generation error: {e}", exc_info=True)
            return f"[{self.name} ERROR: {str(e)[:200]}]"

    def _mock_response(self, prompt: str) -> str:
        """Mock response when LLM unavailable (for testing)."""
        return f"[{self.name} mock response — LLM not loaded]"


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1: Imaging Agent
# ─────────────────────────────────────────────────────────────────────────────

class ImagingAgent(BaseAgent):
    """
    Interprets radiomics features and vision tower image analysis.
    Uses Gemma 4 multimodal to analyze the representative CT/MR slice.
    """

    SYSTEM = (
        "You are an expert radiologist AI. Analyze radiomics features and imaging findings "
        "with precise clinical language. Structure your output as:\n"
        "TUMOR CHARACTERISTICS | LOCATION | MORPHOLOGY | TEXTURE | CLINICAL SIGNIFICANCE\n"
        "Be specific with measurements and quantitative descriptors. Flag any features "
        "relevant to molecular subtype prediction."
    )

    def __init__(self, llm):
        super().__init__("ImagingAgent", llm, self.SYSTEM)

    def run(self, state: PipelineState) -> PipelineState:
        t0 = time.time()
        logger.info("ImagingAgent: Analyzing imaging data...")

        rad = state.radiomics_result
        print(f"Radiomics state: {rad}")
        print(f"Radiomics image slice: {rad.get('representative_slice')}")
        features = rad.get("features", {})
        key_features = rad.get("key_features", {})
        modality = state.imaging_modality

        # Build prompt with radiomics features
        feature_str = "\n".join([
            f"  {k}: {v:.4f}" for k, v in list(key_features.items())[:12]
        ])

        prompt = f"""You are analyzing a {modality} scan for a patient with suspected {state.cancer_type}.

QUANTITATIVE RADIOMICS FEATURES:
{feature_str}

TUMOR GEOMETRY:
  Maximum 3D diameter: {features.get('original_shape_Maximum3DDiameter', 'N/A'):.1f} mm
  Volume: {rad.get('mask_volume_mm3', 0):.0f} mm³
  Sphericity: {key_features.get('original_shape_Sphericity', 0):.3f}
  Elongation: {key_features.get('original_shape_Elongation', 0):.3f}

TEXTURE ANALYSIS:
  GLCM Entropy (heterogeneity): {key_features.get('original_firstorder_Entropy', 0):.3f}
  GLCM Homogeneity: {key_features.get('original_glcm_Homogeneity', 0):.3f}
  GLCM Contrast: {key_features.get('original_glcm_Contrast', 0):.3f}

CLINICAL CONTEXT:
{state.clinical_notes or 'Not provided'}

Based on these quantitative features:
1. Describe the tumor characteristics in radiological terms
2. Identify features suggestive of specific molecular subtypes
3. Flag any imaging features requiring clinical attention
4. Estimate likelihood of malignancy and aggressiveness

Provide a structured radiological interpretation suitable for a tumor board report."""

        # If image slice available, use vision tower
        rep_slice = rad.get("representative_slice")
        if rep_slice is not None and self.llm is not None:
            try:
                t_vision = time.time()
                imaging_analysis = self.llm.analyze_medical_image(
                    rep_slice, modality, state.clinical_notes
                )
                logger.info(f"Vision analysis took {time.time()-t_vision:.2f}s within Imaging vision call"
                )
                # Augment with radiomics
                t_text = time.time()
                detailed = self._generate(prompt)
                logger.info(f"Radiomics interpretation took {time.time()-t_text:.2f}s within Imaging text call")
                state.imaging_analysis = f"{imaging_analysis}\n\nQUANTITATIVE RADIOMICS:\n{detailed}"
            except Exception as e:
                logger.warning(f"Vision analysis failed: {e}")
                state.imaging_analysis = self._generate(prompt)
        else:
            state.imaging_analysis = self._generate(prompt)

        state.timing["imaging_agent"] = round(time.time() - t0, 2)
        logger.info(f"ImagingAgent: Done in {state.timing['imaging_agent']}s")
        return state


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2: Genomics Agent
# ─────────────────────────────────────────────────────────────────────────────

class GenomicsAgent(BaseAgent):
    """
    Interprets genomic variant profile with actionability assessment.
    Uses Gemma 4 function calling to annotate variants.
    """

    SYSTEM = (
        "You are a molecular oncologist AI specializing in genomic variant interpretation. "
        "Analyze variants for: pathogenicity, actionability (tier 1-4), targeted therapy "
        "implications, clinical trial eligibility, and prognostic significance. "
        "Always cite evidence level (FDA-approved, NCCN, investigational)."
    )

    def __init__(self, llm):
        super().__init__("GenomicsAgent", llm, self.SYSTEM)

    def run(self, state: PipelineState) -> PipelineState:
        t0 = time.time()
        logger.info("GenomicsAgent: Interpreting genomic profile...")

        profile = state.genomic_profile
        actionable = profile.get("actionable_variants", [])
        tmb = profile.get("tmb_score", 0)
        msi = profile.get("msi_status", "Unknown")

        actionable_str = ""
        for av in actionable[:8]:
            actionable_str += (
                f"  • {av['gene']} {av.get('variant', '')} (AF: {av.get('af', 0):.1%}) — "
                f"Tier {av.get('tier', '?')} — {av.get('therapy', 'N/A')} "
                f"({av.get('evidence', 'N/A')})\n"
            )

        variants_str = ""
        for v in profile.get("variants", [])[:10]:
            variants_str += (
                f"  {v.get('gene', '?')}: {v.get('hgvsp', v.get('alt', '?'))} "
                f"(AF: {v.get('af', 0):.1%}, {v.get('clinvar', 'Unknown')})\n"
            )

        prompt = f"""Genomic profile for patient with {state.cancer_type}:

TOTAL VARIANTS: {profile.get('variant_count', 0)} (somatic: {profile.get('somatic_variant_count', 0)})
TMB: {tmb} mut/Mb | MSI: {msi}

ACTIONABLE VARIANTS (Tier 1-2):
{actionable_str or '  None identified'}

ALL DETECTED VARIANTS (top 10):
{variants_str or '  No variants provided'}

MUTATED GENES: {', '.join(profile.get('all_genes', [])[:20])}

Please provide:
1. MOLECULAR DIAGNOSIS: Primary driver alteration and molecular subtype
2. ACTIONABILITY ASSESSMENT: Tier 1 therapies and evidence level
3. IMMUNOTHERAPY BIOMARKERS: TMB, MSI, PD-L1 implications
4. PROGNOSTIC IMPLICATIONS: Expected clinical course
5. COMBINATION THERAPY CONSIDERATIONS: Rational combinations
6. RESISTANCE MECHANISMS: Known resistance to identified therapies
7. GERMLINE CONSIDERATIONS: Any variants requiring germline testing

Format as a structured genomics report for tumor board presentation."""

        state.genomics_summary = self._generate(prompt)
        state.timing["genomics_agent"] = round(time.time() - t0, 2)
        logger.info(f"GenomicsAgent: Done in {state.timing['genomics_agent']}s")
        return state


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3: Literature Agent
# ─────────────────────────────────────────────────────────────────────────────

class LiteratureAgent(BaseAgent):
    """
    RAG retrieval agent. Queries knowledge base for supporting evidence
    and synthesizes into a clinical evidence summary.
    """

    SYSTEM = (
        "You are a clinical evidence synthesizer. Given retrieved evidence passages from "
        "PubMed and NCCN guidelines, synthesize a concise evidence-based recommendation. "
        "Always include: evidence level, key trial names, survival outcomes, and guideline status. "
        "Format citations as [PMID:XXXXXXX] or [NCCN guideline name]."
    )

    def __init__(self, llm, knowledge_base, trial_matcher):
        super().__init__("LiteratureAgent", llm, self.SYSTEM)
        self.kb = knowledge_base
        self.trial_matcher = trial_matcher

    def run(self, state: PipelineState) -> PipelineState:
        t0 = time.time()
        logger.info("LiteratureAgent: Retrieving evidence and matching trials...")

        # Build retrieval query from patient profile
        actionable = state.genomic_profile.get("actionable_variants", [])
        query_parts = [state.cancer_type]
        biomarkers = []

        for av in actionable[:3]:
            gene = av.get("gene", "")
            variant = av.get("variant", "")
            query_parts.append(f"{gene} {variant} treatment")
            biomarkers.append(f"{gene} {variant}")

        if not biomarkers:
            biomarkers = [state.cancer_type]

        query = " AND ".join(query_parts)

        # Retrieve evidence
        passages = self.kb.retrieve(
            query=query,
            cancer_type=state.cancer_type.upper()[:4] if len(state.cancer_type) >= 3 else None,
            top_k=8
        )
        state.evidence_passages = passages
        context = self.kb.format_context(passages)

        # Match clinical trials
        def timeout_handler(signum, frame):
            raise TimeoutError("Trial matchning timed out")
        
        #signal.signal(signal.SIGALRM, timeout_handler)
        #signal.alarm(10) # 10 sec timeout

        try:
            state.trial_matches = self.trial_matcher.find_trials(
            cancer_type=state.cancer_type,
            biomarkers=biomarkers
            )
        except TimeoutError:
            logger.warning("Trial matching timeout - skipping")
            state.trial_matches = []
        #finally:
            #signal.alarm(0)  # cancel alarm

        trials_str = self.trial_matcher.format_for_report(state.trial_matches)

        prompt = f"""You are synthesizing clinical evidence for a patient with {state.cancer_type}.

RETRIEVED EVIDENCE:
{context}

CLINICAL TRIAL MATCHES:
{trials_str}

PATIENT BIOMARKERS: {', '.join(biomarkers) or 'Not specified'}
GENOMIC SUMMARY: {state.genomics_summary[:500]}

Based on the retrieved evidence:
1. GUIDELINE RECOMMENDATION: Current NCCN/ESMO guideline-based first-line recommendation
2. KEY TRIAL EVIDENCE: Most relevant trials supporting the recommendation (include outcomes)
3. ALTERNATIVE OPTIONS: Second-line or alternative therapy options
4. TRIAL ELIGIBILITY: Summary of matched open trials
5. EVIDENCE GAPS: What evidence is missing or uncertain

Provide a concise, evidence-graded recommendation with citations."""

        state.literature_summary = self._generate(prompt)
        state.timing["literature_agent"] = round(time.time() - t0, 2)
        logger.info(f"LiteratureAgent: Done in {state.timing['literature_agent']}s")
        return state


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4: Synthesis Agent
# ─────────────────────────────────────────────────────────────────────────────

class SynthesisAgent(BaseAgent):
    """
    Final integration agent. Combines all agent outputs into a
    structured tumor board report with treatment recommendations.
    """

    SYSTEM = (
        "You are a precision oncology AI assistant generating a comprehensive tumor board report. "
        "Integrate imaging, genomic, and literature findings into a clear, actionable report. "
        "Use clinical language appropriate for oncologists and radiologists. "
        "Structure the report for tumor board presentation. "
        "Be definitive in recommendations where evidence supports; flag uncertainty where it exists."
    )

    def __init__(self, llm):
        super().__init__("SynthesisAgent", llm, self.SYSTEM)

    def run(self, state: PipelineState) -> PipelineState:
        t0 = time.time()
        logger.info("SynthesisAgent: Generating tumor board report...")

        rad_summary = state.radiomics_result.get("key_features", {})
        rg = state.radiogenomics_result

        discordance_alert = ""
        if rg.get("has_discordance"):
            discordances = rg.get("discordances", [])
            if discordances:
                discordance_alert = (
                    f"\n⚠️ IMAGING-GENOMIC DISCORDANCE DETECTED:\n"
                    + "\n".join(d.get("message", "") for d in discordances)
                    + "\n"
                )

        prompt = f"""Generate a comprehensive tumor board report for Patient {state.patient_id}.

═══════════════════════════════════════════════════════════
PATIENT SUMMARY
═══════════════════════════════════════════════════════════
Patient ID: {state.patient_id}
Primary Diagnosis: {state.cancer_type}
Imaging Modality: {state.imaging_modality}
Clinical Notes: {state.clinical_notes or 'Not provided'}

═══════════════════════════════════════════════════════════
IMAGING ANALYSIS ({state.imaging_modality})
═══════════════════════════════════════════════════════════
{state.imaging_analysis}

═══════════════════════════════════════════════════════════
GENOMIC PROFILE
═══════════════════════════════════════════════════════════
{state.genomics_summary}

═══════════════════════════════════════════════════════════
RADIOGENOMICS CORRELATION
═══════════════════════════════════════════════════════════
{state.radiogenomics_result.get('recommendations', ['N/A'])[0] if state.radiogenomics_result else 'Not evaluated'}
{discordance_alert}

═══════════════════════════════════════════════════════════
EVIDENCE-BASED RECOMMENDATIONS
═══════════════════════════════════════════════════════════
{state.literature_summary}

═══════════════════════════════════════════════════════════

Please generate a COMPLETE TUMOR BOARD REPORT with these sections:

## 1. PATIENT SUMMARY & CLINICAL STAGE
## 2. IMAGING FINDINGS (structured radiological report)
## 3. MOLECULAR DIAGNOSIS
## 4. TREATMENT RECOMMENDATION (primary, with rationale and evidence level)
## 5. ALTERNATIVE OPTIONS (second-line, clinical trial)
## 6. IMAGING-GENOMIC CONCORDANCE ASSESSMENT
## 7. CLINICAL TRIAL ELIGIBILITY
## 8. RECOMMENDED FOLLOW-UP
## 9. DISCUSSION POINTS FOR TUMOR BOARD

Format as a professional clinical document. Include confidence level for each recommendation."""

        state.synthesis = self._generate(prompt)
        state.tumor_board_report = state.synthesis

        # Extract primary treatment recommendation
        lines = state.synthesis.split("\n")
        for i, line in enumerate(lines):
            if "TREATMENT RECOMMENDATION" in line.upper() or "PRIMARY RECOMMENDATION" in line.upper():
                state.treatment_recommendation = "\n".join(lines[i:i+8]).strip()
                break

        if not state.treatment_recommendation:
            state.treatment_recommendation = state.synthesis[:500]

        state.timing["synthesis_agent"] = round(time.time() - t0, 2)
        logger.info(f"SynthesisAgent: Done in {state.timing['synthesis_agent']}s")
        return state


# ─────────────────────────────────────────────────────────────────────────────
# Agent 5: Uncertainty Agent
# ─────────────────────────────────────────────────────────────────────────────

class UncertaintyAgent(BaseAgent):
    """
    Quantifies confidence in each recommendation and identifies
    evidence gaps, data quality issues, and clinical uncertainties.
    """

    SYSTEM = (
        "You are a clinical AI safety specialist. Assess the confidence, evidence quality, "
        "and potential risks of AI-generated clinical recommendations. "
        "Be conservative — flag any uncertainty that could affect patient safety. "
        "Never overstate confidence in AI recommendations."
    )

    def __init__(self, llm):
        super().__init__("UncertaintyAgent", llm, self.SYSTEM)

    def run(self, state: PipelineState) -> PipelineState:
        t0 = time.time()
        logger.info("UncertaintyAgent: Assessing confidence and evidence quality...")

        # Calculate data completeness
        completeness = self._assess_completeness(state)

        prompt = f"""Assess the confidence and limitations of the following AI-generated oncology analysis.

ANALYSIS SUMMARY:
{state.synthesis[:1500]}

DATA COMPLETENESS ASSESSMENT:
{json.dumps(completeness, indent=2)}

GENOMIC DATA QUALITY:
  Variant count: {state.genomic_profile.get('variant_count', 0)}
  Tier 1 actionable variants: {state.genomic_profile.get('tier1_count', 0)}
  TMB: {state.genomic_profile.get('tmb_score', 0)} mut/Mb

IMAGING DATA QUALITY:
  Modality: {state.imaging_modality}
  Volume: {state.radiomics_result.get('mask_volume_mm3', 0):.0f} mm³
  Features extracted: {len(state.radiomics_result.get('features', {}))}

RADIOGENOMICS CONCORDANCE: {state.radiogenomics_result.get('overall_concordance_score', 'N/A')}
DISCORDANCE DETECTED: {state.radiogenomics_result.get('has_discordance', False)}

Please assess:
1. OVERALL CONFIDENCE: Rate 0-100% with justification
2. DATA QUALITY FLAGS: Missing or low-quality data affecting reliability
3. RECOMMENDATION CONFIDENCE: Rate each major recommendation 0-100%
4. KEY UNCERTAINTIES: Top 3 uncertainties that should be presented at tumor board
5. REQUIRED ADDITIONAL WORKUP: Tests needed to increase confidence
6. AI LIMITATIONS: Specific limitations of AI analysis in this case
7. HUMAN OVERSIGHT REQUIRED: Areas where human expert review is essential

This analysis is AI-generated and requires physician review before clinical use."""

        analysis = self._generate(prompt)

        state.uncertainty_report = {
            "analysis": analysis,
            "completeness": completeness,
            "overall_confidence": self._extract_confidence(analysis),
            "discordance_alert": state.radiogenomics_result.get("has_discordance", False),
            "data_quality_score": self._compute_data_quality(state),
            "requires_human_review": True,  # Always true for clinical AI
        }

        state.timing["uncertainty_agent"] = round(time.time() - t0, 2)
        state.timing["total"] = round(sum(v for v in state.timing.values() if isinstance(v, float)), 2)
        logger.info(f"UncertaintyAgent: Done in {state.timing['uncertainty_agent']}s")
        return state

    def _assess_completeness(self, state: PipelineState) -> dict:
        return {
            "has_imaging": bool(state.imaging_path or state.radiomics_result),
            "has_genomics": bool(state.vcf_path or state.genomic_profile),
            "has_clinical_notes": bool(state.clinical_notes),
            "has_pathology": False,  # Not yet integrated
            "has_prior_imaging": False,
            "imaging_features_count": len(state.radiomics_result.get("features", {})),
            "actionable_variant_count": state.genomic_profile.get("tier1_count", 0),
            "evidence_passages_retrieved": len(state.evidence_passages),
            "trial_matches_found": len(state.trial_matches),
        }

    def _compute_data_quality(self, state: PipelineState) -> float:
        """Compute 0-1 data quality score based on completeness."""
        score = 0.0
        if state.radiomics_result and state.radiomics_result.get("features"):
            score += 0.35
        if state.genomic_profile and state.genomic_profile.get("variant_count", 0) > 0:
            score += 0.35
        if state.clinical_notes:
            score += 0.15
        if state.evidence_passages:
            score += 0.15
        return round(score, 2)

    def _extract_confidence(self, analysis: str) -> int:
        """Extract overall confidence percentage from analysis text."""
        import re
        matches = re.findall(r"(\d{1,3})\s*%", analysis)
        if matches:
            try:
                scores = [int(m) for m in matches if 0 <= int(m) <= 100]
                if scores:
                    return int(sum(scores[:3]) / len(scores[:3]))
            except Exception:
                pass
        return 70  # Default


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class OncoBridgePipeline:
    """
    Full multi-agent pipeline for precision oncology analysis.
    Orchestrates all five agents with shared state.
    """

    def __init__(self, llm, knowledge_base, trial_matcher, radiomics_engine, variant_parser, radiogenomics_engine):
        self.llm = llm
        self.radiomics = radiomics_engine
        self.variant_parser = variant_parser
        self.radiogenomics = radiogenomics_engine

        self.imaging_agent = ImagingAgent(llm)
        self.genomics_agent = GenomicsAgent(llm)
        self.literature_agent = LiteratureAgent(llm, knowledge_base, trial_matcher)
        self.synthesis_agent = SynthesisAgent(llm)
        self.uncertainty_agent = UncertaintyAgent(llm)

    def run(
        self,
        patient_id: str,
        cancer_type: str,
        imaging_path: str = "",
        vcf_path: str = "",
        clinical_notes: str = "",
        imaging_modality: str = "CT",
        progress_callback=None,
    ):
        """
        Run the full OncoBridge pipeline.

        Args:
            patient_id: Patient identifier
            cancer_type: Primary cancer diagnosis
            imaging_path: Path to DICOM/NIfTI imaging data
            vcf_path: Path to VCF genomic file
            clinical_notes: Free-text clinical context
            imaging_modality: CT, MR, PET, XRAY
            progress_callback: Optional fn(step, message) for UI updates

        Returns:
            Completed PipelineState with full tumor board report
        """

        print(f"Inputs \n======================== \nPatient ID: {patient_id} \nCancer Type: {cancer_type} \nImaging Path: {imaging_path} \nVCF Path: {vcf_path} \nClinical Notes: {clinical_notes} \nImaging Modality: {imaging_modality} \n========================")

        def progress(step: str, msg: str):
            logger.info(f"[{step}] {msg}")
            if progress_callback:
                progress_callback(step, msg)

        state = PipelineState(
            patient_id=patient_id,
            cancer_type=cancer_type,
            imaging_path=imaging_path,
            vcf_path=vcf_path,
            clinical_notes=clinical_notes,
            imaging_modality=imaging_modality,
        )

        try:
            # Step 1: Extract radiomics features
            progress("radiomics", f"Extracting radiomics features from {imaging_modality}...")
            if imaging_path:
                state.radiomics_result = self.radiomics.process_dicom(imaging_path, imaging_modality)
            else:
                state.radiomics_result = self.radiomics._mock_features(imaging_modality)

            # Step 2: Parse genomic variants
            progress("genomics_parse", "Parsing genomic variant file...")
            if vcf_path:
                state.genomic_profile = self.variant_parser.parse_vcf(vcf_path)
            else:
                state.genomic_profile = self.variant_parser._demo_profile()

            # Step 3: Radiogenomics correlation
            progress("radiogenomics", "Evaluating imaging-genomic concordance...")
            genomic_markers = [
                av.get("key", "") for av in state.genomic_profile.get("actionable_variants", [])
            ]
            if not genomic_markers:
                genomic_markers = state.genomic_profile.get("all_genes", [])[:3]
            state.radiogenomics_result = self.radiogenomics.evaluate(
                imaging_features={
                    "sphericity": state.radiomics_result.get("key_features", {}).get("original_shape_Sphericity", 0.5),
                    "entropy": state.radiomics_result.get("key_features", {}).get("original_firstorder_Entropy", 5.0),
                    "glcm_homogeneity": state.radiomics_result.get("key_features", {}).get("original_glcm_Homogeneity", 0.7),
                    "glcm_contrast": state.radiomics_result.get("key_features", {}).get("original_glcm_Contrast", 0.2),
                },
                genomic_markers=genomic_markers,
                cancer_type=cancer_type.upper()[:4] if len(cancer_type) >= 3 else cancer_type,
            )

            # Step 4: Agent pipeline (with error isolation)
            progress("imaging_agent", "Imaging agent: interpreting radiological findings...")
            try:
                state = self.imaging_agent.run(state)
            except Exception as e:
                logger.error(f"Imaging agent failed: {e}", exc_info=True)
                state.imaging_analysis = f"[Imaging agent error: {str(e)[:200]}]"
                state.errors.append(f"imaging_agent: {e}")

            yield state

            progress("genomics_agent", "Genomics agent: annotating molecular profile...")

            try:
                state = self.genomics_agent.run(state)
            except Exception as e:
                logger.error(f"Genomics agent failed: {e}", exc_info=True)
                state.genomics_summary = f"[Genomics agent error: {str(e)[:200]}]"
                state.errors.append(f"genomics_agent: {e}")

            yield state

            progress("literature_agent", "Literature agent: retrieving evidence + trial matching...")

            try:
                state = self.literature_agent.run(state)
            except Exception as e:
                logger.error(f"Literature agent failed: {e}", exc_info=True)
                state.literature_summary = f"[Literature agent error: {str(e)[:200]}]"
                state.errors.append(f"literature_agent: {e}")

            yield state

            progress("synthesis_agent", "Synthesis agent: generating tumor board report...")

            try:
                state = self.synthesis_agent.run(state)
            except Exception as e:
                logger.error(f"Synthesis agent failed: {e}", exc_info=True)
                state.synthesis = f"[Synthesis agent error: {str(e)[:200]}]"
                state.tumor_board_report = (
                    f"Synthesis agent encountered an error: {e}\n\n"
                    "Partial analysis from earlier agents:\n\n"
                    f"Imaging: {state.imaging_analysis[:500] if state.imaging_analysis else 'N/A'}\n\n"
                    f"Genomics: {state.genomics_summary[:500] if state.genomics_summary else 'N/A'}\n\n"
                    f"Literature: {state.literature_summary[:500] if state.literature_summary else 'N/A'}"
                )
                state.errors.append(f"synthesis_agent: {e}")

            yield state

            progress("uncertainty_agent", "Uncertainty agent: scoring confidence...")

            try:
                state = self.uncertainty_agent.run(state)
            except Exception as e:
                logger.error(f"Uncertainty agent failed: {e}", exc_info=True)
                state.uncertainty_report = {"analysis": f"[Uncertainty agent error: {str(e)[:200]}]"}
                state.errors.append(f"uncertainty_agent: {e}")

            yield state

            progress("complete", f"Analysis complete in {state.timing.get('total', 0):.1f}s")

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            state.errors.append(str(e))
            if not state.tumor_board_report:
                state.tumor_board_report = (
                    f"Pipeline error: {e}\n\n"
                    "Partial analysis may be available in individual agent outputs."
                )

        return
