"""
OncoBridge AI — Gemma 4 26B-A4B Engine
Tuned specifically for the locally-downloaded Kaggle model folder.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT YOU DOWNLOADED FROM KAGGLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  kaggle.com/models/google/gemma-4  →  gemma-4-26b-a4b variant

Your local folder should contain these files:
  gemma-4-26b-a4b/
  ├── config.json                   ← model architecture definition
  ├── generation_config.json        ← default sampling settings
  ├── processor_config.json         ← vision processor config
  ├── tokenizer.json                ← 262K-token vocabulary (32.2 MB)
  ├── tokenizer_config.json         ← chat template (Gemma 4 format)
  ├── chat_template.jinja           ← Jinja2 chat template
  ├── model-00001-of-00014.safetensors   ┐
  ├── model-00002-of-00014.safetensors   │  sharded weights
  ├── ...                                │  (14 shards × ~1.8 GB each)
  └── model-00014-of-00014.safetensors   ┘
  └── model.safetensors.index.json  ← shard routing index

WHY 26B-A4B IS THE RIGHT CHOICE FOR THIS APPLICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • MoE architecture: 25.2B total params, only 3.8B ACTIVE per token
    → runs at near 4B speed while delivering ~13B quality
  • 256K context window → fits an entire patient record + imaging
    report + genomics + guidelines in ONE context
  • Native function calling → clean agent tool dispatch
  • Vision tower (550M params) → processes CT/MR slices directly
  • 82.6% MMLU Pro, 82.4% MATH-Vision — highest benchmark scores
    among open models at this inference cost
  • MedXPertQA MM: 58.1% → proven multimodal medical reasoning

MEMORY REQUIREMENTS (Xeon 6)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BF16 (full precision):   ~52 GB RAM   ← ideal on 64 GB Xeon 6
  INT8 (bitsandbytes):     ~27 GB RAM   ← comfortable on 32 GB
  INT4 (bitsandbytes):     ~15 GB RAM   ← minimum viable

QUICK START
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # 1. Install dependencies
  pip install transformers>=4.51.0 accelerate>=1.0.0 torch>=2.3.0
  pip install bitsandbytes>=0.45.0   # for INT4/INT8 quantization
  pip install Pillow numpy

  # 2. Point to your local folder and run
  python models/gemma_engine.py --model-dir /path/to/gemma-4-26b-a4b --test

  # 3. Run full OncoBridge demo
  python main.py --model-dir /path/to/gemma-4-26b-a4b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Generator, Iterator, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional OpenVINO acceleration for Xeon 6 ─────────────────────────────────
try:
    import openvino as ov
    import openvino_genai as ov_genai
    OV_AVAILABLE = True
except ImportError:
    OV_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Model folder validation
# ─────────────────────────────────────────────────────────────────────────────

# Files that MUST be present in the Kaggle download folder
REQUIRED_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
]

# At least one of these weight patterns must match
WEIGHT_PATTERNS = [
    "model-00001-of-*.safetensors",  # sharded safetensors (Kaggle default)
    "model.safetensors",             # single-file safetensors
    "pytorch_model.bin",             # legacy PyTorch
]


def validate_model_folder(model_dir: str) -> dict:
    """
    Inspect the downloaded Kaggle model folder and report what was found.
    Returns a dict with keys: valid, issues, weight_type, shard_count.
    """
    path = Path(model_dir)
    issues = []
    info = {"valid": False, "issues": [], "weight_type": None, "shard_count": 0}

    if not path.exists():
        info["issues"] = [f"Folder not found: {model_dir}"]
        return info

    # Check required config files
    for fname in REQUIRED_FILES:
        if not (path / fname).exists():
            issues.append(f"Missing required file: {fname}")

    # Detect weight format
    safetensors_shards = sorted(path.glob("model-*.safetensors"))
    single_safetensor  = path / "model.safetensors"
    pytorch_bin        = path / "pytorch_model.bin"

    if safetensors_shards:
        info["weight_type"] = "safetensors_sharded"
        info["shard_count"] = len(safetensors_shards)
    elif single_safetensor.exists():
        info["weight_type"] = "safetensors_single"
        info["shard_count"] = 1
    elif pytorch_bin.exists():
        info["weight_type"] = "pytorch_bin"
        info["shard_count"] = 1
    else:
        issues.append(
            "No model weight files found. Expected model-00001-of-XXXXX.safetensors "
            "or model.safetensors. The Kaggle download may be incomplete."
        )

    info["issues"] = issues
    info["valid"] = len(issues) == 0
    return info


def print_folder_check(model_dir: str):
    """Print a friendly report of what's in the model folder."""
    info = validate_model_folder(model_dir)
    print(f"\n{'='*60}")
    print(f"Model folder: {model_dir}")
    print(f"{'='*60}")
    if info["valid"]:
        print(f"✓ Folder is valid")
        print(f"✓ Weight format: {info['weight_type']}")
        print(f"✓ Shards found: {info['shard_count']}")
    else:
        print("✗ Folder issues detected:")
        for issue in info["issues"]:
            print(f"  → {issue}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Intel Xeon 6 — PyTorch CPU optimisations
# ─────────────────────────────────────────────────────────────────────────────

def configure_xeon6_torch():
    """
    Apply Intel Xeon 6 optimisations to PyTorch:
      - Use all physical cores (no hyperthreading waste)
      - Enable BF16 for AMX (Advanced Matrix Extensions)
      - Enable IPEX if available for further Xeon speedups
    """
    try:
        import torch
        import psutil

        num_physical_cores = psutil.cpu_count(logical=False) or 16
        torch.set_num_threads(num_physical_cores)
        torch.set_num_interop_threads(max(1, num_physical_cores // 4))

        # AMX-BF16 is the key Xeon 6 feature — massively speeds matrix ops
        if torch.cpu.is_available() and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        logger.info(f"Xeon 6: {num_physical_cores} physical cores configured for PyTorch")

        # Intel Extension for PyTorch (optional but recommended on Xeon 6)
        try:
            import intel_extension_for_pytorch as ipex
            logger.info("✓ Intel Extension for PyTorch (IPEX) active")
        except ImportError:
            logger.debug("IPEX not installed — using standard PyTorch (optional: pip install intel-extension-for-pytorch)")

    except Exception as e:
        logger.debug(f"Xeon 6 config: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Gemma 4 chat template helpers
# ─────────────────────────────────────────────────────────────────────────────

# Official Gemma 4 thinking-mode tags
THINK_START = "<think>"
THINK_END   = "</think>"

# Official Gemma 4 turn markers
BOS = "<bos>"
TURN_START = "<start_of_turn>"
TURN_END   = "<end_of_turn>"


def build_chat_prompt(
    user_message: str,
    system_prompt: str = "",
    history: Optional[list[dict]] = None,
    tools: Optional[list[dict]] = None,
    enable_thinking: bool = False,
) -> str:
    """
    Build a Gemma 4 chat-formatted prompt string.

    Gemma 4 format:
      <bos><start_of_turn>system
      {system}<end_of_turn>
      <start_of_turn>user
      {user}<end_of_turn>
      <start_of_turn>model
      {optional: <think>}

    Notes:
      - System role is NATIVE in Gemma 4 (new vs Gemma 3)
      - Tools are injected into the system message as JSON
      - Thinking mode adds <think> at the end to prime CoT
      - History must NOT include previous <think> blocks
        (Gemma 4 spec: strip thinking from history turns)
    """
    parts = [BOS]

    # Build system block (includes tools if provided)
    system_content = system_prompt or ""
    if tools:
        tools_json = json.dumps(tools, indent=2)
        tool_instructions = (
            "\n\nYou have access to the following tools. "
            "When you need to use a tool, respond with a JSON block "
            "inside ```tool_call``` fences:\n"
            "```tool_call\n"
            "{\"name\": \"tool_name\", \"arguments\": {\"key\": \"value\"}}\n"
            "```\n\n"
            f"Available tools:\n{tools_json}"
        )
        system_content = system_content + tool_instructions

    if system_content:
        parts.append(f"{TURN_START}system\n{system_content}{TURN_END}\n")

    # Inject conversation history (strip any <think> blocks from prior turns)
    for turn in (history or []):
        role    = turn.get("role", "user")
        content = _strip_thinking(str(turn.get("content", "")))
        parts.append(f"{TURN_START}{role}\n{content}{TURN_END}\n")

    # Current user message
    parts.append(f"{TURN_START}user\n{user_message}{TURN_END}\n")

    # Prime model response
    parts.append(f"{TURN_START}model\n")
    if enable_thinking:
        parts.append(THINK_START)

    return "".join(parts)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model outputs (required for history)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_tool_call(text: str) -> Optional[dict]:
    """
    Extract the first tool call JSON block from model output.
    Handles both ```tool_call``` fences and bare JSON with 'name'+'arguments'.
    """
    # Primary: fenced block
    m = re.search(r"```tool_call\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Fallback: any JSON object with "name" and "arguments" keys
    m = re.search(r'\{\s*"name"\s*:.*?"arguments"\s*:\s*\{.*?\}\s*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# OncoBridge tool schemas (Gemma 4 native function calling)
# ─────────────────────────────────────────────────────────────────────────────

ONCOBRIDGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_radiomics_features",
            "description": (
                "Extract quantitative radiomics features (shape, texture, intensity) "
                "from a medical image region of interest (ROI). "
                "Returns 100+ PyRadiomics features including GLCM, shape, first-order, wavelet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "modality":        {"type": "string", "enum": ["CT", "MR", "PET", "XRAY"],
                                        "description": "Imaging modality"},
                    "roi":             {"type": "string",
                                        "description": "Region of interest label (e.g. 'primary_tumor')"},
                    "feature_classes": {"type": "array", "items": {"type": "string"},
                                        "description": "Feature classes to extract",
                                        "default": ["shape", "firstorder", "glcm"]},
                },
                "required": ["modality", "roi"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_clinical_trials",
            "description": (
                "Search ClinicalTrials.gov for open recruiting trials that match the "
                "patient's molecular profile, imaging characteristics, and clinical parameters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cancer_type":     {"type": "string",
                                        "description": "e.g. 'Non-Small Cell Lung Cancer'"},
                    "biomarkers":      {"type": "array", "items": {"type": "string"},
                                        "description": "e.g. ['EGFR exon 19 deletion', 'TP53 R175H']"},
                    "prior_therapies": {"type": "array", "items": {"type": "string"},
                                        "description": "Prior lines of therapy received"},
                    "ecog_status":     {"type": "integer", "minimum": 0, "maximum": 4,
                                        "description": "ECOG performance status"},
                },
                "required": ["cancer_type", "biomarkers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_evidence",
            "description": (
                "Retrieve evidence-based passages from PubMed and NCCN guidelines "
                "relevant to a specific clinical question or patient profile."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":         {"type": "string",
                                      "description": "Clinical question or patient context"},
                    "evidence_type": {"type": "string",
                                      "enum": ["treatment", "biomarker", "prognosis", "guideline"],
                                      "description": "Type of evidence to retrieve"},
                    "top_k":         {"type": "integer", "default": 6,
                                      "description": "Number of passages to return"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_radiogenomics_correlation",
            "description": (
                "Evaluate whether the patient's imaging phenotype (radiomics features) "
                "is concordant with their genomic alteration profile. "
                "Flags imaging-genomic discordances that may indicate biopsy sampling error "
                "or tumor heterogeneity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "imaging_features":  {"type": "object",
                                          "description": "Dict of radiomics features (name: value)"},
                    "genomic_markers":   {"type": "array", "items": {"type": "string"},
                                          "description": "List of detected genomic alterations"},
                    "cancer_type":       {"type": "string",
                                          "description": "Cancer type for rule lookup"},
                },
                "required": ["imaging_features", "genomic_markers"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Main Gemma4Engine class
# ─────────────────────────────────────────────────────────────────────────────

class Gemma4Engine:
    """
    Inference engine for the locally-downloaded Kaggle Gemma 4 26B-A4B model.

    Handles:
      ● Text generation (clinical reasoning, report synthesis)
      ● Multimodal generation (CT/MR slice analysis via vision tower)
      ● Native function calling (agentic tool dispatch)
      ● Streaming token output (for Gradio UI)
      ● INT4 / INT8 / BF16 quantization (auto-selected by RAM)
      ● OpenVINO conversion + inference (optional, best Xeon 6 throughput)

    Public API
    ──────────
      engine.generate(prompt, image, tools, stream, system_prompt)
      engine.analyze_medical_image(image_array, modality, clinical_context)
      engine.generate_with_tools(prompt, tools, tool_executor, ...)
      engine.get_performance_stats()
    """

    def __init__(self, config: dict):
        """
        Args:
            config: OncoBridge config dict (from configs/config.yaml).
                    The key field is config["model"]["local_model_dir"] which
                    must point to your downloaded Kaggle folder.
        """
        self.config          = config
        self.model_dir       = self._resolve_model_dir(config)
        self.max_new_tokens  = config["model"].get("max_new_tokens", 2048)
        self.temperature     = config["model"].get("temperature", 0.1)
        self.context_length  = config["model"].get("context_length", 32768)
        self.enable_thinking = config["model"].get("enable_thinking", False)

        # Internals
        self._model      = None
        self._processor  = None
        self._tokenizer  = None
        self._ov_pipe    = None   # OpenVINO GenAI pipeline (if converted)
        self._device     = None
        self._precision  = None

        # Apply Xeon 6 CPU optimisations
        configure_xeon6_torch()

        # Validate folder before loading
        check = validate_model_folder(self.model_dir)
        if not check["valid"]:
            raise FileNotFoundError(
                f"Model folder validation failed:\n"
                + "\n".join(f"  • {i}" for i in check["issues"])
                + f"\n\nExpected folder: {self.model_dir}"
                + "\nKaggle download: kaggle.com/models/google/gemma-4"
            )

        logger.info(
            f"Model folder OK — {check['shard_count']} weight shard(s), "
            f"format: {check['weight_type']}"
        )

        # Load model
        self._load()

    # ── Configuration helpers ─────────────────────────────────────────────────

    def _resolve_model_dir(self, config: dict) -> str:
        """
        Resolve model directory from config or environment variable.
        Priority: ONCOBRIDGE_MODEL_DIR env → config local_model_dir → config openvino_model_dir
        """
        env_dir = os.environ.get("ONCOBRIDGE_MODEL_DIR", "")
        if env_dir:
            return env_dir

        local_dir = config.get("model", {}).get("local_model_dir", "")
        if local_dir:
            return local_dir

        # Legacy key used by older config versions
        ov_dir = config.get("model", {}).get("openvino_model_dir", "")
        if ov_dir:
            return ov_dir

        raise ValueError(
            "Model directory not specified. Set one of:\n"
            "  1. ONCOBRIDGE_MODEL_DIR=/path/to/gemma-4-26b-a4b (env var)\n"
            "  2. model.local_model_dir in configs/config.yaml\n"
            "  3. Pass --model-dir /path/to/gemma-4-26b-a4b to CLI"
        )

    def _detect_precision(self) -> str:
        """
        Auto-select precision based on available RAM.
        Returns: 'bf16', 'int8', or 'int4'
        """
        forced = os.environ.get("ONCOBRIDGE_PRECISION", "").lower()
        if forced in ("bf16", "int8", "int4"):
            return forced

        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / 1e9
            if ram_gb >= 56:
                return "bf16"    # Full precision — best quality
            elif ram_gb >= 30:
                return "int8"    # 8-bit — good quality, half RAM
            else:
                return "int4"    # 4-bit — fits 16 GB+
        except ImportError:
            return "int8"

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load(self):
        """Load Gemma 4 26B-A4B from local folder using HuggingFace Transformers."""
        try:
            import torch
            from transformers import AutoProcessor, Gemma4ForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "HuggingFace Transformers not installed.\n"
                "Run: pip install transformers>=4.51.0 accelerate bitsandbytes torch"
            ) from e

        self._precision = self._detect_precision()
        logger.info(f"Loading Gemma 4 26B-A4B from: {self.model_dir}")
        logger.info(f"Precision: {self._precision.upper()} | Xeon 6 optimised")

        t0 = time.time()

        # ── Load processor (handles text tokenisation + image preprocessing) ──
        self._processor = AutoProcessor.from_pretrained(
            self.model_dir,
            local_files_only=True,   # Never phone home — use only local files
        )
        # The processor contains both tokenizer and image processor
        self._tokenizer = self._processor.tokenizer

        # ── Build quantization config ─────────────────────────────────────────
        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "device_map": "cpu",        # Xeon 6: CPU inference
            "attn_implementation": "eager",  # Required for MoE on CPU
        }

        if self._precision == "bf16":
            load_kwargs["torch_dtype"] = torch.bfloat16
            # BF16 uses AMX on Xeon 6 — fastest option with enough RAM

        elif self._precision == "int8":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                load_kwargs["torch_dtype"] = torch.float16
            except ImportError:
                logger.warning("bitsandbytes not found — falling back to BF16")
                load_kwargs["torch_dtype"] = torch.bfloat16
                self._precision = "bf16"

        elif self._precision == "int4":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",   # NF4 = best quality at 4-bit
                )
            except ImportError:
                logger.warning("bitsandbytes not found — falling back to BF16")
                load_kwargs["torch_dtype"] = torch.bfloat16
                self._precision = "bf16"

        # ── Load the model ────────────────────────────────────────────────────
        self._model = Gemma4ForConditionalGeneration.from_pretrained(
            self.model_dir,
            **load_kwargs,
        )
        self._model.eval()

        elapsed = time.time() - t0
        self._device = str(next(self._model.parameters()).device)

        logger.info(
            f"✓ Gemma 4 26B-A4B loaded in {elapsed:.1f}s | "
            f"precision: {self._precision.upper()} | device: {self._device}"
        )

        # Optionally convert to OpenVINO for even faster Xeon 6 inference
        if OV_AVAILABLE and self.config.get("openvino", {}).get("convert_on_load", False):
            self._try_convert_to_openvino()

    def _try_convert_to_openvino(self):
        """
        Optional: convert loaded model to OpenVINO IR for maximum Xeon 6 throughput.
        Only runs if openvino_genai is installed AND convert_on_load: true in config.
        """
        ov_dir = Path(self.config["model"].get("openvino_ir_dir", "./models/gemma4_ov"))
        if (ov_dir / "openvino_model.xml").exists():
            logger.info(f"Loading pre-converted OpenVINO model from {ov_dir}")
            self._ov_pipe = ov_genai.LLMPipeline(
                str(ov_dir), "CPU",
                PERFORMANCE_HINT="THROUGHPUT",
                INFERENCE_PRECISION_HINT="bf16",   # AMX-BF16
                DYNAMIC_QUANTIZATION_GROUP_SIZE="32",
            )
            return

        try:
            logger.info("Converting to OpenVINO IR (one-time, ~5 min)…")
            from optimum.intel import OVModelForCausalLM
            ov_dir.mkdir(parents=True, exist_ok=True)
            ov_model = OVModelForCausalLM.from_pretrained(
                self.model_dir,
                export=True,
                quantization_config={"bits": 4, "sym": True, "group_size": 64},
                ov_config={"PERFORMANCE_HINT": "THROUGHPUT", "INFERENCE_PRECISION_HINT": "bf16"},
                local_files_only=True,
            )
            ov_model.save_pretrained(str(ov_dir))
            self._processor.save_pretrained(str(ov_dir))
            self._ov_pipe = ov_genai.LLMPipeline(str(ov_dir), "CPU")
            logger.info(f"✓ OpenVINO model saved to {ov_dir}")
        except Exception as e:
            logger.warning(f"OpenVINO conversion skipped: {e}")

    # ── Core generation ───────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        image: Optional[np.ndarray] = None,
        tools: Optional[list] = None,
        stream: bool = False,
        system_prompt: str = "",
        history: Optional[list[dict]] = None,
    ) -> Union[str, Iterator[str]]:
        """
        Generate a response from Gemma 4 26B-A4B.

        Args:
            prompt:        User message text
            image:         Optional CT/MR/PET image as numpy array (H, W) or (H, W, 3)
                           — fed directly into the Gemma 4 vision tower
            tools:         Optional list of function schemas for native tool calling
            stream:        If True, returns a generator of string tokens
            system_prompt: System instruction (injected into Gemma 4 system role)
            history:       Prior conversation turns [{"role": ..., "content": ...}]

        Returns:
            str (complete response) or Iterator[str] (token stream)
        """
        import torch

        # ── Use OpenVINO pipeline if available (text-only) ────────────────────
        if self._ov_pipe is not None and image is None:
            return self._generate_openvino(prompt, system_prompt, tools, stream)

        # ── Build inputs ──────────────────────────────────────────────────────
        messages = self._build_messages(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            tools=tools,
            image=image,
        )

        processor_template = getattr(self._processor, "chat_template", None)
        tokenizer = getattr(self._processor, "tokenizer", None)
        tokenizer_template = getattr(tokenizer, "chat_template", None)

        if processor_template:
            print("Using processor chat template", flush=True)

            # apply_chat_template handles the Gemma 4 turn format correctly
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
        elif tokenizer is not None and tokenizer_template:
            print("Using tokenizer chat template", flush=True)
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

        else:
            print("\nNO CHAT TEMPLATE FOUND", flush=True)
            import pprint
            pprint.pp(messages)
            raise RuntimeError(
                "debug stop"
            )
            #raise RuntimeError(
            #    "The loaded model has no processor or tokenizer chat template. "
            #    "Use a compatible instruction-tuned checkpoint, restore its "
            #    "tokenizer_config.json, or implement an explicit prompt fallback."
            #)

        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if self.temperature > 0 else None,
            do_sample=self.temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        if stream:
            return self._stream_tokens(gen_kwargs)

        t0 = time.time()
        with torch.no_grad():
            output_ids = self._model.generate(**gen_kwargs)

        # Decode only new tokens (not the prompt)
        input_len  = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        response   = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        logger.debug(
            f"Generation: {len(new_tokens)} tokens in {time.time()-t0:.2f}s "
            f"({len(new_tokens)/(time.time()-t0+1e-9):.1f} tok/s)"
        )
        return response

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str,
        history: Optional[list],
        tools: Optional[list],
        image: Optional[np.ndarray],
    ) -> list[dict]:
        """
        Build the messages list in Gemma 4 chat format.
        Images are placed BEFORE text in the user turn (Gemma 4 best practice).
        """
        messages = []

        # ── System message ────────────────────────────────────────────────────
        system_content = system_prompt or ""
        if tools:
            tools_json = json.dumps(tools, indent=2)
            system_content += (
                "\n\nYou have access to the following tools. "
                "To use a tool, respond with a JSON block in ```tool_call``` fences:\n"
                "```tool_call\n{\"name\": \"...\", \"arguments\": {...}}\n```\n\n"
                f"Available tools:\n{tools_json}"
            )
        if system_content:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_content}]
            })

        # ── History turns (strip <think> blocks per Gemma 4 spec) ─────────────
        for turn in (history or []):
            role    = turn.get("role", "user")
            content = _strip_thinking(str(turn.get("content", "")))
            messages.append({
                "role": role,
                "content": [{"type": "text", "text": content}]
            })

        # ── Current user turn (image BEFORE text for multimodal) ──────────────
        user_content: list[dict] = []

        if image is not None:
            from PIL import Image as PILImage
            # Normalise numpy array → PIL Image
            img = self._preprocess_medical_image(image)
            user_content.append({"type": "image", "image": img})

        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        return messages

    def _preprocess_medical_image(self, image: np.ndarray):
        """
        Convert a medical image numpy array to PIL Image for the vision tower.
        Handles:
          - Grayscale CT/MR (H, W) → RGB (H, W, 3)
          - HU windowing for CT
          - Percentile normalisation for MR/PET
          - Uint8 conversion for PIL
        """
        from PIL import Image as PILImage

        img = image.astype(np.float32)

        if img.ndim == 2:
            # Grayscale medical image — apply windowed normalisation
            p1, p99 = np.percentile(img[img > img.min()], [1, 99]) \
                      if img.max() > img.min() else (img.min(), img.max())
            img = np.clip((img - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
            img = (img * 255).astype(np.uint8)
            img = np.stack([img, img, img], axis=-1)   # → (H, W, 3) RGB
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.squeeze(img, axis=2)
            img = np.stack([img, img, img], axis=-1)
        else:
            # Already RGB — just normalise range
            if img.max() <= 1.0:
                img = (img * 255)
            img = np.clip(img, 0, 255).astype(np.uint8)

        return PILImage.fromarray(img)

    def _stream_tokens(self, gen_kwargs: dict) -> Iterator[str]:
        """
        Stream tokens from the model using HuggingFace TextIteratorStreamer.
        The Gradio UI receives tokens as they are generated.
        """
        import torch
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )
        gen_kwargs["streamer"] = streamer

        thread = threading.Thread(
            target=self._model.generate,
            kwargs=gen_kwargs,
            daemon=True,
        )
        thread.start()

        # Yield tokens as they arrive
        for token in streamer:
            yield token

    def _generate_openvino(
        self,
        prompt: str,
        system_prompt: str,
        tools: Optional[list],
        stream: bool,
    ) -> Union[str, Iterator[str]]:
        """Text-only generation via OpenVINO GenAI pipeline (faster on Xeon 6)."""
        raw_prompt = build_chat_prompt(
            user_message=prompt,
            system_prompt=system_prompt,
            tools=tools,
            enable_thinking=self.enable_thinking,
        )
        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = self.max_new_tokens
        cfg.temperature    = self.temperature
        cfg.do_sample      = self.temperature > 0

        if stream:
            tokens: list[str] = []

            def cb(token: str) -> bool:
                tokens.append(token)
                return False

            self._ov_pipe.generate(raw_prompt, cfg, streamer=cb)
            return iter(tokens)

        return str(self._ov_pipe.generate(raw_prompt, cfg))

    # ── Medical image analysis ────────────────────────────────────────────────

    MODALITY_PROMPTS = {
        "CT": (
            "You are analyzing a CT scan for oncological assessment. "
            "Report on: lesion size (mm), morphology (round/irregular/spiculated), "
            "density (HU range), margins, ground-glass opacity component, "
            "cavitation, satellite nodules, lymphadenopathy, pleural/pericardial involvement. "
            "Estimate SUV if PET overlay visible. "
        ),
        "MR": (
            "You are analyzing an MRI scan for tumor characterization. "
            "Report on: T1/T2 signal intensity, enhancement pattern (rim/homogeneous/heterogeneous), "
            "necrosis fraction, peritumoral edema extent, mass effect, "
            "infiltration of adjacent structures, restricted diffusion. "
        ),
        "PET": (
            "You are analyzing a PET scan for metabolic tumor assessment. "
            "Report on: FDG avidity pattern, estimated SUVmax, metabolic tumor volume, "
            "total lesion glycolysis, heterogeneity of uptake, "
            "sites of distant uptake, background suppression quality. "
        ),
        "XRAY": (
            "You are analyzing a chest X-ray for oncological findings. "
            "Report on: lung field opacities, nodules/masses (size, location, zone), "
            "consolidation, pleural effusion, mediastinal widening, "
            "hilar adenopathy, bony lesions, diaphragm position. "
        ),
    }

    def analyze_medical_image(
        self,
        image: np.ndarray,
        modality: str = "CT",
        clinical_context: str = "",
    ) -> str:
        """
        Analyze a medical image slice using the Gemma 4 vision tower.

        Args:
            image:            2D numpy array (H, W) or RGB (H, W, 3)
            modality:         "CT" | "MR" | "PET" | "XRAY"
            clinical_context: Free-text patient context

        Returns:
            Structured radiological analysis string
        """
        modality_instruction = self.MODALITY_PROMPTS.get(
            modality.upper(), self.MODALITY_PROMPTS["CT"]
        )

        prompt = (
            f"{modality_instruction}\n\n"
            f"Patient context: {clinical_context or 'Not provided'}\n\n"
            "Provide a structured radiological analysis with:\n"
            "1. PRIMARY FINDING — size, location, characteristics\n"
            "2. MORPHOLOGICAL FEATURES — shape, margins, internal structure\n"
            "3. TEXTURE/SIGNAL — quantitative descriptors where estimable\n"
            "4. ASSOCIATED FINDINGS — nodes, effusions, metastases\n"
            "5. IMAGING IMPRESSION — malignancy likelihood, differential\n"
            "6. MOLECULAR SUBTYPE CLUES — imaging features suggesting specific genomic alterations"
        )

        system = (
            "You are an expert oncologic radiologist AI integrated into OncoBridge AI. "
            "Provide precise, structured, quantitative imaging analysis "
            "suitable for presentation at a multidisciplinary tumor board. "
            "Use standard radiological terminology. "
            "Always note when image quality limits assessment."
        )

        return str(self.generate(
            prompt=prompt,
            image=image,
            system_prompt=system,
            stream=False,
        ))

    # ── Agentic function calling ──────────────────────────────────────────────

    def generate_with_tools(
        self,
        prompt: str,
        tools: list,
        tool_executor,
        system_prompt: str = "",
        history: Optional[list] = None,
        max_tool_rounds: int = 6,
        enable_thinking: bool = False,
    ) -> dict:
        """
        Gemma 4 native function-calling loop for OncoBridge agents.

        Generates a response, parses any tool call, executes it via
        tool_executor, feeds the result back, and repeats until the
        model produces a plain-text final answer (no tool call) or
        max_tool_rounds is reached.

        Args:
            prompt:          Initial user/agent prompt
            tools:           List of tool schema dicts (ONCOBRIDGE_TOOLS)
            tool_executor:   Callable(tool_name: str, args: dict) -> Any
            system_prompt:   System instruction for this agent
            history:         Prior conversation history
            max_tool_rounds: Safety limit on tool call iterations
            enable_thinking: Enable Gemma 4 chain-of-thought thinking mode

        Returns:
            dict with keys:
              response:        Final answer text
              tool_calls:      List of {"name": ..., "args": ..., "result": ...}
              reasoning_trace: Per-round model output snippets
              rounds:          Number of tool call rounds used
        """
        tool_calls_log: list[dict] = []
        reasoning_trace: list[dict] = []
        conv_history = list(history or [])

        for round_num in range(max_tool_rounds):
            # Generate
            response = str(self.generate(
                prompt=prompt if round_num == 0 else
                       f"Tool result received. Continue your clinical analysis.",
                tools=tools,
                system_prompt=system_prompt,
                history=conv_history,
                stream=False,
            ))

            clean_response = _strip_thinking(response)
            reasoning_trace.append({
                "round": round_num,
                "output_preview": clean_response[:400] + ("…" if len(clean_response) > 400 else ""),
            })

            # Try to parse a tool call
            tool_call = parse_tool_call(response)

            if tool_call is None:
                # No tool call → this is the final answer
                logger.info(f"Tool loop complete after {round_num} rounds")
                return {
                    "response": clean_response,
                    "tool_calls": tool_calls_log,
                    "reasoning_trace": reasoning_trace,
                    "rounds": round_num,
                }

            # Execute tool
            tool_name = tool_call.get("name", "unknown")
            tool_args = tool_call.get("arguments", {})
            logger.info(f"[Round {round_num+1}] Calling tool: {tool_name}({list(tool_args.keys())})")

            try:
                tool_result = tool_executor(tool_name, tool_args)
            except Exception as exc:
                tool_result = {"error": str(exc), "tool": tool_name}
                logger.warning(f"Tool {tool_name} raised: {exc}")

            tool_calls_log.append({
                "name":   tool_name,
                "args":   tool_args,
                "result": tool_result,
            })

            # Add model turn + tool result to history for next round
            conv_history.append({"role": "model",   "content": clean_response})
            conv_history.append({
                "role": "user",
                "content": (
                    f"Tool '{tool_name}' returned:\n"
                    f"```json\n{json.dumps(tool_result, indent=2, default=str)}\n```\n"
                    "Continue your clinical analysis using this result."
                ),
            })

        # Hit max rounds — return whatever we have
        logger.warning(f"Tool loop hit max_tool_rounds={max_tool_rounds}")
        return {
            "response": _strip_thinking(response),
            "tool_calls": tool_calls_log,
            "reasoning_trace": reasoning_trace,
            "rounds": max_tool_rounds,
        }

    # ── Performance & diagnostics ─────────────────────────────────────────────

    def get_performance_stats(self) -> dict:
        """Return a dict of current performance metrics for the Gradio UI."""
        stats: dict[str, Any] = {
            "model":      "Gemma 4 26B-A4B (MoE)",
            "model_dir":  self.model_dir,
            "backend":    "OpenVINO GenAI" if self._ov_pipe else "HuggingFace Transformers",
            "precision":  self._precision.upper() if self._precision else "N/A",
            "device":     self._device or "cpu",
            "active_params_per_token": "~3.8B (MoE routing)",
            "context_window": "256K tokens",
        }
        try:
            import psutil, cpuinfo
            cpu = cpuinfo.get_cpu_info()
            stats["cpu"]          = cpu.get("brand_raw", "Intel Xeon 6")
            stats["cpu_cores"]    = psutil.cpu_count(logical=False)
            stats["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
            stats["ram_used_gb"]  = round(psutil.virtual_memory().used  / 1e9, 1)
            stats["cpu_pct"]      = psutil.cpu_percent(interval=0.5)
        except ImportError:
            pass
        return stats

    def benchmark(self, prompt: str = "Summarize the key features of EGFR-mutant NSCLC.", n_tokens: int = 100) -> dict:
        """
        Quick throughput benchmark.
        Returns tokens/second and time-to-first-token estimates.
        """
        logger.info(f"Benchmarking: generating {n_tokens} tokens…")
        t0 = time.time()
        count = 0
        for _ in self._stream_tokens({
            "input_ids": self._tokenizer(prompt, return_tensors="pt")["input_ids"].to(self._device),
            "max_new_tokens": n_tokens,
            "do_sample": False,
        }):
            count += 1
            if count >= n_tokens:
                break
        elapsed = time.time() - t0
        return {
            "tokens_generated": count,
            "elapsed_s": round(elapsed, 2),
            "tok_per_sec": round(count / elapsed, 1),
            "precision": self._precision,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility shim — keeps agents/pipeline.py working unchanged
# ─────────────────────────────────────────────────────────────────────────────

# The pipeline and agents call engine.generate(...) and engine.analyze_medical_image(...)
# Those are already defined above. The shim only re-exports the tool list.
TOOLS = ONCOBRIDGE_TOOLS


# ─────────────────────────────────────────────────────────────────────────────
# CLI — setup, validation and quick test
# ─────────────────────────────────────────────────────────────────────────────

def _build_minimal_config(model_dir: str) -> dict:
    """Build a minimal config dict for CLI use."""
    return {
        "model": {
            "local_model_dir": model_dir,
            "max_new_tokens":  512,
            "temperature":     0.1,
            "context_length":  8192,
            "enable_thinking": False,
        },
        "openvino": {
            "device": "CPU",
            "convert_on_load": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="OncoBridge — Gemma 4 26B-A4B Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Path to downloaded Kaggle model folder (contains config.json + *.safetensors)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Validate the model folder and exit",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Load the model and run a short OncoBridge test prompt",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run a token throughput benchmark and report tok/s",
    )
    parser.add_argument(
        "--precision", choices=["bf16", "int8", "int4"], default=None,
        help="Override auto-detected precision (default: auto by RAM)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ── --check ───────────────────────────────────────────────────────────────
    if args.check:
        print_folder_check(args.model_dir)
        info = validate_model_folder(args.model_dir)
        sys.exit(0 if info["valid"] else 1)

    # ── --test / --benchmark ──────────────────────────────────────────────────
    if args.test or args.benchmark:
        config = _build_minimal_config(args.model_dir)
        if args.precision:
            os.environ["ONCOBRIDGE_PRECISION"] = args.precision

        print(f"\nLoading Gemma 4 26B-A4B from: {args.model_dir}")
        engine = Gemma4Engine(config)

        if args.benchmark:
            print("\nRunning throughput benchmark (100 tokens)…")
            result = engine.benchmark(n_tokens=100)
            print(f"  Precision:   {result['precision'].upper()}")
            print(f"  Tokens:      {result['tokens_generated']}")
            print(f"  Time:        {result['elapsed_s']} s")
            print(f"  Throughput:  {result['tok_per_sec']} tok/s")

        if args.test:
            print("\n─── Test: Text generation ───")
            prompt = (
                "A 65-year-old female non-smoker with EGFR exon 19 deletion "
                "and a 2.4 cm part-solid CT nodule. What is the first-line treatment "
                "recommendation with evidence level?"
            )
            print(f"Prompt: {prompt}\n")
            response = engine.generate(
                prompt=prompt,
                system_prompt=(
                    "You are an expert oncology AI. "
                    "Provide concise, evidence-based clinical recommendations."
                ),
            )
            print(f"Response:\n{response}\n")

            print("─── Stats ───")
            for k, v in engine.get_performance_stats().items():
                print(f"  {k}: {v}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
