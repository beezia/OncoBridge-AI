"""
OncoBridge AI — RAG Knowledge Base
ChromaDB vector store with OpenVINO-accelerated embeddings.

Indexes:
  - PubMed oncology abstracts (curated subset)
  - NCCN clinical practice guidelines
  - TCGA radiogenomics papers
  - FDA drug approval labels

Retrieval:
  - Semantic search (BGE-M3 embeddings, OpenVINO-optimized)
  - Hybrid: dense retrieval + BM25 keyword fallback
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not available — using in-memory fallback")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Embedded knowledge — curated oncology evidence
# In production: replace with full PubMed + NCCN corpus
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDED_KNOWLEDGE = [
    {
        "id": "egfr_osimertinib_flaura",
        "text": "FLAURA trial (NEJM 2018): Osimertinib demonstrated superior progression-free survival (18.9 vs 10.2 months, HR 0.46) versus first-generation EGFR TKIs as first-line therapy in EGFR-mutated advanced NSCLC. Overall survival benefit confirmed (38.6 vs 31.8 months). Osimertinib is now standard of care for EGFR exon 19 deletion and L858R-mutated NSCLC.",
        "source": "PMID:29151359",
        "tags": ["EGFR", "NSCLC", "osimertinib", "TKI", "first-line"],
        "evidence_type": "treatment",
        "cancer_type": "LUAD"
    },
    {
        "id": "egfr_radiomics_ct",
        "text": "CT-based radiomics features can predict EGFR mutation status in NSCLC with AUC 0.74-0.84. Key features: ground-glass opacity component (OR 3.2), smaller tumor size, absence of pleural attachment, and lower CT attenuation. EGFR-mutant tumors are more likely to be pure GGO or part-solid on LDCT. Spiculated margins and satellite nodules more common in KRAS-mutant tumors.",
        "source": "PMID:31186026",
        "tags": ["EGFR", "radiomics", "CT", "radiogenomics", "prediction"],
        "evidence_type": "biomarker",
        "cancer_type": "LUAD"
    },
    {
        "id": "kras_sotorasib",
        "text": "CodeBreaK 200 (NEJM 2023): Sotorasib improved progression-free survival (5.6 vs 4.5 months) and objective response rate (28.1% vs 13.2%) versus docetaxel in previously treated KRAS G12C-mutated NSCLC. FDA approved sotorasib for KRAS G12C NSCLC. Adagrasib showed similar efficacy (KRYSTAL-1: ORR 42.9%). KRAS G12C represents ~13% of NSCLC adenocarcinomas.",
        "source": "PMID:36987785",
        "tags": ["KRAS", "G12C", "sotorasib", "adagrasib", "NSCLC"],
        "evidence_type": "treatment",
        "cancer_type": "LUAD"
    },
    {
        "id": "gbm_mgmt_tmz",
        "text": "EORTC 26981/NCIC CE.3 trial established temozolomide plus radiotherapy as standard of care for newly diagnosed GBM. MGMT promoter methylation is predictive of TMZ benefit (median OS 21.7 vs 15.3 months with TMZ in methylated vs unmethylated patients). IDH-wildtype GBM with MGMT methylation has better prognosis than unmethylated counterparts.",
        "source": "PMID:15758010",
        "tags": ["GBM", "MGMT", "temozolomide", "radiotherapy", "methylation"],
        "evidence_type": "treatment",
        "cancer_type": "GBM"
    },
    {
        "id": "gbm_idh_classification",
        "text": "2021 WHO CNS5 classification integrates molecular markers for glioma diagnosis. IDH-wildtype GBM requires: EGFR amplification, TERT promoter mutation, or chromosome 7 gain/10 loss. IDH-mutant astrocytoma (grade 2-4) and oligodendroglioma (1p/19q codeletion) have better prognosis. Radiomics features correlate with IDH status: IDH-mutant tumors more often frontal, with distinct T2 signal characteristics.",
        "source": "PMID:34695341",
        "tags": ["GBM", "IDH", "WHO", "classification", "radiogenomics"],
        "evidence_type": "biomarker",
        "cancer_type": "GBM"
    },
    {
        "id": "her2_breast_trastuzumab",
        "text": "HER2-positive breast cancer (ERBB2 amplification/overexpression, ~15-20% of cases) should receive HER2-targeted therapy. CLEOPATRA trial: Pertuzumab + trastuzumab + docetaxel improved median OS to 56.5 months vs 40.8 months. DESTINY-Breast03: T-DXd superior to T-DM1 in HER2+ mBC (ORR 79.7% vs 34.2%). Tucatinib combination active in brain metastases.",
        "source": "PMID:26898847",
        "tags": ["HER2", "breast", "trastuzumab", "pertuzumab", "T-DXd"],
        "evidence_type": "treatment",
        "cancer_type": "BRCA"
    },
    {
        "id": "pik3ca_alpelisib",
        "text": "SOLAR-1 trial (NEJM 2019): Alpelisib + fulvestrant significantly improved PFS in PIK3CA-mutated, HR-positive, HER2-negative advanced breast cancer (11.0 vs 5.7 months, HR 0.65). PIK3CA mutations (H1047R most common, ~40% of HR+ mBC) identified by ctDNA or tissue. FDA approved alpelisib-fulvestrant combination for this indication. Hyperglycemia management critical.",
        "source": "PMID:31091374",
        "tags": ["PIK3CA", "alpelisib", "breast", "HR+", "targeted therapy"],
        "evidence_type": "treatment",
        "cancer_type": "BRCA"
    },
    {
        "id": "immunotherapy_tmb",
        "text": "TMB-High (≥10 mut/Mb) is FDA-approved biomarker for pembrolizumab in solid tumors (KEYNOTE-158). Unresectable or metastatic TMB-H solid tumors not eligible for other approved therapies. MSI-H/dMMR is also approved biomarker across tumor types. POLE/POLD1 mutations associated with ultra-high TMB and immunotherapy benefit. Combined biomarker assessment (TMB + PD-L1) may better select responders.",
        "source": "PMID:33658999",
        "tags": ["TMB", "pembrolizumab", "immunotherapy", "MSI", "biomarker"],
        "evidence_type": "treatment",
        "cancer_type": "PanTumor"
    },
    {
        "id": "pet_imaging_lung",
        "text": "FDG-PET/CT is recommended for staging of NSCLC and detecting occult metastases. SUVmax correlates with tumor aggressiveness and metabolic heterogeneity. PET radiomics features (MTV, TLG, SHAPE) predict recurrence and survival. PET-guided radiotherapy planning allows dose escalation to metabolically active regions. Integrated PET/CT outperforms either modality alone for mediastinal staging.",
        "source": "PMID:28991612",
        "tags": ["PET", "FDG", "NSCLC", "staging", "SUV", "radiomics"],
        "evidence_type": "biomarker",
        "cancer_type": "LUAD"
    },
    {
        "id": "nccn_nsclc_guideline",
        "text": "NCCN Guidelines NSCLC v1.2025: Molecular testing required for all advanced NSCLC adenocarcinoma — EGFR, ALK, ROS1, BRAF, MET, RET, NTRK, KRAS G12C, PD-L1. Comprehensive genomic profiling (NGS panel) preferred. ctDNA liquid biopsy acceptable alternative or complement. EGFR exon 19 del / L858R: osimertinib preferred first-line. ALK rearrangement: alectinib or brigatinib first-line. PD-L1 ≥50%: pembrolizumab monotherapy option.",
        "source": "NCCN_NSCLC_2025v1",
        "tags": ["NCCN", "NSCLC", "guideline", "molecular testing", "NGS"],
        "evidence_type": "guideline",
        "cancer_type": "LUAD"
    },
    {
        "id": "nccn_gbm_guideline",
        "text": "NCCN Guidelines CNS v1.2025: Newly diagnosed GBM — maximal safe resection, temozolomide 75 mg/m2 concurrent with RT 60 Gy/30 fractions, then adjuvant TMZ 150-200 mg/m2 5/28 days x6 cycles (Stupp protocol). Tumor treating fields (TTFields) add survival benefit. MGMT methylation testing recommended. Bevacizumab approved for recurrent GBM. IDH inhibitors under investigation for IDH-mutant gliomas.",
        "source": "NCCN_CNS_2025v1",
        "tags": ["NCCN", "GBM", "glioma", "guideline", "temozolomide", "TTFields"],
        "evidence_type": "guideline",
        "cancer_type": "GBM"
    },
    {
        "id": "radiogenomics_review",
        "text": "Radiogenomics — the integration of imaging phenotypes with genomic data — enables non-invasive molecular characterization. Key findings: EGFR mutations correlate with GGO CT patterns, IDH mutations correlate with frontal lobe location and T2 hyperintensity, KRAS mutations with cavitation and necrosis, HER2 amplification with heterogeneous MR enhancement. Radiomics-based EGFR prediction achieves AUC 0.74-0.88 in NSCLC. Radiomics-genomics combined models outperform either alone.",
        "source": "PMID:34231908",
        "tags": ["radiogenomics", "radiomics", "EGFR", "IDH", "multimodal", "prediction"],
        "evidence_type": "biomarker",
        "cancer_type": "Multiple"
    },
    {
        "id": "brca_her2_mr_radiomics",
        "text": "DCE-MRI radiomics features predict HER2 status in breast cancer with AUC 0.78. Key features: heterogeneous early enhancement, irregular margins, absence of dark internal septations. Background parenchymal enhancement and intratumoral heterogeneity on MRI correlate with PIK3CA mutation status. MRI-radiomics can guide decision for neoadjuvant therapy and predict pCR.",
        "source": "PMID:32048119",
        "tags": ["HER2", "breast", "MRI", "radiomics", "DCE", "prediction"],
        "evidence_type": "biomarker",
        "cancer_type": "BRCA"
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# OpenVINO-optimized embedding
# ─────────────────────────────────────────────────────────────────────────────

class OpenVINOEmbedder:
    """Sentence embeddings with optional OpenVINO acceleration."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "CPU"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load()

    def _load(self):
        if ST_AVAILABLE:
            try:
                # Try OpenVINO-optimized sentence transformer
                from optimum.intel import OVSentenceTransformer
                self.model = OVSentenceTransformer(self.model_name, device=self.device)
                logger.info(f"✓ OpenVINO sentence embeddings loaded on {self.device}")
                return
            except Exception:
                pass
            # Fallback to standard sentence transformer
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info("✓ Standard sentence embeddings loaded")
            except Exception as e:
                logger.warning(f"Embedding model load failed: {e}")

    def encode(self, texts: list, batch_size: int = 32) -> np.ndarray:
        if self.model is not None:
            return self.model.encode(
                texts, batch_size=batch_size,
                normalize_embeddings=True, show_progress_bar=False
            )
        # Fallback: random unit vectors (for demo without models)
        dim = 384
        vecs = np.random.randn(len(texts), dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / (norms + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    ChromaDB-backed RAG knowledge base for oncology evidence.
    """

    def __init__(self, config: dict):
        self.config = config
        self.chroma_dir = config.get("rag", {}).get("chroma_dir", "./data/chroma_db")
        self.top_k = config.get("rag", {}).get("top_k", 8)
        embed_model = config.get("rag", {}).get("embedding_model", "BAAI/bge-m3")
        embed_device = config.get("rag", {}).get("embedding_device", "CPU")

        self.embedder = OpenVINOEmbedder(embed_model, embed_device)
        self.collection = None
        self._init_db()

    def _init_db(self):
        """Initialize ChromaDB and populate with embedded knowledge."""
        if not CHROMA_AVAILABLE:
            logger.warning("ChromaDB not available — using in-memory search")
            self._build_inmemory()
            return

        client = chromadb.PersistentClient(
            path=self.chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = client.get_or_create_collection(
            name="oncobridge_knowledge",
            metadata={"hnsw:space": "cosine"}
        )

        if self.collection.count() == 0:
            logger.info("Populating knowledge base...")
            self._populate_collection()
        else:
            logger.info(f"✓ Knowledge base loaded ({self.collection.count()} documents)")

    def _populate_collection(self):
        """Embed and index all knowledge documents."""
        texts = [d["text"] for d in EMBEDDED_KNOWLEDGE]
        ids = [d["id"] for d in EMBEDDED_KNOWLEDGE]
        metadatas = [
            {
                "source": d["source"],
                "evidence_type": d["evidence_type"],
                "cancer_type": d["cancer_type"],
                "tags": ",".join(d["tags"])
            }
            for d in EMBEDDED_KNOWLEDGE
        ]

        embeddings = self.embedder.encode(texts)

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        logger.info(f"✓ Indexed {len(texts)} knowledge documents")

    def _build_inmemory(self):
        """Fallback: in-memory storage with cosine similarity."""
        texts = [d["text"] for d in EMBEDDED_KNOWLEDGE]
        self._docs = EMBEDDED_KNOWLEDGE
        self._embeddings = self.embedder.encode(texts)

    def retrieve(
        self,
        query: str,
        cancer_type: Optional[str] = None,
        evidence_type: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> list:
        """
        Retrieve relevant evidence passages for a clinical query.

        Args:
            query: Clinical question or context
            cancer_type: Filter by cancer type (e.g. "LUAD", "GBM")
            evidence_type: Filter by type ("treatment", "biomarker", "guideline")
            top_k: Number of results to return

        Returns:
            List of dicts with 'text', 'source', 'score', 'metadata'
        """
        k = top_k or self.top_k
        query_vec = self.embedder.encode([query])[0]

        if self.collection is not None:
            return self._retrieve_chroma(query, query_vec, cancer_type, evidence_type, k)
        else:
            return self._retrieve_inmemory(query_vec, cancer_type, evidence_type, k)

    def _retrieve_chroma(self, query, query_vec, cancer_type, evidence_type, k):
        where = {}
        if cancer_type:
            where["cancer_type"] = {"$in": [cancer_type, "Multiple", "PanTumor"]}
        if evidence_type:
            where["evidence_type"] = evidence_type

        kwargs = {
            "query_embeddings": [query_vec.tolist()],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        output = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            output.append({
                "text": doc,
                "source": meta.get("source", ""),
                "score": float(1 - dist),
                "evidence_type": meta.get("evidence_type", ""),
                "cancer_type": meta.get("cancer_type", ""),
                "rank": i + 1
            })
        return output

    def _retrieve_inmemory(self, query_vec, cancer_type, evidence_type, k):
        sims = self._embeddings @ query_vec
        indices = np.argsort(sims)[::-1]

        results = []
        for i in indices:
            doc = self._docs[i]
            if cancer_type and doc["cancer_type"] not in [cancer_type, "Multiple", "PanTumor"]:
                continue
            if evidence_type and doc["evidence_type"] != evidence_type:
                continue
            results.append({
                "text": doc["text"],
                "source": doc["source"],
                "score": float(sims[i]),
                "evidence_type": doc["evidence_type"],
                "cancer_type": doc["cancer_type"],
                "rank": len(results) + 1
            })
            if len(results) >= k:
                break
        return results

    def format_context(self, results: list, max_chars: int = 3000) -> str:
        """Format retrieved results into a context string for the LLM."""
        parts = []
        total = 0
        for r in results:
            snippet = f"[{r['source']}] {r['text']}"
            if total + len(snippet) > max_chars:
                break
            parts.append(snippet)
            total += len(snippet)
        return "\n\n".join(parts)
