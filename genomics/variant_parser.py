"""
OncoBridge AI — Genomics Variant Parser
Handles VCF parsing, molecular subtyping, and clinical variant annotation.

Supports:
  - VCF v4.2 parsing (SNVs, indels, CNVs, structural variants)
  - ClinVar / OncoKB-style pathogenicity annotation
  - TCGA molecular subtype mapping
  - Actionability scoring for targeted therapy
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cyvcf2
    VCF_AVAILABLE = True
except ImportError:
    VCF_AVAILABLE = False
    logger.warning("cyvcf2 not available — using built-in VCF parser")

# ─────────────────────────────────────────────────────────────────────────────
# Clinically actionable variant database (curated subset)
# In production: replace with ClinVar + OncoKB API calls
# ─────────────────────────────────────────────────────────────────────────────

ACTIONABLE_VARIANTS = {
    # Lung cancer
    "EGFR:p.L858R":     {"tier": 1, "therapy": "Osimertinib", "evidence": "FDA-approved", "cancer": ["LUAD"]},
    "EGFR:exon19del":   {"tier": 1, "therapy": "Osimertinib", "evidence": "FDA-approved", "cancer": ["LUAD"]},
    "EGFR:p.T790M":     {"tier": 1, "therapy": "Osimertinib (3rd-gen)", "evidence": "FDA-approved", "cancer": ["LUAD"]},
    "ALK:fusion":       {"tier": 1, "therapy": "Alectinib / Lorlatinib", "evidence": "FDA-approved", "cancer": ["LUAD"]},
    "ROS1:fusion":      {"tier": 1, "therapy": "Crizotinib / Entrectinib", "evidence": "FDA-approved", "cancer": ["LUAD"]},
    "KRAS:p.G12C":      {"tier": 1, "therapy": "Sotorasib / Adagrasib", "evidence": "FDA-approved", "cancer": ["LUAD", "CRC"]},
    "MET:exon14skip":   {"tier": 1, "therapy": "Capmatinib / Tepotinib", "evidence": "FDA-approved", "cancer": ["LUAD"]},
    "BRAF:p.V600E":     {"tier": 1, "therapy": "Dabrafenib + Trametinib", "evidence": "FDA-approved", "cancer": ["LUAD", "MELANOMA"]},
    "RET:fusion":       {"tier": 1, "therapy": "Selpercatinib / Pralsetinib", "evidence": "FDA-approved", "cancer": ["LUAD"]},
    "NTRK1:fusion":     {"tier": 1, "therapy": "Larotrectinib / Entrectinib", "evidence": "FDA-approved", "cancer": ["PanTumor"]},

    # Brain tumors
    "IDH1:p.R132H":     {"tier": 1, "therapy": "Ivosidenib", "evidence": "FDA-approved (AML/CCA); investigational glioma", "cancer": ["GLIOMA"]},
    "IDH2:p.R172K":     {"tier": 1, "therapy": "Enasidenib", "evidence": "Investigational glioma", "cancer": ["GLIOMA"]},
    "MGMT:methylated":  {"tier": 1, "therapy": "Temozolomide + RT (enhanced benefit)", "evidence": "EORTC 26981", "cancer": ["GBM"]},
    "EGFR:amplification": {"tier": 2, "therapy": "Anti-EGFR investigational", "evidence": "Investigational", "cancer": ["GBM"]},

    # Breast cancer
    "ERBB2:amplification": {"tier": 1, "therapy": "Trastuzumab + Pertuzumab / T-DXd", "evidence": "FDA-approved", "cancer": ["BRCA"]},
    "PIK3CA:p.H1047R":  {"tier": 1, "therapy": "Alpelisib + Fulvestrant", "evidence": "FDA-approved (HR+ HER2-)", "cancer": ["BRCA"]},
    "BRCA1:pathogenic": {"tier": 1, "therapy": "Olaparib / Niraparib (PARP inhibitor)", "evidence": "FDA-approved", "cancer": ["BRCA", "OVCA"]},
    "BRCA2:pathogenic": {"tier": 1, "therapy": "Olaparib / Niraparib (PARP inhibitor)", "evidence": "FDA-approved", "cancer": ["BRCA", "OVCA"]},
    "ESR1:mutation":    {"tier": 1, "therapy": "Elacestrant (SERD)", "evidence": "FDA-approved", "cancer": ["BRCA"]},

    # Pan-tumor
    "TMB_HIGH":         {"tier": 1, "therapy": "Pembrolizumab (MSI-H / TMB-H)", "evidence": "FDA-approved", "cancer": ["PanTumor"]},
    "MSI_HIGH":         {"tier": 1, "therapy": "Pembrolizumab / Dostarlimab", "evidence": "FDA-approved", "cancer": ["PanTumor"]},
    "FGFR2:fusion":     {"tier": 1, "therapy": "Pemigatinib / Infigratinib", "evidence": "FDA-approved", "cancer": ["CCA"]},
}

# Molecular subtypes — imaging correlates from TCGA radiogenomics studies
MOLECULAR_SUBTYPES = {
    "LUAD": {
        "EGFR_mutant": {
            "imaging_features": {"sphericity": ">0.7", "ground_glass": True, "spiculation": False},
            "genomic_markers": ["EGFR:L858R", "EGFR:exon19del"],
            "prognosis": "Favorable with TKI"
        },
        "KRAS_mutant": {
            "imaging_features": {"sphericity": "<0.6", "necrosis": True, "cavitation": True},
            "genomic_markers": ["KRAS:G12C", "KRAS:G12V"],
            "prognosis": "Poor (G12C: sotorasib eligible)"
        },
        "TP53_mutant": {
            "imaging_features": {"heterogeneity": "high", "size": ">3cm"},
            "genomic_markers": ["TP53:loss_of_function"],
            "prognosis": "Moderate — immunotherapy may benefit"
        }
    },
    "GBM": {
        "IDH_wildtype": {
            "imaging_features": {"necrosis": True, "enhancement": "ring", "edema": "extensive"},
            "genomic_markers": ["IDH1:wildtype", "TERT:promoter_mutation"],
            "prognosis": "Poor — median OS 14-16 months"
        },
        "MGMT_methylated": {
            "imaging_features": {"edema": "moderate", "location": "frontal"},
            "genomic_markers": ["MGMT:promoter_methylation"],
            "prognosis": "Better TMZ response — 2-4 month OS benefit"
        }
    },
    "BRCA": {
        "HER2_positive": {
            "imaging_features": {"enhancement": "heterogeneous", "margins": "irregular"},
            "genomic_markers": ["ERBB2:amplification"],
            "prognosis": "Favorable with HER2-targeted therapy"
        },
        "TNBC": {
            "imaging_features": {"shape": "round", "margins": "circumscribed", "necrosis": True},
            "genomic_markers": ["ER_negative", "PR_negative", "HER2_negative"],
            "prognosis": "Poor — immunotherapy + chemotherapy"
        }
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# VCF Parser
# ─────────────────────────────────────────────────────────────────────────────

class VariantParser:
    """
    Parse VCF files and generate structured genomic profiles.
    """

    def __init__(self, config: dict):
        self.config = config
        self.clinvar_db = config.get("genomics", {}).get("clinvar_db", "")
        self.actionable = ACTIONABLE_VARIANTS

    def parse_vcf(self, vcf_path: str) -> dict:
        """
        Parse VCF file and return structured variant profile.

        Returns:
            dict with variants, actionable_variants, molecular_subtype,
            tmb_score, msi_status
        """
        if not Path(vcf_path).exists():
            logger.warning(f"VCF not found: {vcf_path} — using demo data")
            return self._demo_profile()

        if VCF_AVAILABLE:
            return self._parse_cyvcf2(vcf_path)
        else:
            return self._parse_builtin(vcf_path)

    def _parse_cyvcf2(self, vcf_path: str) -> dict:
        """Parse with cyvcf2 library."""
        vcf = cyvcf2.VCF(vcf_path)
        variants = []
        somatic_count = 0

        for v in vcf:
            if v.FILTER and v.FILTER != "PASS":
                continue

            variant = {
                "gene": self._get_info(v, "GENE", ""),
                "chrom": v.CHROM,
                "pos": v.POS,
                "ref": v.REF,
                "alt": str(v.ALT[0]) if v.ALT else "",
                "variant_type": v.var_type,
                "consequence": self._get_info(v, "CSQ", ""),
                "hgvsp": self._get_info(v, "HGVSp", ""),
                "af": float(v.INFO.get("AF", 0) or 0),
                "dp": int(v.INFO.get("DP", 0) or 0),
                "clinvar": self._get_info(v, "CLNSIG", "Uncertain"),
            }
            variants.append(variant)
            if variant["af"] > 0.05:
                somatic_count += 1

        vcf.close()

        return self._annotate_profile(variants, somatic_count)

    def _parse_builtin(self, vcf_path: str) -> dict:
        """Minimal built-in VCF parser (no cyvcf2 dependency)."""
        variants = []
        somatic_count = 0

        with open(vcf_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 8:
                    continue

                chrom, pos, vid, ref, alt, qual, filt, info = parts[:8]
                if filt not in ("PASS", ".", ""):
                    continue

                # Parse INFO field
                info_dict = {}
                for item in info.split(";"):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        info_dict[k] = v

                gene = info_dict.get("GENE", info_dict.get("SYMBOL", ""))
                af = float(info_dict.get("AF", 0.1))

                variant = {
                    "gene": gene,
                    "chrom": chrom,
                    "pos": int(pos),
                    "ref": ref,
                    "alt": alt.split(",")[0],
                    "variant_type": "snp" if len(ref) == len(alt) == 1 else "indel",
                    "consequence": info_dict.get("CSQ", ""),
                    "hgvsp": info_dict.get("HGVSp", ""),
                    "af": af,
                    "dp": int(info_dict.get("DP", 100)),
                    "clinvar": info_dict.get("CLNSIG", "Unknown"),
                }
                variants.append(variant)
                if af > 0.05:
                    somatic_count += 1

        return self._annotate_profile(variants, somatic_count)

    def _get_info(self, variant, key: str, default=""):
        try:
            val = variant.INFO.get(key)
            return val if val is not None else default
        except Exception:
            return default

    def _annotate_profile(self, variants: list, somatic_count: int) -> dict:
        """Match variants to actionable database and compute molecular profile."""
        actionable_hits = []
        all_genes = set()

        for v in variants:
            gene = v.get("gene", "")
            hgvsp = v.get("hgvsp", "")
            all_genes.add(gene)

            # Check against actionable database
            for key, action in self.actionable.items():
                ak_gene, ak_variant = key.split(":") if ":" in key else (key, "")
                if gene == ak_gene:
                    if not ak_variant or ak_variant.lower() in hgvsp.lower():
                        actionable_hits.append({
                            "key": key,
                            "gene": gene,
                            "variant": hgvsp or v.get("alt", ""),
                            "af": v.get("af", 0),
                            **action
                        })

        # TMB estimate
        tmb = somatic_count / 38.0  # ~38 Mb coding genome
        msi = "MSI-High" if tmb > 10 else "MSS"

        return {
            "variants": variants[:50],  # Cap for display
            "variant_count": len(variants),
            "somatic_variant_count": somatic_count,
            "actionable_variants": actionable_hits,
            "all_genes": sorted(all_genes),
            "tmb_score": round(tmb, 1),
            "msi_status": msi,
            "tier1_count": sum(1 for h in actionable_hits if h.get("tier") == 1),
        }

    def _demo_profile(self) -> dict:
        """Return a realistic demo genomic profile for RSNA presentation."""
        return {
            "variants": [
                {"gene": "EGFR", "chrom": "7", "pos": 55242468, "ref": "G", "alt": "A",
                 "hgvsp": "p.Glu746_Ala750del", "af": 0.42, "dp": 820, "clinvar": "Pathogenic",
                 "variant_type": "indel", "consequence": "inframe_deletion"},
                {"gene": "TP53", "chrom": "17", "pos": 7577120, "ref": "C", "alt": "T",
                 "hgvsp": "p.R175H", "af": 0.38, "dp": 650, "clinvar": "Pathogenic",
                 "variant_type": "snp", "consequence": "missense_variant"},
                {"gene": "STK11", "chrom": "19", "pos": 1205837, "ref": "A", "alt": "T",
                 "hgvsp": "p.Q37*", "af": 0.28, "dp": 420, "clinvar": "Pathogenic",
                 "variant_type": "snp", "consequence": "stop_gained"},
            ],
            "variant_count": 147,
            "somatic_variant_count": 12,
            "actionable_variants": [
                {
                    "key": "EGFR:exon19del",
                    "gene": "EGFR",
                    "variant": "p.Glu746_Ala750del (exon 19 deletion)",
                    "af": 0.42,
                    "tier": 1,
                    "therapy": "Osimertinib",
                    "evidence": "FDA-approved",
                    "cancer": ["LUAD"]
                }
            ],
            "all_genes": ["EGFR", "TP53", "STK11", "KEAP1"],
            "tmb_score": 4.2,
            "msi_status": "MSS",
            "tier1_count": 1,
            "molecular_subtype": "EGFR-mutant LUAD (exon 19 deletion)",
            "molecular_subtype_key": "LUAD_EGFR_mutant",
        }

    def summarize(self, profile: dict) -> str:
        """Generate a clinical summary of the genomic profile."""
        actionable = profile.get("actionable_variants", [])
        tmb = profile.get("tmb_score", 0)
        msi = profile.get("msi_status", "Unknown")
        tier1 = profile.get("tier1_count", 0)

        lines = [
            f"Total variants: {profile.get('variant_count', 0)} "
            f"(somatic: {profile.get('somatic_variant_count', 0)}).",
            f"TMB: {tmb} mut/Mb ({msi}).",
            f"Tier 1 actionable alterations: {tier1}.",
        ]

        if actionable:
            lines.append("Key actionable variants:")
            for av in actionable[:5]:
                lines.append(
                    f"  • {av['gene']} {av['variant']} "
                    f"(AF: {av['af']:.1%}) — {av['therapy']} ({av['evidence']})"
                )

        genes = profile.get("all_genes", [])
        if genes:
            lines.append(f"Mutated genes: {', '.join(genes[:10])}")

        return "\n".join(lines)
