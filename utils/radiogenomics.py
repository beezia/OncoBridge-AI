"""
OncoBridge AI — Radiogenomics Correlation Engine
Maps imaging phenotypes to expected genomic alterations
and flags imaging-genomic discordances.

Based on TCGA radiogenomics studies (LUAD, GBM, BRCA cohorts).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Radiogenomics correlation rules (derived from TCGA studies)
# Format: {cancer_type: {genomic_marker: {expected_imaging_features, confidence}}}
RADIOGENOMICS_RULES = {
    "LUAD": {
        "EGFR_mutant": {
            "expected": {
                "sphericity": (">", 0.65, "Spherical/round tumor"),
                "ground_glass": True,
                "spiculation": False,
                "size_mm": ("<", 30, "Often smaller at diagnosis"),
                "glcm_homogeneity": (">", 0.70, "More homogeneous texture"),
            },
            "correlation_refs": ["PMID:31186026", "PMID:28094636"],
            "confidence": 0.74,
            "discordance_threshold": 0.50
        },
        "KRAS_mutant": {
            "expected": {
                "sphericity": ("<", 0.65, "Irregular morphology"),
                "cavitation": True,
                "necrosis": True,
                "glcm_entropy": (">", 5.0, "Higher heterogeneity"),
            },
            "correlation_refs": ["PMID:29438337"],
            "confidence": 0.68,
            "discordance_threshold": 0.45
        },
        "ALK_fusion": {
            "expected": {
                "pericardial_effusion": True,
                "pleural_effusion": True,
                "mucin_density": True,
                "size_mm": (">", 30, "Often larger at presentation"),
            },
            "correlation_refs": ["PMID:25399271"],
            "confidence": 0.65,
            "discordance_threshold": 0.45
        },
        "STK11_mutant": {
            "expected": {
                "consolidation": True,
                "mucus_plugging": True,
                "glcm_entropy": (">", 5.5, "Very heterogeneous"),
            },
            "correlation_refs": ["PMID:30413532"],
            "confidence": 0.61,
            "discordance_threshold": 0.40
        }
    },
    "GBM": {
        "IDH_wildtype": {
            "expected": {
                "ring_enhancement": True,
                "necrosis": True,
                "edema_extent": "extensive",
                "multifocal": False,
                "location": "various",
            },
            "correlation_refs": ["PMID:30946123"],
            "confidence": 0.79,
            "discordance_threshold": 0.55
        },
        "IDH_mutant": {
            "expected": {
                "location": "frontal_lobe",
                "t2_hyperintensity": "well_defined",
                "enhancement": "minimal",
                "infiltrative": True,
            },
            "correlation_refs": ["PMID:32234292"],
            "confidence": 0.82,
            "discordance_threshold": 0.60
        },
        "MGMT_methylated": {
            "expected": {
                "location": "frontal_preferred",
                "edema_extent": "moderate",
                "necrosis": True,
            },
            "correlation_refs": ["PMID:28945706"],
            "confidence": 0.67,
            "discordance_threshold": 0.45
        }
    },
    "BRCA": {
        "HER2_positive": {
            "expected": {
                "enhancement": "heterogeneous",
                "margins": "irregular",
                "spiculation": True,
                "necrosis": True,
                "pet_avidity": "high",
            },
            "correlation_refs": ["PMID:32048119"],
            "confidence": 0.78,
            "discordance_threshold": 0.55
        },
        "PIK3CA_mutant": {
            "expected": {
                "enhancement": "heterogeneous",
                "internal_septations": True,
                "background_enhancement": "high",
            },
            "correlation_refs": ["PMID:31428895"],
            "confidence": 0.71,
            "discordance_threshold": 0.50
        },
        "TNBC": {
            "expected": {
                "shape": "oval/round",
                "margins": "circumscribed",
                "necrosis": True,
                "pet_avidity": "very_high",
                "enhancement_kinetics": "rapid_washout",
            },
            "correlation_refs": ["PMID:28455238"],
            "confidence": 0.75,
            "discordance_threshold": 0.52
        }
    }
}


class RadiogenomicsEngine:
    """
    Evaluates concordance between imaging phenotype and genomic profile.
    Flags discordances that may indicate:
      - Incorrect genomic test result
      - Tumor heterogeneity / sampling bias in biopsy
      - Mixed histology
      - Need for re-biopsy
    """

    def __init__(self):
        self.rules = RADIOGENOMICS_RULES

    def evaluate(
        self,
        imaging_features: dict,
        genomic_markers: list,
        cancer_type: str,
    ) -> dict:
        """
        Evaluate concordance between imaging and genomic data.

        Args:
            imaging_features: Dict of quantitative radiomics features
            genomic_markers: List of detected genomic alterations
                             e.g. ["EGFR:exon19del", "TP53:R175H"]
            cancer_type: e.g. "LUAD", "GBM", "BRCA"

        Returns:
            dict with concordance_score, discordances, recommendations
        """
        cancer_rules = self.rules.get(cancer_type, {})
        if not cancer_rules:
            return self._no_rules_result(cancer_type)

        evaluations = []
        discordances = []
        concordances = []

        for marker_str in genomic_markers:
            # Normalize marker to rule key
            rule_key = self._match_rule_key(marker_str, cancer_rules)
            if rule_key is None:
                continue

            rule = cancer_rules[rule_key]
            expected = rule["expected"]
            confidence = rule["confidence"]

            matched_features = 0
            total_features = 0
            feature_details = []

            for feature_name, expected_value in expected.items():
                total_features += 1
                actual = imaging_features.get(feature_name)

                if actual is None:
                    feature_details.append(f"  • {feature_name}: not available in imaging")
                    continue

                concordant = self._check_concordance(actual, expected_value)

                if concordant:
                    matched_features += 1
                    feature_details.append(
                        f"  ✓ {feature_name}: {self._format_expected(expected_value)} — concordant"
                    )
                else:
                    feature_details.append(
                        f"  ✗ {feature_name}: expected {self._format_expected(expected_value)}, "
                        f"found {actual} — discordant"
                    )

            concordance_score = matched_features / max(total_features, 1)
            discordance_threshold = rule.get("discordance_threshold", 0.5)

            eval_result = {
                "marker": marker_str,
                "rule_key": rule_key,
                "concordance_score": concordance_score,
                "confidence": confidence,
                "feature_details": feature_details,
                "references": rule.get("correlation_refs", []),
                "is_discordant": concordance_score < discordance_threshold
            }

            evaluations.append(eval_result)

            if eval_result["is_discordant"]:
                discordances.append({
                    "marker": marker_str,
                    "concordance_score": concordance_score,
                    "threshold": discordance_threshold,
                    "message": (
                        f"Imaging phenotype does not match expected pattern for {rule_key}. "
                        f"Only {matched_features}/{total_features} features concordant "
                        f"(confidence: {confidence:.0%}). Consider repeat biopsy or "
                        f"comprehensive genomic re-testing."
                    )
                })
            else:
                concordances.append(marker_str)

        # Overall concordance
        if evaluations:
            overall_score = sum(e["concordance_score"] for e in evaluations) / len(evaluations)
        else:
            overall_score = None

        # Generate clinical recommendations
        recommendations = self._generate_recommendations(
            discordances, concordances, cancer_type, overall_score
        )

        return {
            "cancer_type": cancer_type,
            "genomic_markers_evaluated": genomic_markers,
            "evaluations": evaluations,
            "discordances": discordances,
            "concordances": concordances,
            "overall_concordance_score": overall_score,
            "has_discordance": len(discordances) > 0,
            "recommendations": recommendations,
            "clinical_alert": len(discordances) > 0
        }

    def _match_rule_key(self, marker_str: str, rules: dict) -> Optional[str]:
        """Map a genomic marker string to a rule key."""
        marker_lower = marker_str.lower().replace(" ", "_").replace(":", "_")
        for key in rules:
            key_lower = key.lower()
            # Try direct match or substring match
            gene = marker_str.split(":")[0].lower()
            if gene in key_lower:
                return key
        return None

    def _check_concordance(self, actual, expected) -> bool:
        """Check if actual value matches expected specification."""
        if isinstance(expected, bool):
            return bool(actual) == expected
        elif isinstance(expected, tuple) and len(expected) >= 2:
            op, threshold = expected[0], expected[1]
            try:
                actual_f = float(actual)
                if op == ">":
                    return actual_f > threshold
                elif op == "<":
                    return actual_f < threshold
                elif op == ">=":
                    return actual_f >= threshold
                elif op == "<=":
                    return actual_f <= threshold
            except (TypeError, ValueError):
                pass
        elif isinstance(expected, str):
            return str(actual).lower() == expected.lower()
        return False

    def _format_expected(self, expected) -> str:
        if isinstance(expected, bool):
            return "present" if expected else "absent"
        elif isinstance(expected, tuple) and len(expected) >= 3:
            return f"{expected[0]}{expected[1]} ({expected[2]})"
        elif isinstance(expected, tuple) and len(expected) >= 2:
            return f"{expected[0]}{expected[1]}"
        return str(expected)

    def _generate_recommendations(
        self,
        discordances: list,
        concordances: list,
        cancer_type: str,
        score: Optional[float]
    ) -> list:
        recs = []

        if not discordances:
            recs.append(
                "Imaging phenotype is concordant with genomic profile. "
                "Proceed with genomics-guided treatment planning."
            )
        else:
            for d in discordances:
                recs.append(
                    f"⚠️ DISCORDANCE ALERT — {d['marker']}: {d['message']}"
                )
            recs.append(
                "Clinical recommendation: Present discordance findings to tumor board. "
                "Consider repeat tissue biopsy from the dominant imaging lesion, "
                "or liquid biopsy (ctDNA) to confirm genomic profile."
            )

        if concordances:
            recs.append(
                f"Concordant markers: {', '.join(concordances)}. "
                "These genomic alterations are supported by imaging phenotype."
            )

        return recs

    def _no_rules_result(self, cancer_type: str) -> dict:
        return {
            "cancer_type": cancer_type,
            "genomic_markers_evaluated": [],
            "evaluations": [],
            "discordances": [],
            "concordances": [],
            "overall_concordance_score": None,
            "has_discordance": False,
            "recommendations": [
                f"No radiogenomics correlation rules available for {cancer_type}. "
                "Proceeding with standard genomics-guided treatment planning."
            ],
            "clinical_alert": False
        }

    def format_summary(self, result: dict) -> str:
        """Format radiogenomics evaluation for report."""
        lines = []

        score = result.get("overall_concordance_score")
        if score is not None:
            lines.append(
                f"Radiogenomics concordance score: {score:.0%} "
                f"({'Concordant' if score >= 0.5 else '⚠️ DISCORDANT'})"
            )

        for rec in result.get("recommendations", []):
            lines.append(rec)

        return "\n".join(lines)
