"""Recommendation generator for DrugOracle."""

from typing import Dict, List


def generate_recommendations(admet_preds: Dict[str, float], alerts: List[str]) -> List[Dict]:
    recs = []

    for alert in alerts:
        recs.append(
            {
                "type": "Structural Alert",
                "issue": alert,
                "suggestion": "Modify or replace substructure",
                "severity": "high",
                "expected_improvement": "Reduce toxicity risk",
            }
        )

    if admet_preds.get("herg", 0) > 0.5:
        recs.append(
            {
                "type": "Safety",
                "issue": "hERG inhibition risk",
                "suggestion": "Reduce LogP or remove basic amines",
                "severity": "high",
                "expected_improvement": "Lower cardiotoxicity",
            }
        )

    if admet_preds.get("bioavailability_ma", 1) < 0.5:
        recs.append(
            {
                "type": "Bioavailability",
                "issue": "Low predicted bioavailability",
                "suggestion": "Add polar groups or reduce MW",
                "severity": "medium",
                "expected_improvement": "Improve oral exposure",
            }
        )

    return recs
