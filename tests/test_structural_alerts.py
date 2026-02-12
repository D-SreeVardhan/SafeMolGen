"""Tests for structural alerts (dataset load and detection)."""

import csv
import tempfile
from pathlib import Path

import pytest

from models.oracle.structural_alerts import (
    STRUCTURAL_ALERTS_DB,
    StructuralAlert,
    detect_structural_alerts,
    load_structural_alerts_from_csv,
)


def test_load_structural_alerts_from_csv_returns_dict():
    """load_structural_alerts_from_csv returns a dict keyed by id."""
    path = Path(__file__).resolve().parents[1] / "data" / "structural_alerts.csv"
    result = load_structural_alerts_from_csv(path)
    assert isinstance(result, dict)
    # With default data/structural_alerts.csv present, we expect at least the built-in set
    assert len(result) >= 5, "Expected at least 5 alerts (built-in or from CSV)"


def test_load_structural_alerts_from_csv_missing_file():
    """Missing file returns empty dict."""
    result = load_structural_alerts_from_csv(Path("/nonexistent/alerts.csv"))
    assert result == {}


def test_load_structural_alerts_from_csv_valid_row():
    """A valid CSV row is loaded as StructuralAlert."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "smarts", "category", "severity", "recommendation"])
        w.writerow(["test_alert", "Test Alert", "[#6]", "general", "low", "Review"])
        path = Path(f.name)
    try:
        result = load_structural_alerts_from_csv(path)
        assert "test_alert" in result
        alert = result["test_alert"]
        assert isinstance(alert, StructuralAlert)
        assert alert.name == "Test Alert"
        assert alert.smarts == "[#6]"
        assert alert.category == "general"
    finally:
        path.unlink(missing_ok=True)


def test_detect_structural_alerts_returns_names_resolvable_by_pipeline():
    """detect_structural_alerts returns names that pipeline can resolve to SMARTS."""
    # Any name in STRUCTURAL_ALERTS_DB values should be resolvable
    names_seen = {a.name for a in STRUCTURAL_ALERTS_DB.values()}
    assert len(names_seen) >= 5
    # Smoke: detect on a simple molecule; returned names should be in our DB
    hits, atoms = detect_structural_alerts("CCO")
    assert isinstance(hits, list)
    for name in hits:
        assert name in names_seen
    # Molecule with nitro should trigger Aromatic Nitro if that alert is loaded
    hits_nitro, _ = detect_structural_alerts("c1ccccc1[N+](=O)[O-]")
    assert isinstance(hits_nitro, list)


def test_structural_alerts_db_non_empty():
    """STRUCTURAL_ALERTS_DB is non-empty (from CSV or built-in fallback)."""
    assert len(STRUCTURAL_ALERTS_DB) >= 5
    for key, alert in STRUCTURAL_ALERTS_DB.items():
        assert alert.name
        assert alert.smarts
        assert alert.pattern() is not None, f"Invalid SMARTS for {key}"
