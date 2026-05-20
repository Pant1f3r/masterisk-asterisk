"""
Bias Skimmer v1.0.1 — Test Suite

Verifies every audit catches what it should and does not false-positive
when data is clean. Tests integration patterns (sklearn, FastAPI router,
asterisk_bias_gate) and the AuditReport / AuditFinding data shapes.

Run: python3 test_bias_skimmer.py
"""

import sys
import os
import numpy as np
import pandas as pd
import traceback

# Make sure we import the local module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bias_skimmer import (
    BiasSkimmer,
    AuditReport,
    AuditFinding,
    BiasAuditException,
    asterisk_bias_gate,
    BiasSkimmerTransformer,
    SEVERITY_CRITICAL,
    SEVERITY_VULNERABILITY,
    SEVERITY_PROXIED,
)

# ───────────────────────────────────────────────────────────────────────────
# TEST HARNESS
# ───────────────────────────────────────────────────────────────────────────

PASSED = []
FAILED = []

def test(name):
    """Decorator: register a test function."""
    def wrap(fn):
        def runner():
            try:
                fn()
                PASSED.append(name)
                print(f"  ✓ {name}")
            except AssertionError as e:
                FAILED.append((name, str(e)))
                print(f"  ✗ {name}\n      {e}")
            except Exception as e:
                FAILED.append((name, f"{type(e).__name__}: {e}"))
                print(f"  ✗ {name} (CRASH)\n      {type(e).__name__}: {e}")
                traceback.print_exc()
        runner.__name__ = fn.__name__
        return runner
    return wrap


# ───────────────────────────────────────────────────────────────────────────
# DATASET BUILDERS
# ───────────────────────────────────────────────────────────────────────────

def biased_dataset(n=1000, seed=42):
    """Dataset built to trigger Void + Mirror audits."""
    np.random.seed(seed)
    color = np.random.randint(0, 256, n)
    is_dark = color < 51
    return pd.DataFrame({
        "color_value":       color,
        "reflectance_score": np.clip(color / 255 + np.random.normal(0, 0.05, n), 0.01, 1.0),
        "zip_code":          np.where(is_dark,
                                np.random.randint(10000, 30000, n),
                                np.random.randint(50000, 99999, n)),
        "demographic_proxy": np.where(is_dark,
                                np.random.uniform(0.6, 1.0, n),
                                np.random.uniform(0.0, 0.5, n)),
        "error":             np.where(is_dark,
                                np.random.uniform(0.3, 0.6, n),
                                np.random.uniform(0.0, 0.2, n)),
        "outcome":           np.where(is_dark,
                                (np.random.rand(n) > 0.6).astype(int),
                                (np.random.rand(n) > 0.35).astype(int)),
    })

def clean_dataset(n=1000, seed=7):
    """Dataset with no engineered bias — Void and Mirror should NOT fire."""
    np.random.seed(seed)
    color = np.random.randint(1, 256, n)  # no zeros, so Void check has zeros only as nulls
    return pd.DataFrame({
        "color_value":       color,
        "reflectance_score": np.clip(np.random.uniform(0.4, 0.9, n), 0.01, 1.0),
        "zip_code":          np.random.randint(10000, 99999, n),
        "demographic_proxy": np.random.uniform(0.0, 1.0, n),
        "error":             np.random.uniform(0.0, 0.15, n),
        "outcome":           (np.random.rand(n) > 0.5).astype(int),
    })

def proxy_weaponized_dataset(n=1000, seed=11):
    """Dataset with a feature highly correlated to a demographic signal.
    Should trigger Trickster."""
    np.random.seed(seed)
    demographic = np.random.uniform(0, 1, n)
    # zip_code is a near-perfect linear function of demographic_proxy
    zip_code = (demographic * 80000 + 10000 + np.random.normal(0, 200, n)).astype(int)
    return pd.DataFrame({
        "color_value":       np.random.randint(0, 256, n),
        "reflectance_score": np.random.uniform(0.4, 0.9, n),
        "zip_code":          zip_code,
        "demographic_proxy": demographic,
        "error":             np.random.uniform(0.0, 0.1, n),
        "outcome":           (np.random.rand(n) > 0.5).astype(int),
    })


# ───────────────────────────────────────────────────────────────────────────
# UNIT TESTS — VOID AUDIT (zero-point erasure)
# ───────────────────────────────────────────────────────────────────────────

@test("Void Audit fires on biased dataset")
def test_void_fires():
    df = biased_dataset()
    s = BiasSkimmer(df, verbose=False)
    finding = s.check_zero_point_erasure("color_value")
    assert finding is not None, "Expected Void Audit to fire on biased data"
    assert finding.severity == SEVERITY_CRITICAL, f"Expected CRITICAL, got {finding.severity}"
    assert finding.check_name == "ZERO_POINT_ERASURE"
    assert finding.metric < finding.threshold, "metric must be below threshold to fire"

@test("Void Audit nominal on clean dataset")
def test_void_nominal():
    df = clean_dataset()
    s = BiasSkimmer(df, verbose=False)
    finding = s.check_zero_point_erasure("color_value")
    # With no zeros and no engineered bias, should not flag
    assert finding is None, f"Expected no finding on clean data; got {finding}"

@test("Void Audit handles missing column gracefully")
def test_void_missing_col():
    df = clean_dataset()
    s = BiasSkimmer(df, verbose=False)
    finding = s.check_zero_point_erasure("nonexistent_column")
    assert finding is None, "Missing column should return None, not crash"


# ───────────────────────────────────────────────────────────────────────────
# UNIT TESTS — MIRROR AUDIT (reflectance/signal loss)
# ───────────────────────────────────────────────────────────────────────────

@test("Mirror Audit fires on biased dataset")
def test_mirror_fires():
    df = biased_dataset()
    s = BiasSkimmer(df, verbose=False)
    finding = s.check_reflectance_thresholds("reflectance_score")
    assert finding is not None, "Expected Mirror Audit to fire on biased data"
    assert finding.severity == SEVERITY_VULNERABILITY
    assert finding.metric > finding.threshold

@test("Mirror Audit nominal on clean dataset")
def test_mirror_nominal():
    df = clean_dataset()
    s = BiasSkimmer(df, verbose=False)
    finding = s.check_reflectance_thresholds("reflectance_score")
    assert finding is None, f"Expected no Mirror finding on clean data; got {finding}"

@test("Mirror Audit handles missing column gracefully")
def test_mirror_missing_col():
    df = clean_dataset()
    s = BiasSkimmer(df, verbose=False)
    finding = s.check_reflectance_thresholds("nonexistent_column")
    assert finding is None


# ───────────────────────────────────────────────────────────────────────────
# UNIT TESTS — TRICKSTER AUDIT (proxy weaponization)
# ───────────────────────────────────────────────────────────────────────────

@test("Trickster Audit fires on proxy-weaponized dataset")
def test_trickster_fires():
    df = proxy_weaponized_dataset()
    s = BiasSkimmer(df, verbose=False)
    finding = s.detect_proxy_weaponization("zip_code", "demographic_proxy")
    assert finding is not None, "Expected Trickster Audit to fire on proxy-correlated data"
    assert finding.severity == SEVERITY_PROXIED
    assert abs(finding.metric) > finding.threshold

@test("Trickster Audit nominal on clean dataset")
def test_trickster_nominal():
    df = clean_dataset()
    s = BiasSkimmer(df, verbose=False)
    finding = s.detect_proxy_weaponization("zip_code", "demographic_proxy")
    assert finding is None, f"Expected no Trickster finding; got {finding}"

@test("Trickster Audit handles missing column gracefully")
def test_trickster_missing_col():
    df = clean_dataset()
    s = BiasSkimmer(df, verbose=False)
    finding = s.detect_proxy_weaponization("nonexistent", "demographic_proxy")
    assert finding is None


# ───────────────────────────────────────────────────────────────────────────
# UNIT TESTS — CROSSROADS AUDIT (intersectional amplification)
# ───────────────────────────────────────────────────────────────────────────

@test("Crossroads Audit fires on intersectionally biased data")
def test_crossroads_fires():
    np.random.seed(99)
    n = 500
    df = pd.DataFrame({
        "group": np.random.choice([0, 1, 2, 3], n),
        "outcome": 1
    })
    # Force group=0 to have very low outcome rate (intersectional disparity)
    df.loc[df["group"] == 0, "outcome"] = (np.random.rand(sum(df["group"]==0)) > 0.85).astype(int)
    # Other groups have moderate outcome
    for g in [1,2,3]:
        df.loc[df["group"]==g, "outcome"] = (np.random.rand(sum(df["group"]==g)) > 0.3).astype(int)

    s = BiasSkimmer(df, verbose=False)
    findings = s.check_intersectional_amplification("group", feature_cols=[])
    assert len(findings) >= 1, "Expected at least one Crossroads finding"
    assert any(f.severity == SEVERITY_CRITICAL for f in findings)

@test("Crossroads Audit nominal when subsets are balanced")
def test_crossroads_nominal():
    np.random.seed(7)
    n = 1000
    df = pd.DataFrame({
        "group": np.random.choice([0, 1, 2, 3], n),
        "outcome": (np.random.rand(n) > 0.5).astype(int)
    })
    s = BiasSkimmer(df, verbose=False)
    findings = s.check_intersectional_amplification("group", feature_cols=[])
    assert len(findings) == 0, f"Expected no Crossroads findings on balanced data; got {len(findings)}"

@test("Crossroads Audit skips small subsets (n < 20)")
def test_crossroads_small_subset():
    df = pd.DataFrame({
        "group": [0]*10 + [1]*500,
        "outcome": [0]*10 + [1]*500
    })
    s = BiasSkimmer(df, verbose=False)
    findings = s.check_intersectional_amplification("group", feature_cols=[])
    # group=0 has 100% bad outcome but only 10 rows — should be skipped
    assert len(findings) == 0, "Should not flag group with fewer than 20 rows"


# ───────────────────────────────────────────────────────────────────────────
# REPORT & FINDING DATA SHAPE
# ───────────────────────────────────────────────────────────────────────────

@test("AuditFinding has correct dataclass fields")
def test_finding_shape():
    df = biased_dataset()
    s = BiasSkimmer(df, verbose=False)
    finding = s.check_zero_point_erasure("color_value")
    for field in ['severity', 'check_name', 'column', 'message', 'metric', 'threshold', 'timestamp']:
        assert hasattr(finding, field), f"Missing field: {field}"
    assert isinstance(finding.timestamp, str)
    assert 'T' in finding.timestamp, "Timestamp should be ISO format"

@test("AuditReport.to_dict() serializes correctly")
def test_report_to_dict():
    df = biased_dataset()
    s = BiasSkimmer(df, verbose=False)
    report = s.run_audit()
    d = report.to_dict()
    assert "timestamp" in d
    assert "has_critical" in d
    assert "findings" in d
    assert "correlation_matrix" in d
    assert isinstance(d["findings"], list)
    if d["findings"]:
        first = d["findings"][0]
        assert "severity" in first
        assert "metric" in first
        assert "threshold" in first

@test("AuditReport.has_critical accurate")
def test_report_has_critical():
    df_biased = biased_dataset()
    df_clean = clean_dataset()
    r_biased = BiasSkimmer(df_biased, verbose=False).run_audit()
    r_clean = BiasSkimmer(df_clean, verbose=False).run_audit()
    assert r_biased.has_critical is True, "Biased data should have critical finding"
    assert r_clean.has_critical is False, f"Clean data should not have critical finding (got {r_clean.critical_count})"


# ───────────────────────────────────────────────────────────────────────────
# INTEGRATION — asterisk_bias_gate
# ───────────────────────────────────────────────────────────────────────────

@test("asterisk_bias_gate raises BiasAuditException on critical")
def test_gate_raises():
    df = biased_dataset()
    raised = False
    try:
        asterisk_bias_gate(df, halt_on_critical=True)
    except BiasAuditException:
        raised = True
    assert raised, "Gate should raise on critical findings when halt_on_critical=True"

@test("asterisk_bias_gate does not raise when halt_on_critical=False")
def test_gate_silent():
    df = biased_dataset()
    report = asterisk_bias_gate(df, halt_on_critical=False)
    assert isinstance(report, AuditReport)
    assert report.has_critical, "Critical was detected but not raised — correct behavior with halt=False"

@test("asterisk_bias_gate passes clean data without raising")
def test_gate_clean():
    df = clean_dataset()
    try:
        report = asterisk_bias_gate(df, halt_on_critical=True)
        passed = True
    except BiasAuditException:
        passed = False
    assert passed, "Gate must NOT raise on clean data with halt_on_critical=True"


# ───────────────────────────────────────────────────────────────────────────
# INTEGRATION — sklearn transformer
# ───────────────────────────────────────────────────────────────────────────

@test("BiasSkimmerTransformer fits without crash")
def test_transformer_fits():
    df = biased_dataset()
    y = df["outcome"]
    X = df.drop("outcome", axis=1)
    t = BiasSkimmerTransformer(halt_on_critical=False)
    t.fit(X, y)
    assert t.last_report_ is not None
    assert isinstance(t.last_report_, AuditReport)

@test("BiasSkimmerTransformer transform is pass-through")
def test_transformer_passthrough():
    df = biased_dataset()
    t = BiasSkimmerTransformer(halt_on_critical=False)
    t.fit(df.drop("outcome", axis=1), df["outcome"])
    out = t.transform(df.drop("outcome", axis=1))
    assert len(out) == len(df), "Transform should not change row count"

@test("BiasSkimmerTransformer halts on critical when configured")
def test_transformer_halts():
    df = biased_dataset()
    t = BiasSkimmerTransformer(halt_on_critical=True)
    raised = False
    try:
        t.fit(df.drop("outcome", axis=1), df["outcome"])
    except BiasAuditException:
        raised = True
    assert raised, "Transformer should raise on critical findings when configured"


# ───────────────────────────────────────────────────────────────────────────
# RUN
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n╔════════════════════════════════════════════════════════════════════╗")
    print("║  BIAS SKIMMER v1.0.1 — TEST SUITE                                  ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")

    tests = [v for k, v in list(globals().items()) if callable(v) and k.startswith("test_")]
    for t in tests:
        t()

    print(f"\n────────────────────────────────────────────────────────────────────")
    print(f"  PASSED : {len(PASSED)}")
    print(f"  FAILED : {len(FAILED)}")
    print(f"  TOTAL  : {len(PASSED) + len(FAILED)}")
    print(f"────────────────────────────────────────────────────────────────────")
    if FAILED:
        print("\n  FAILURES:")
        for name, msg in FAILED:
            print(f"    ✗ {name}: {msg}")
        sys.exit(1)
    else:
        print("\n  ✓ ALL TESTS PASSED — Bias Skimmer v1.0.1 verified clean.")
        sys.exit(0)
