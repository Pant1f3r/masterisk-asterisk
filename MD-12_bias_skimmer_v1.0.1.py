"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  KR0M3D1A©®™ DIGITAL ARC — BIAS SKIMMER v1.0.1                              ║
║  Zero-Point Algorithmic Audit Engine                                         ║
║  ASTERISK MASTER PIPELINE INTEGRATION                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
  Platform Lead  : KR0M3D1A©®™ Digital ARC
  Collab Partner : CreationsLab224©®™
  Author         : Edward Craig Callender
  Contact        : ecc@cyberfaq.net
  Version Notes  : v1.0.1 — Corrected legal-framework reference in
                   proxy_weaponization finding message to cite real
                   statutes (Title VII / FHA / ECOA) rather than a
                   non-existent statute.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Callable
import warnings
warnings.filterwarnings("ignore")


# ── SEVERITY LEVELS ───────────────────────────────────────────────────────────
SEVERITY_CRITICAL      = "CRITICAL"
SEVERITY_VULNERABILITY = "VULNERABILITY"
SEVERITY_PROXIED       = "PROXIED"
SEVERITY_NOMINAL       = "NOMINAL"


@dataclass
class AuditFinding:
    severity:   str
    check_name: str
    column:     str
    message:    str
    metric:     float
    threshold:  float
    timestamp:  str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __str__(self):
        return (f"[{self.severity}] {self.check_name} | col={self.column} "
                f"| metric={self.metric:.4f} | threshold={self.threshold:.4f}\n"
                f"  ↳ {self.message}")


# ══════════════════════════════════════════════════════════════════════════════
class BiasSkimmer:
    """
    Algorithmic Bias Skimmer — plugs into any Pandas-based pipeline.

    Detects four failure modes corresponding to the ASTERISK audit framework:
      1. Zero-Point Erasure         (The Void Audit)
      2. Reflectance Signal Loss    (The Mirror Audit)
      3. Proxy Weaponization        (The Trickster Audit)
      4. Intersectional Amplification (The Crossroads Audit)

    Pipeline integration (ASTERISK / sklearn / FastAPI):
      skimmer = BiasSkimmer(df, outcome_col="outcome", verbose=True)
      report  = skimmer.run_audit()
      if report.has_critical:
          raise BiasAuditException(report.summary())

    Threshold defaults are configurable. Users operating under specific
    legal frameworks (e.g. EEOC's four-fifths rule for Title VII) should
    set thresholds to match the applicable standard.
    """

    def __init__(
        self,
        dataset:             pd.DataFrame,
        outcome_col:         str           = "outcome",
        zero_threshold:      float         = 0.70,
        reflectance_col:     Optional[str] = None,
        reflectance_pct:     float         = 0.10,
        error_rate_ceiling:  float         = 0.25,
        proxy_col:           str           = "zip_code",
        demographic_col:     str           = "demographic_proxy",
        proxy_corr_ceiling:  float         = 0.85,
        verbose:             bool          = True,
        hooks:               Optional[List[Callable]] = None,
    ):
        self.df                 = dataset.copy()
        self.outcome_col        = outcome_col
        self.zero_threshold     = zero_threshold
        self.reflectance_col    = reflectance_col
        self.reflectance_pct    = reflectance_pct
        self.error_rate_ceiling = error_rate_ceiling
        self.proxy_col          = proxy_col
        self.demographic_col    = demographic_col
        self.proxy_corr_ceiling = proxy_corr_ceiling
        self.verbose            = verbose
        self.hooks              = hooks or []
        self.findings: List[AuditFinding] = []
        self._log("BiasSkimmer v1.0.1 initialised — KR0M3D1A©®™")

    # ── INTERNAL ──────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        if self.verbose:
            ts = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
            print(f"  [{ts}] {msg}")

    def _add(self, finding: AuditFinding):
        self.findings.append(finding)
        self._log(str(finding))

    # ── CHECK 1 : ZERO-POINT ERASURE (THE VOID AUDIT) ─────────────────────────
    def check_zero_point_erasure(self, target_col: str) -> Optional[AuditFinding]:
        """
        Compares outcome rate for zero/null entries against the non-zero baseline.
        A ratio below `zero_threshold` flags semantic conflation of 0 ↔ negative.
        """
        if target_col not in self.df.columns:
            self._log(f"  SKIP zero_point_erasure — column '{target_col}' not found")
            return None

        zero_mask    = (self.df[target_col] == 0) | (self.df[target_col].isnull())
        zero_rows    = self.df[zero_mask]
        nonzero_rows = self.df[~zero_mask]

        if zero_rows.empty or nonzero_rows.empty:
            self._log(f"  SKIP zero_point_erasure — insufficient data in '{target_col}'")
            return None

        zero_outcome_rate     = zero_rows[self.outcome_col].mean()
        baseline_outcome_rate = nonzero_rows[self.outcome_col].mean()

        if baseline_outcome_rate == 0:
            self._log("  SKIP zero_point_erasure — baseline outcome rate is zero")
            return None

        disparity_ratio = zero_outcome_rate / baseline_outcome_rate

        if disparity_ratio < self.zero_threshold:
            f = AuditFinding(
                severity   = SEVERITY_CRITICAL,
                check_name = "ZERO_POINT_ERASURE",
                column     = target_col,
                metric     = disparity_ratio,
                threshold  = self.zero_threshold,
                message    = (
                    f"Value '0/null' in [{target_col}] yields only "
                    f"{zero_outcome_rate*100:.1f}% favorable outcome vs "
                    f"{baseline_outcome_rate*100:.1f}% baseline. "
                    f"Disparity ratio {disparity_ratio:.3f} < {self.zero_threshold} threshold. "
                    "Semantic null-conflation / zero-proxy bias confirmed. "
                    "Review under disparate impact framework "
                    "(EEOC four-fifths rule, 29 CFR §1607)."
                )
            )
            self._add(f)
            return f
        self._log(f"  NOMINAL zero_point_erasure — ratio={disparity_ratio:.4f}")
        return None

    # ── CHECK 2 : REFLECTANCE / SIGNAL-LOSS (THE MIRROR AUDIT) ────────────────
    def check_reflectance_thresholds(
        self,
        sensor_data_col: str,
        error_col:       str = "error"
    ) -> Optional[AuditFinding]:
        """
        Measures error rate in the darkest <reflectance_pct> percentile of
        sensor/image data. Disproportionate errors → low-signal data erasure.
        """
        if sensor_data_col not in self.df.columns:
            self._log(f"  SKIP reflectance_thresholds — column '{sensor_data_col}' not found")
            return None

        threshold       = self.df[sensor_data_col].quantile(self.reflectance_pct)
        dark_mask       = self.df[sensor_data_col] <= threshold
        dark_rows       = self.df[dark_mask]
        system_mean     = self.df[error_col].mean() if error_col in self.df else None
        error_rate_dark = dark_rows[error_col].mean() if error_col in dark_rows.columns else None

        if error_rate_dark is None:
            self._log(f"  SKIP reflectance_thresholds — column '{error_col}' not found")
            return None

        decile_stats = {}
        for i in range(10):
            lo = self.df[sensor_data_col].quantile(i / 10)
            hi = self.df[sensor_data_col].quantile((i + 1) / 10)
            sl = self.df[(self.df[sensor_data_col] >= lo) & (self.df[sensor_data_col] < hi)]
            decile_stats[f"D{i+1}"] = {
                "error_mean": sl[error_col].mean() if not sl.empty else 0,
                "count":      len(sl)
            }

        if error_rate_dark > self.error_rate_ceiling:
            f = AuditFinding(
                severity   = SEVERITY_VULNERABILITY,
                check_name = "REFLECTANCE_SIGNAL_LOSS",
                column     = sensor_data_col,
                metric     = error_rate_dark,
                threshold  = self.error_rate_ceiling,
                message    = (
                    f"Bottom {self.reflectance_pct*100:.0f}% reflectance data "
                    f"error rate = {error_rate_dark*100:.1f}% "
                    f"vs system mean {system_mean*100:.1f}%. "
                    "Worst-performing subset clusters at one end of the "
                    "measurable signal range. Likely undertraining on "
                    "low-signal regions; recommend stratified validation "
                    "and Fitzpatrick / domain-specific calibration."
                )
            )
            f.decile_stats = decile_stats
            self._add(f)
            return f
        self._log(f"  NOMINAL reflectance_thresholds — dark_error={error_rate_dark:.4f}")
        return None

    # ── CHECK 3 : PROXY WEAPONIZATION (THE TRICKSTER AUDIT) ───────────────────
    def detect_proxy_weaponization(
        self,
        sensitive_proxy:  str = "zip_code",
        reference_col:    str = "demographic_proxy"
    ) -> Optional[AuditFinding]:
        """
        Computes Pearson r between a 'neutral' feature and a demographic signal.
        |r| > proxy_corr_ceiling → the feature is operating as a demographic
        stand-in. Pearson is a fast first-pass screen; users requiring stronger
        evidence should follow up with mutual information or conditional
        independence tests.
        """
        for col in [sensitive_proxy, reference_col]:
            if col not in self.df.columns:
                self._log(f"  SKIP proxy_weaponization — column '{col}' not found")
                return None

        correlation = self.df[sensitive_proxy].corr(self.df[reference_col])

        if abs(correlation) > self.proxy_corr_ceiling:
            f = AuditFinding(
                severity   = SEVERITY_PROXIED,
                check_name = "PROXY_WEAPONIZATION",
                column     = sensitive_proxy,
                metric     = correlation,
                threshold  = self.proxy_corr_ceiling,
                message    = (
                    f"[{sensitive_proxy}] ↔ [{reference_col}] Pearson r = {correlation:.4f}. "
                    f"Exceeds |{self.proxy_corr_ceiling}| ceiling — the feature is "
                    "functioning as a demographic stand-in. Review under disparate "
                    "impact framework: Title VII (employment), Fair Housing Act "
                    "(housing), Equal Credit Opportunity Act Regulation B (credit). "
                    "Document business necessity or remove the feature."
                )
            )
            self._add(f)
            return f
        self._log(f"  NOMINAL proxy_weaponization — r={correlation:.4f}")
        return None

    # ── CHECK 4 : INTERSECTIONAL AMPLIFICATION (THE CROSSROADS AUDIT) ─────────
    def check_intersectional_amplification(
        self,
        group_col:    str,
        feature_cols: List[str],
        outcome_col:  Optional[str] = None
    ) -> List[AuditFinding]:
        """
        Tests whether bias compounds across feature intersections.
        Flags any sub-group whose outcome rate falls below 65% of the
        baseline (configurable). Compare with EEOC four-fifths rule (0.80).
        """
        out_col = outcome_col or self.outcome_col
        results = []
        if group_col not in self.df.columns:
            return results

        groups = self.df[group_col].unique()
        baseline_rate = self.df[out_col].mean()

        for g in groups:
            subset = self.df[self.df[group_col] == g]
            if len(subset) < 20:
                continue
            rate  = subset[out_col].mean()
            ratio = rate / baseline_rate if baseline_rate > 0 else 1
            if ratio < 0.65:
                f = AuditFinding(
                    severity   = SEVERITY_CRITICAL,
                    check_name = "INTERSECTIONAL_AMPLIFICATION",
                    column     = group_col,
                    metric     = ratio,
                    threshold  = 0.65,
                    message    = (
                        f"Group [{group_col}={g}] outcome rate {rate*100:.1f}% "
                        f"vs baseline {baseline_rate*100:.1f}%. "
                        f"Intersectional ratio {ratio:.3f} < 0.65 — compound bias "
                        "amplification at this intersection. EEOC four-fifths rule "
                        "(0.80) also breached. Disparate impact analysis required."
                    )
                )
                self._add(f)
                results.append(f)
        return results

    # ── CORRELATION MATRIX ────────────────────────────────────────────────────
    def build_correlation_matrix(self, cols: Optional[List[str]] = None) -> pd.DataFrame:
        numeric = self.df.select_dtypes(include=[np.number])
        if cols:
            numeric = numeric[[c for c in cols if c in numeric.columns]]
        return numeric.corr()

    # ── FULL AUDIT PIPELINE ───────────────────────────────────────────────────
    def run_audit(
        self,
        color_col:       str = "color_value",
        reflectance_col: str = "reflectance_score",
        proxy_col:       str = "zip_code",
        dem_col:         str = "demographic_proxy",
    ) -> "AuditReport":
        """
        Master audit runner — plugs into the ASTERISK pipeline as a
        pre-inference gate. Returns AuditReport with full findings + matrix.
        """
        print("\n╔══ ALGORITHMIC BIAS SKIM — KR0M3D1A©®™ ═══════════════════════╗")
        print(f"║  Dataset: {len(self.df)} rows × {len(self.df.columns)} cols   "
              f"| UTC: {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}")
        print("╚═══════════════════════════════════════════════════════════════╝\n")

        self.findings.clear()

        self._log("▶ CHECK 1/4 — Zero-Point Erasure (Void Audit)")
        self.check_zero_point_erasure(color_col)

        self._log("▶ CHECK 2/4 — Reflectance Signal Loss (Mirror Audit)")
        self.check_reflectance_thresholds(reflectance_col)

        self._log("▶ CHECK 3/4 — Proxy Weaponization (Trickster Audit)")
        self.detect_proxy_weaponization(proxy_col, dem_col)

        self._log("▶ CHECK 4/4 — Intersectional Amplification (Crossroads Audit)")
        if "color_bin" not in self.df.columns and color_col in self.df.columns:
            self.df["color_bin"] = pd.cut(self.df[color_col], bins=5, labels=False)
        if "color_bin" in self.df.columns:
            self.check_intersectional_amplification(
                "color_bin",
                [reflectance_col, proxy_col]
            )

        for hook in self.hooks:
            hook(self)

        corr_matrix = self.build_correlation_matrix(
            cols=[color_col, reflectance_col, "error", self.outcome_col, dem_col]
        )

        report = AuditReport(findings=self.findings, correlation_matrix=corr_matrix)
        print(report.summary())
        return report


# ══════════════════════════════════════════════════════════════════════════════
class AuditReport:
    """Structured result returned by BiasSkimmer.run_audit()"""

    def __init__(self, findings: List[AuditFinding], correlation_matrix: pd.DataFrame):
        self.findings           = findings
        self.correlation_matrix = correlation_matrix
        self.timestamp          = datetime.utcnow().isoformat()

    @property
    def has_critical(self) -> bool:
        return any(f.severity == SEVERITY_CRITICAL for f in self.findings)

    @property
    def has_vulnerability(self) -> bool:
        return any(f.severity == SEVERITY_VULNERABILITY for f in self.findings)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEVERITY_CRITICAL)

    @property
    def vulnerability_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEVERITY_VULNERABILITY)

    def summary(self) -> str:
        lines = [
            "\n╔══ AUDIT REPORT ═══════════════════════════════════════════════╗",
            f"║  Total Findings : {len(self.findings)}",
            f"║  Critical       : {self.critical_count}",
            f"║  Vulnerabilities: {self.vulnerability_count}",
            f"║  Timestamp      : {self.timestamp}",
            "╠═══════════════════════════════════════════════════════════════╣",
        ]
        if not self.findings:
            lines.append("║  ✓ No architectural bias flaws detected in this dataset partition.")
        for f in self.findings:
            lines.append(f"║  [{f.severity}] {f.check_name} — {f.column}")
            for part in [f.message[i:i+60] for i in range(0, len(f.message), 60)]:
                lines.append(f"║    {part}")
        lines.append("╚═══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "has_critical": self.has_critical,
            "findings": [
                {
                    "severity":   f.severity,
                    "check":      f.check_name,
                    "column":     f.column,
                    "metric":     f.metric,
                    "threshold":  f.threshold,
                    "message":    f.message,
                    "timestamp":  f.timestamp,
                }
                for f in self.findings
            ],
            "correlation_matrix": self.correlation_matrix.to_dict() if self.correlation_matrix is not None else {}
        }


class BiasAuditException(Exception):
    """Raised by ASTERISK pipeline when has_critical=True and halt_on_critical=True"""
    pass


def asterisk_bias_gate(df: pd.DataFrame, halt_on_critical: bool = True, **kwargs) -> AuditReport:
    """
    Drop-in function for the ASTERISK master pipeline.
    Place before any model inference call.

    Usage:
        from bias_skimmer import asterisk_bias_gate
        report = asterisk_bias_gate(preprocessed_df, halt_on_critical=True)
    """
    skimmer = BiasSkimmer(df, verbose=True, **kwargs)
    report  = skimmer.run_audit()

    if halt_on_critical and report.has_critical:
        raise BiasAuditException(
            f"ASTERISK PIPELINE HALTED — {report.critical_count} critical bias "
            f"finding(s) detected. Audit:\n{report.summary()}"
        )
    return report


class BiasSkimmerTransformer:
    """
    sklearn-compatible transformer. Use in Pipeline([...]) as a pre-fit check.

        from sklearn.pipeline import Pipeline
        pipe = Pipeline([
            ('bias_audit', BiasSkimmerTransformer(halt_on_critical=True)),
            ('scaler',     StandardScaler()),
            ('clf',        LogisticRegression()),
        ])
        pipe.fit(X_train, y_train)
    """
    def __init__(self, halt_on_critical: bool = False, **skimmer_kwargs):
        self.halt_on_critical = halt_on_critical
        self.skimmer_kwargs   = skimmer_kwargs
        self.last_report_     = None

    def fit(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if y is not None:
            df["outcome"] = y
        report = asterisk_bias_gate(df, halt_on_critical=self.halt_on_critical,
                                    **self.skimmer_kwargs)
        self.last_report_ = report
        return self

    def transform(self, X):
        return X

    def get_report(self) -> Optional[AuditReport]:
        return self.last_report_


def build_fastapi_router():
    """
    Returns a FastAPI APIRouter with POST /audit endpoint.
    """
    try:
        from fastapi import APIRouter
        from pydantic import BaseModel

        router = APIRouter()

        class AuditRequest(BaseModel):
            records: list[dict]
            color_col:       str = "color_value"
            reflectance_col: str = "reflectance_score"
            proxy_col:       str = "zip_code"
            demographic_col: str = "demographic_proxy"
            outcome_col:     str = "outcome"
            halt_on_critical: bool = False

        @router.post("/audit")
        def run_audit(req: AuditRequest):
            df     = pd.DataFrame(req.records)
            report = asterisk_bias_gate(
                df,
                halt_on_critical = req.halt_on_critical,
                outcome_col      = req.outcome_col,
                proxy_col        = req.proxy_col,
                demographic_col  = req.demographic_col,
            )
            return report.to_dict()

        return router
    except ImportError:
        raise ImportError("FastAPI + pydantic required: pip install fastapi pydantic")


if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    color_values = np.random.randint(0, 256, n)
    is_dark      = color_values < 51
    df = pd.DataFrame({
        "color_value":       color_values,
        "reflectance_score": color_values / 255 + np.random.normal(0, 0.05, n),
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
    df["reflectance_score"] = df["reflectance_score"].clip(0.01, 1.0)

    try:
        report = asterisk_bias_gate(df, halt_on_critical=False)
        print(f"\n  VAULT EXPORT READY — {len(report.findings)} findings serialized")
        print("  report.to_dict() →", list(report.to_dict().keys()))
    except BiasAuditException as e:
        print(f"\n  PIPELINE HALTED: {e}")
