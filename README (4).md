# Bias Skimmer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-21%20passing-brightgreen.svg)](tests/)
[![Version](https://img.shields.io/badge/version-1.0.1-green.svg)](#)

> The reference open-source implementation of the four-audit framework from *The Materisk Asterisk: A Field Guide to Algorithmic Civil Rights* by Edward Craig Callender (KR0M3D1A©®™ Digital ARC, 2026).

This repository ships **three companion tools** that together make up the public-facing layer of the project:

1. **`bias_skimmer.py`** — Python audit tool for developers and data scientists
2. **`demo.html`** — Browser-based live demo of the four audits
3. **`assessment.html`** — Individual decision-assessment wizard for people facing automated denials

A landing page at `index.html` ties them together.

---

## What This Is

The Bias Skimmer runs four audits against a pandas DataFrame and flags algorithmic decision systems that exhibit specific failure modes documented in the book:

| Audit | Failure Mode | Legal Hook |
|---|---|---|
| **Void Audit** | Zero/null values treated as proxies for negative outcomes | EEOC four-fifths rule, 29 CFR §1607 |
| **Mirror Audit** | Disproportionate error rates in worst-case data subsets | Stratified validation requirements |
| **Trickster Audit** | Features acting as 1:1 stand-ins for protected characteristics | Title VII, FHA, ECOA Regulation B |
| **Crossroads Audit** | Compound disparities at intersections of features | EEOC four-fifths rule, disparate impact |

Each finding carries its severity (`CRITICAL`, `VULNERABILITY`, `PROXIED`, `NOMINAL`), the measured metric, the threshold, and a message citing the relevant legal framework.

---

## What This Is NOT

Honest scope limitations — important before you adopt this tool:

- **The Skimmer audits datasets, not companies.** You cannot type in "Bank of America" and get a bias score. The data needed for that audit is private until a regulator (CFPB, EEOC, HUD) or a court orders disclosure. The Skimmer works against datasets you already possess, or against public datasets like HMDA mortgage data.
- **The Skimmer is not 100% accurate.** It has documented limitations (see Section 13.7 of the book and the `# Known Limitations` block in `bias_skimmer.py`). Pearson correlation is linear-only; there's no statistical-significance testing yet; no domain-specific calibration; no protected-class imputation. These are roadmap items for v1.1, not hidden defects. **Use the Skimmer as a fast first-pass screen, not a finishing certificate.**
- **The Assessment Wizard is not legal advice.** It identifies red flags and points to the correct agency. The work of the law is done by your complaint, by the regulator's investigation, and by the attorney you may consult.

For deeper statistical fairness machinery, supplement the Skimmer with [IBM AI Fairness 360](https://github.com/Trusted-AI/AIF360) or [Microsoft Fairlearn](https://github.com/fairlearn/fairlearn).

---

## Installation

```bash
pip install pandas numpy
```

Optional, depending on which integration you want:

```bash
pip install scikit-learn        # for BiasSkimmerTransformer
pip install fastapi pydantic    # for build_fastapi_router()
```

Drop `bias_skimmer.py` into your project directory and import:

```python
from bias_skimmer import BiasSkimmer, asterisk_bias_gate
```

---

## Quick Start

```python
import pandas as pd
from bias_skimmer import BiasSkimmer

df = pd.read_csv("your_decision_data.csv")
skimmer = BiasSkimmer(df, outcome_col="approved", verbose=True)
report = skimmer.run_audit(
    color_col       = "skin_tone_index",
    reflectance_col = "sensor_confidence",
    proxy_col       = "zip_code",
    dem_col         = "neighborhood_minority_share",
)

if report.has_critical:
    print(f"Found {report.critical_count} critical bias finding(s):")
    for f in report.findings:
        print(f"  [{f.severity}] {f.check_name}: {f.message}")
```

Or run the built-in demo:

```bash
python bias_skimmer.py
```

---

## Three Integration Patterns

### 1. Pipeline Gate
Raises `BiasAuditException` if any audit flags `CRITICAL`. Use at the entry point of an inference pipeline:

```python
from bias_skimmer import asterisk_bias_gate, BiasAuditException
try:
    asterisk_bias_gate(preprocessed_df, halt_on_critical=True)
    # proceed with inference
except BiasAuditException as e:
    halt_pipeline()
```

### 2. sklearn Transformer
Slots into any `Pipeline` as a pre-fit check:

```python
from sklearn.pipeline import Pipeline
from bias_skimmer import BiasSkimmerTransformer

pipe = Pipeline([
    ('bias_audit', BiasSkimmerTransformer(halt_on_critical=True)),
    # ... your model steps
])
pipe.fit(X_train, y_train)
```

### 3. FastAPI Router
Exposes audits as an HTTP service:

```python
from fastapi import FastAPI
from bias_skimmer import build_fastapi_router

app = FastAPI()
app.include_router(build_fastapi_router(), prefix="/bias")
```

---

## Configuration

All thresholds are configurable in `BiasSkimmer()` kwargs:

| Parameter | Default | Notes |
|---|---|---|
| `zero_threshold` | 0.70 | Set to **0.80** for EEOC four-fifths rule |
| `reflectance_pct` | 0.10 | Bottom decile by default |
| `error_rate_ceiling` | 0.25 | Lower (e.g. 0.05) for high-stakes systems |
| `proxy_corr_ceiling` | 0.85 | Lower for stricter screening |

---

## Verified Tests

`tests/test_bias_skimmer.py` includes **21 test cases** covering all four audits, data shapes, and integration paths. Run them:

```bash
python3 tests/test_bias_skimmer.py
```

Expected output ends with:
```
PASSED : 21
FAILED : 0
TOTAL  : 21
✓ ALL TESTS PASSED — Bias Skimmer v1.0.1 verified clean.
```

A JavaScript port of the same 21 tests runs automatically in your browser when you open `demo.html`.

---

## Browser-Based Demo and Assessment

Two HTML tools require no installation — open in any browser:

### `demo.html` — Live Skimmer Demo
- Runs all 21 tests automatically on page load
- Three synthetic datasets (biased / clean / proxy-weaponized) to demonstrate each audit
- Same logic, same thresholds, same legal-citation messages as the Python version
- All client-side JavaScript; no data sent anywhere

### `assessment.html` — Individual Decision Assessment Wizard
- 6-step interactive wizard for people who have been denied credit, housing, employment, insurance, or healthcare
- Identifies red flags in the user's situation
- Names the correct federal/state enforcement agency
- Generates two pre-filled letter templates: an adverse-action-notice request and a discrimination complaint
- Everything stays on the user's device — never transmitted

---

## The Book

The Skimmer is the reference tool documented in:

> Callender, E.C. (2026). *The Materisk Asterisk: The Algorithm Bias — A Field Guide to Algorithmic Civil Rights.* KR0M3D1A©®™ Digital ARC.

- **Chapter 13** walks the source code line by line
- **Chapter 14** runs an end-to-end audit on a public dataset
- **Chapter 15** covers reading the resulting metrics with statistical care and legal precision
- **Appendix B** reproduces the full source

Available as paperback, hardcover, and ebook through Amazon, Apple Books, Google Play Books, and Kindle.

---

## Repository Layout

```
bias-skimmer/
├── README.md                  ← this file
├── LICENSE                    ← MIT license
├── requirements.txt           ← Python dependencies
├── .gitignore                 ← Python standard
├── bias_skimmer.py            ← the audit tool (489 lines)
├── index.html                 ← project landing page
├── demo.html                  ← live browser demo of the Skimmer
├── assessment.html            ← individual decision-assessment wizard
└── tests/
    └── test_bias_skimmer.py   ← 21-case Python test suite
```

---

## Contributing

Issues, pull requests, and substantive corrections welcome.

If you find a use of the Skimmer that surfaces a real disparity in a real production system — particularly one that leads to remediation — the author would like to hear about it. Confidentiality respected.

Roadmap for v1.1 (documented in Chapter 13.7 of the book):
- Mutual-information and conditional-independence tests in the Trickster Audit
- Bootstrap confidence intervals and significance testing on findings
- Domain-specific Mirror Audit variants (Fitzpatrick, dialect, age)
- Protected-class imputation via Bayesian Improved Surname Geocoding (BISG)

---

## License

[MIT License](LICENSE) — free for any use, commercial or non-commercial. The author requests attribution where the Skimmer is built into commercial bias-auditing services.

---

## Legal Disclaimer

The Bias Skimmer and Assessment Wizard are screening and guidance tools. They do not constitute legal advice. Findings produced by these tools are statistical observations or red-flag indicators, not legal conclusions; whether a given finding constitutes actionable disparate impact under any particular statute depends on facts and law beyond what these tools measure. Users facing potential litigation or regulatory exposure should consult qualified counsel.

---

## Author and Contact

**Edward Craig Callender**
Founder, KR0M3D1A©®™ Digital ARC
[ecc@cyberfaq.net](mailto:ecc@cyberfaq.net) · [cyberfaq.net](https://cyberfaq.net)

---

*The asterisk marks the place where the official record stopped and the human story continued.*
