# DrugMatch-Confidence 1.1.0.dev0 — evidence review

Matched evaluation, independent calibration, explicit abstention, and five new review figures are now included in the existing repository layout. Review bundles remain unreviewed; this is a research software update, not a claim of prospective validation.

See [the current README](README.md), [evidence review](docs/PIERRE_FABRE_REVIEW.md), [MAPK experiment plan](docs/MAPK_EXPERIMENT_PLAN.md), and [verification report](reports/TEST_REPORT.md).

---

# 1.1.0.dev0 — September scientific review

Independent calibration partitions, exact conformal intervals and explicit abstention. Added matched OOF comparisons for three model families, five retrained review bundles, retrospective GDSC2 re-evaluation, and a MAPK experimental-budget case. The linear baseline matches or exceeds the strongest XGBoost examples; strict categorical acceptance is not demonstrated. New bundles remain unreviewed. Historical results remain available and are labelled separately.

Validation: 35 unit/integration tests pass; Ruff and whitespace checks pass. Source hashes, fold memberships and all computed results are included. See `docs/PIERRE_FABRE_REVIEW.md`.

# DrugMatch-Confidence v1.0.1

This patch release stabilizes the initial public release.

## Fixed

- Fixed compatibility with pandas Arrow-backed indexes during grouped train,
  validation and test splitting.
- Fixed Ruff lint and import-order issues.
- Fixed static typing errors reported by mypy.
- Removed an unused variable and obsolete type suppressions.
- Improved compatibility with current pandas, scikit-learn and Ruff versions.

## Validation

- Ruff: passed.
- Mypy: passed.
- Pytest: 22 tests passed.
- Python compilation: passed.
- CLI smoke test: passed.
- GitHub Actions: passed.

The trained models, scientific results and intended preclinical use remain
unchanged from v1.0.0.
