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
