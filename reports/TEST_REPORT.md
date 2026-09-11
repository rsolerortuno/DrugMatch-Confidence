# Verification report — 1.1.0.dev0

Reverified locally on 2026-09-11 in a new CPython 3.12 environment on Linux with `.[dev]` only. All 41 tests, lint, typing, both example predictions, all ten bundled models, five regenerated figures, dependency consistency, and wheel/source builds passed. Local README links and artifact checksums were checked.

The release checker now targets the existing historical/review bundle directories and example input; CI runs it for all ten models. The README defaults to review predictions and documents the preserved folder layout. No model was retrained and no decision gate was relaxed.

| Check | Result |
|---|---|
| Clean editable install with development extras only (`.[dev]`) | Passed; Streamlit absent, PyArrow explicitly installed |
| Earlier install with app and development extras | Passed in the preceding verification |
| Dependency consistency (`python -m pip check`) | Passed |
| Byte-code compilation of src, app, scripts and tests | Passed |
| Ruff 0.16.0 (`ruff check src tests app scripts`) | Passed |
| mypy (`mypy src`) | Passed; 29 source files |
| pytest | **41 passed**, including Arrow-backed identifier regression, interval feasibility and analytic tie-expectation checks |
| CLI help | Passed |
| Historical trametinib example prediction | Passed |
| Review trametinib example prediction | Passed |
| Load and predict using all five historical and five review bundles | Passed; training-reference inputs, functional smoke tests only |
| Regenerate all five review figures from committed CSV tables | Passed; visually inspected |
| Build wheel and source distribution (`python -m build`) | Passed |
| `git diff --check` | Passed |
| Docker build | **Not run locally: Docker is unavailable**; retained as a required CI job |
| GitHub Actions remote status | See the CI badge and the publishing PR for authoritative remote results; pending at preparation time |

## Fixes required by the clean-install check

The final dev-only check closes a missing direct test dependency: `pyarrow>=10.0.1` is now declared in `dev`. Previously, the two Arrow-backed integration cases passed only when Streamlit supplied PyArrow transitively. CI quality checks now install only `[dev]` to preserve this regression check.

The initial Python 3.11 installation resolved pandas 3.0.5 and exposed three integration-test failures: Arrow-backed model identifiers were passed directly to scikit-learn's `train_test_split`. Training and both OOF paths now pass NumPy identifier arrays, preserving the data membership and split seeds. Two existing integration tests now exercise both object and Arrow-backed identifiers.

The recorded review-model environment uses XGBoost 3.4.1 and SHAP 0.52.0, which require Python 3.12. CI, Docker and package metadata now agree on that minimum. NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, XGBoost 3.4.1 and SHAP 0.52.0 are pinned to the saved training provenance. Dependency versions from the final dev-only check are recorded in `reports/ci_environment.txt`.

Historical XGBoost bundles still emit a serialization compatibility warning but load and predict successfully. SHAP emits matplotlib deprecation warnings. Neither warning was suppressed or converted into a false test pass. Historical models and results have not been overwritten.

## Reproduce locally

Use Python 3.12, create a virtual environment, and run:

```bash
python -m pip install -e '.[dev]'
make check
python -m build
# Requires a running Docker installation:
docker build -t drugmatch-confidence:ci .
```

`make check` runs compilation, lint, typing, tests, CLI help, historical/review example predictions and review-figure generation. CI runs on every branch push, pull request and manual dispatch. A successful local check cannot guarantee the availability of GitHub runners, package registries or the Docker registry.

These are software checks, not new scientific validation. Review bundles remain unreviewed and the conservative policy may abstain. No scientific training run was repeated to regenerate the supplied evidence.

## Follow-up scientific audit

Added per-fold interval feasibility, radius sensitivity, analytic boundary-tie ranges/expectations, a paired AUROC figure highlighting gemcitabine, and explicit outcome-stratification provenance. Future OOF runs can save calibration residuals; the analysis verifies their IDs and their reproduction of the 90% radius before computing nominal coverage feasibility. Original residuals were not present and were not fabricated. The categorical API policy was not relaxed.
