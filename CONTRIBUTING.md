# Contributing

1. Create a focused branch.
2. Keep production logic in `src/drugmatch/`.
3. Add success and failure tests.
4. Do not use the internal test set or GDSC2 for tuning.
5. Do not commit public raw datasets or secrets.
6. Run:

```bash
pytest -q
ruff check src tests app scripts
mypy src/drugmatch
```

7. Update documentation for user-visible behaviour.
8. Preserve the preclinical-only limitation.
