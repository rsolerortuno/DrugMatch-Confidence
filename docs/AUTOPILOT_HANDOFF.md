# Autopilot handoff

Version 1.0.0 is a frozen real-data release. Future automated work must preserve:

1. PRISM grouped splits and stored test IDs.
2. GDSC2 as an untouched external dataset.
3. Training-only feature selection.
4. Separate damaging and hotspot mutation features.
5. Honest validation labels in model bundles.
6. The preclinical-only boundary.

Recommended next issues:

- add pathway-score features while comparing against the frozen v1.0.0 models;
- investigate palbociclib transfer failure without touching the strict GDSC2 outcomes during tuning;
- add organoid evaluation as a separate domain;
- improve missing-feature handling and assay-panel input templates;
- add model-registry metadata and automated release checks.

Do not silently replace weak models or rewrite their metrics. New models require a new version and side-by-side comparison.
