# Web verification

The user authorized a web dashboard matching `fifty startup` on 2026-10-22. This is a separate presentation/inference layer; no approved experiment was rerun or changed.

## Executed checks

- `export_checks.txt`: exported metrics and selection match saved CSV/JSON; 17 PNGs and 10 downloadable files match originals byte-for-byte; local HTML asset links resolve.
- `browser_results.txt`: real headless Chrome execution of `browser-qa.html`. All 4,324 Test rows across six models plus two preprocessing edge cases per model (25,956 predictions total) matched the saved Python results within USD 1e-6. Observed maximum absolute difference: USD 3.259629011154175e-9. Covered model switching, scenario editing/reset, area identity, rejected empty inputs, feature search and no page overflow at 390px.
- `engineering_tests.txt`: all 16 existing engineering/portability tests passed. Fourteen pre-existing Matplotlib/pyparsing deprecation warnings remain.
- `research_validation.txt`: existing frozen-artifact validator passed, including original dataset SHA256, metrics, conformal arithmetic and model predictions.
- `artifact_integrity.json`: all 68 pre-existing research files (all verified outputs, dataset and experiment implementation) remain byte-identical.
- `../../docs/images/dashboard-desktop.png` and `dashboard-mobile.png`: visually inspected desktop and 390px mobile layouts. The mobile screenshot uses an isolated 390px iframe, matching the layout assertion.

The in-app browser integration failed before loading the page due to a sandbox initialization error. Browser QA therefore used installed headless Google Chrome with a fresh task-specific temporary profile, not the user's normal browser profile. Two JavaScript bracket syntax errors discovered during initial QA were corrected before the passing run. Unknown-ZIP warnings in `export_checks.txt` are expected from deliberately exercising OneHotEncoder's unknown-category behavior.

## Reproduce

From the repository root:

```sh
.venv/bin/python scripts/export_dashboard.py
.venv/bin/python scripts/verify_dashboard.py
.venv/bin/python -m http.server 8765
```

Open `http://localhost:8765/verification/web/browser-qa.html` in a browser. The page displays PASS/FAIL. The fixtures are generated from the frozen saved Test predictions and inference-only edge cases; there is no model fitting. `mobile-preview.html` provides a fixed-width visual check.

The production page uses only the 100 fixed-seed Test plot observations for scatter/interactive examples; displayed research metrics and interval coverage always use the saved full-Test results. Browser edits cannot change the frozen model-selection decision. Intervals are displayed only for the calibrated M3 model. Scenario inputs are restricted to observed per-feature ranges, which does not establish joint-distribution validity or a coverage guarantee for edited scenarios.

## Files added

- `dashboard/index.html`, `styles.css`, `app.js`, `predict.js`: accessible responsive UI and deterministic frozen-model inference.
- `dashboard/data/results.json`: export of saved model parameters, metrics and fixed examples.
- `dashboard/vendor/`: Chart.js copied from the existing fifty startup dashboard with its license.
- `dashboard/figures/`, `dashboard/downloads/`: unmodified copies of verified research assets.
- `scripts/export_dashboard.py`, `scripts/verify_dashboard.py`: reproducible export and verification.
- `.github/workflows/pages.yml`: deploy the static `dashboard/` directory to GitHub Pages.
- `README.md`: web entry point, preview/export/test instructions.

The historical `outputs_verified/` files were deliberately not edited to add website documentation. Stage 4 research provenance is preserved.
