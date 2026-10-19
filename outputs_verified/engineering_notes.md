# Engineering execution notes

- Official Kaggle public API download succeeded after network access was authorized.
- Python dependencies installed in an isolated `.venv`; `pip check` passed.
- Initial raw-coordinate LASSO with max_iter=100,000 was slow. A Development-inner-training-only diagnostic for alpha=1 confirmed a convergence warning at that iteration limit.
- The numerically equivalent Gram-precomputed solver with max_iter=1,000,000 converged on that same training fold after 149,141 iterations, without a convergence warning. Alpha grid, objective, threshold, tolerance, original USD targets and final OLS estimator remain unchanged.
- Final verified experiments use `precompute=True`, `max_iter=1,000,000`, `tol=1e-5`, seed=42. Invalid candidates, if any, are explicitly recorded in `tables/inner_cv_results.csv`.
- The first run in `../outputs/` initially continued because stop permission was declined. The user subsequently approved stopping it; it has now been terminated before completion. It is preliminary and MUST NOT be used for the research handoff. The corrected full run and authoritative artifacts are in `outputs_verified/`.
- Matplotlib/pyparsing emits third-party deprecation warnings in tests. joblib cannot query physical CPU count under the sandbox and falls back to logical CPU count. These warnings are retained in logs. BLAS threads are explicitly limited to one; no CV folds or observations were reduced.
- Final verified run completed all 30 outer model fits, six full-Development refits and 792 inner candidate evaluations, with zero invalid candidates. Runtime was 240.788352 seconds. All 13 engineering tests and independent saved-output checks passed.
- Optional `compileall` could not write Apple's system bytecode-cache directory under sandbox permissions. Normal source execution, imports and tests succeeded; this did not affect any experiment or deliverable.
- All 17 PNGs were visually inspected as a contact sheet. Dense selection and LASSO charts were reformatted into two panels and regenerated from saved values only. The statistical-model fits and frozen predictions were unchanged.
- BP p-value underflows to 0.0 in floating-point arithmetic; the diagnostics and handoff explicitly identify this as numerical underflow rather than an exact probability of zero.

- Student ID corrected by the user to `7115056123`. Entry point renamed to `7115056123_hw2.py`; README, handoff, summary and archive updated. Original pasted prompt is preserved unchanged as historical input.
