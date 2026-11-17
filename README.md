# Incremental visual embedding recognition

Python 3.11, NumPy and SciPy are available. The vectors are small fixed visual features; no image backbone or remote model is needed. Run `python workflow.py fixtures/concepts.json` to see the ordinary ridge/centroid starting behavior. All starting functions run and their current behavior is described in docstrings.

`fit_features`/`fit_targets` are paired supervised representation adaptation data. `anchors` are old visual directions that must stay fixed under the requested extension. `support` supplies class-labeled prototype fitting vectors; `calibration` is a separate class-labeled calibration collection, never part of representation training. `queries` have no labels. Row-vector convention is used throughout.

The task text defines the new statistical and numerical contracts. Existing interfaces may be extended, and the internal implementation may be reorganized. Do not assume these fixtures enumerate every legal input.
