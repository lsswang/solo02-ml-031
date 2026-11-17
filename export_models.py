"""Export independently loadable adapter/recognizer models for a fixture.

Usage:  python export_models.py [fixture.json] [out_dir]

Writes ``adapter.json``, ``recognizer.json`` and ``results.json`` (the
structured pipeline result).  Then reloads both models in a *fresh process
state* (plain JSON, no reference to the fit data) and confirms predictions
are reproduced, demonstrating the saved model is self-sufficient.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import adapter
import recognizer
import workflow


def export(path="fixtures/concepts.json", out_dir="artifacts"):
    data = json.loads(Path(path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    a = adapter.fit(data.get("fit_features", []), data["fit_targets"],
                    data.get("anchors", []), data.get("ridge", 0.5))
    r = recognizer.build(data["support"], data["calibration"], a,
                         data.get("shrinkage", 0.2))
    predictions = recognizer.predict(data.get("queries", []), r,
                                     data.get("alpha", 0.4))

    adapter.save(a, out / "adapter.json")
    recognizer.save(r, out / "recognizer.json")

    # Structured result of the given material.
    result = workflow.run(data)
    (out / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2))

    # Reload purely from disk and reproduce predictions.
    a2 = adapter.load(out / "adapter.json")
    r2 = recognizer.load(out / "recognizer.json")
    reloaded = recognizer.predict(data.get("queries", []), r2,
                                  data.get("alpha", 0.4))
    consistent = recognizer.verify_model(r2, data["support"],
                                         data["calibration"])["is_consistent"]
    match = reloaded == predictions

    return {
        "adapter_model": str(out / "adapter.json"),
        "recognizer_model": str(out / "recognizer.json"),
        "results": str(out / "results.json"),
        "reload_predictions_match": match,
        "reload_is_consistent": consistent,
        "adapter_fingerprint": adapter.fingerprint(a2),
        "recognizer_fingerprint": r2["fingerprint"],
        "predictions": predictions,
    }


if __name__ == "__main__":
    fixture = sys.argv[1] if len(sys.argv) > 1 else "fixtures/concepts.json"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "artifacts"
    summary = export(fixture, out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
