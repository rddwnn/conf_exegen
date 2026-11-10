"""CPU logistic error-risk models, frozen validation selection, and JSON inference."""

import importlib.metadata
import math
import time

from common import read_json, resolve, sha256, write_json
from confidence_features import CONFIDENCE_FEATURES, CORRECTNESS_FEATURES, verified_prepared
from prepare import json_lines

VARIANTS = ("correctness_only", "structured_without_confidence", "structured_confidence")


def success_probability(model, values):
    if len(values) != len(model["feature_names"]):
        raise ValueError("Feature width differs from fitted model")
    score = model["intercept"] + sum(
        coefficient * (value - mean) / scale
        for coefficient, value, mean, scale in zip(
            model["coefficients"], values, model["mean"], model["scale"], strict=True
        )
    )
    # Model predicts error risk. Export correctness probabilities for Brier/AUROC.
    if score >= 0:
        z = math.exp(-score)
        return z / (1 + z)
    z = math.exp(score)
    return 1 / (1 + z)


def feature_values(pair, variant, mask=0.0):
    return (
        pair["features_correctness"]
        if variant == "structured_without_confidence"
        else pair["features_confidence"][str(float(mask))]
    )


def fit_models(cfg):
    import warnings

    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    folder, manifest = verified_prepared(cfg, ["train", "validation"])
    train, validation = (
        list(json_lines(folder / "train.jsonl")),
        list(json_lines(folder / "validation.jsonl")),
    )
    if not train or not validation:
        raise ValueError("Training and validation pairs must both be nonempty")
    y = np.array([1 - row["y_success"] for row in train])
    if len(set(y.tolist())) != 2:
        raise ValueError(
            "Training pairs need both correct and incorrect answers for logistic regression"
        )
    candidates = sorted(set(cfg["model"]["c_values"]))
    if not candidates or any(not math.isfinite(c) or c <= 0 for c in candidates):
        raise ValueError("c_values must be finite positive regularization strengths")
    destination = resolve(cfg["model"]["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    models = {}
    started = time.perf_counter()
    for variant in VARIANTS[1:]:
        x = np.array([feature_values(row, variant) for row in train])
        vx = np.array([feature_values(row, variant) for row in validation])
        scores = []
        best = None
        for strength in candidates:
            pipeline = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=strength,
                    penalty="l2",
                    solver="lbfgs",
                    class_weight=None,
                    max_iter=cfg["model"]["max_iterations"],
                    random_state=42,
                ),
            )
            try:
                with threadpool_limits(limits=3), warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    pipeline.fit(x, y)
                    predictions = 1 - pipeline.predict_proba(vx)[:, 1]
            except ConvergenceWarning as exc:
                raise RuntimeError(
                    f"{variant} did not converge at C={strength}; no model selected"
                ) from exc
            brier = float(
                np.mean((predictions - np.array([r["y_success"] for r in validation])) ** 2)
            )
            scores.append({"C": strength, "validation_brier": brier})
            if best is None or brier < best[0]:
                best = (brier, strength, pipeline)
        _, strength, pipeline = best
        scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
        model = {
            "schema_version": 1,
            "variant": variant,
            "target": "error",
            "C": strength,
            "feature_names": CORRECTNESS_FEATURES
            if variant == "structured_without_confidence"
            else CONFIDENCE_FEATURES,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coefficients": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]),
            "prepared_identity": manifest["identity"],
            "selection": scores,
            "train_students": sorted({row["student_id"] for row in train}),
            "validation_students": sorted({row["student_id"] for row in validation}),
            "train_pairs": len(train),
            "validation_pairs": len(validation),
            "sklearn_version": importlib.metadata.version("scikit-learn"),
            "notes": [
                "Scaler and coefficients fit on training students only.",
                "C selected by unmasked validation Brier; smallest C wins exact ties.",
                "No test rows read or used during fitting or selection.",
                "Models are not refit on validation data.",
            ],
        }
        write_json(destination / (variant + ".json"), model)
        models[variant] = model
    write_json(
        destination / "fit.json",
        {
            "prepared_identity": manifest["identity"],
            "settings": cfg["model"],
            "elapsed_seconds": time.perf_counter() - started,
            "inputs": {name: manifest["files"][name] for name in ("train", "validation")},
            "models": {name: sha256(destination / (name + ".json")) for name in models},
            "baseline": "correctness_only uses mean per-concept Beta(1,1) posterior success probability",
        },
    )
    return {
        name: {"C": model["C"], "selection": model["selection"]} for name, model in models.items()
    }


def load_models(cfg, identity):
    folder = resolve(cfg["model"]["output_dir"])
    fit = read_json(folder / "fit.json")
    if fit["prepared_identity"] != identity:
        raise ValueError("Models were fitted to a different confidence preparation")
    models = {}
    for variant, expected in fit["models"].items():
        path = folder / (variant + ".json")
        if sha256(path) != expected:
            raise ValueError("Model file changed after fitting")
        models[variant] = read_json(path)
    return models


def predict_models(cfg, split="test"):
    import csv

    if split not in ("validation", "test"):
        raise ValueError("Predict only validation or test splits")
    folder, manifest = verified_prepared(cfg, [split])
    models = load_models(cfg, manifest["identity"])
    output = resolve(cfg["evaluation"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{split}-predictions.csv"
    temporary = path.with_suffix(".csv.part")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["method", "mask_fraction", "student_id", "attempt_id", "p_success"]
        )
        writer.writeheader()
        for pair in json_lines(folder / f"{split}.jsonl"):
            for variant in VARIANTS:
                for mask in cfg["features"]["mask_fractions"]:
                    probability = (
                        pair["simple_probability"]
                        if variant == "correctness_only"
                        else success_probability(
                            models[variant], feature_values(pair, variant, mask)
                        )
                    )
                    writer.writerow(
                        {
                            "method": variant,
                            "mask_fraction": mask,
                            "student_id": pair["student_id"],
                            "attempt_id": pair["attempt_id"],
                            "p_success": probability,
                        }
                    )
                    count += 1
    temporary.replace(path)
    write_json(
        output / f"{split}-prediction-manifest.json",
        {
            "prepared_identity": manifest["identity"],
            "split": split,
            "predictions": sha256(path),
            "models": {
                name: sha256(resolve(cfg["model"]["output_dir"]) / (name + ".json"))
                for name in models
            },
            "rows": count,
            "note": "No validation/test outcome or confidence is an input feature.",
        },
    )
    return path
