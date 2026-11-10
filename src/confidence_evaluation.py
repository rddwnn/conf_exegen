"""Future-answer probability evaluation against sealed held-out outcomes."""

import csv
import math
import random
from collections import defaultdict

from common import resolve, sha256, write_json
from confidence_features import verified_prepared
from prepare import json_lines


def probability_metrics(pairs, bins=10):
    from sklearn.metrics import roc_auc_score

    if bins < 1:
        raise ValueError("calibration_bins must be positive")
    if not pairs:
        return {"count": 0, "brier": None, "auroc": None, "ece": None, "calibration": []}
    groups = [[] for _ in range(bins)]
    for outcome, probability in pairs:
        groups[min(int(probability * bins), bins - 1)].append((outcome, probability))
    calibration = []
    ece = 0.0
    for index, values in enumerate(groups):
        count = len(values)
        predicted = sum(p for _, p in values) / count if count else None
        observed = sum(y for y, _ in values) / count if count else None
        if count:
            ece += count / len(pairs) * abs(predicted - observed)
        calibration.append(
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "mean_probability": predicted,
                "accuracy": observed,
            }
        )
    y, p = zip(*pairs)
    return {
        "count": len(pairs),
        "brier": sum((pred - target) ** 2 for target, pred in pairs) / len(pairs),
        "auroc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "ece": ece,
        "calibration": calibration,
    }


def paired_brier(left, right, truth, replicates=2000, seed=42):
    if replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    students = defaultdict(lambda: [0.0, 0])
    for key in sorted(left.keys() & right.keys()):
        target = truth[key]
        students[key[0]][0] += (left[key] - target) ** 2 - (right[key] - target) ** 2
        students[key[0]][1] += 1
    if not students:
        return {"students": 0, "targets": 0, "difference": None, "ci95": None}
    values = list(students.values())
    result = {
        "students": len(values),
        "targets": sum(n for _, n in values),
        "difference": sum(s for s, _ in values) / sum(n for _, n in values),
        "ci95": None,
    }
    if len(values) > 1:
        rng = random.Random(seed)
        boot = []
        for _ in range(replicates):
            sample = rng.choices(values, k=len(values))
            boot.append(sum(s for s, _ in sample) / sum(n for _, n in sample))
        boot.sort()
        result["ci95"] = [boot[int(0.025 * (replicates - 1))], boot[int(0.975 * (replicates - 1))]]
    return result


def evaluate_predictions(cfg, predictions, split="test", compare=None):
    if split not in ("validation", "test"):
        raise ValueError("Evaluate only validation or test splits")
    folder, manifest = verified_prepared(cfg, [split])
    truth = {
        (r["student_id"], r["attempt_id"]): r["y_success"]
        for r in json_lines(folder / f"{split}.jsonl")
    }
    if not truth:
        raise ValueError("No held-out targets")
    groups, attempted = defaultdict(dict), defaultdict(set)
    invalid = defaultdict(int)
    with resolve(predictions).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"method", "mask_fraction", "student_id", "attempt_id", "p_success"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Prediction columns must include {sorted(required)}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed prediction CSV row")
            key = row["student_id"], row["attempt_id"]
            if key not in truth:
                raise ValueError(f"Prediction references an unknown {split} target: {key}")
            mask = float(row["mask_fraction"])
            if not math.isfinite(mask) or not 0 <= mask <= 1 or not row["method"].strip():
                raise ValueError("Invalid method or mask fraction")
            group = row["method"], mask
            if key in attempted[group]:
                raise ValueError(f"Duplicate prediction: {group}, {key}")
            attempted[group].add(key)
            try:
                probability = float(row["p_success"])
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("Invalid probability")
            except (TypeError, ValueError):
                invalid[group] += 1
                continue
            groups[group][key] = probability
    if not attempted:
        raise ValueError("No predictions or attempted methods in input")
    report = {}
    calibration = []
    for group in sorted(attempted):
        values = groups[group]
        metrics = probability_metrics(
            [(truth[k], p) for k, p in values.items()], cfg["evaluation"]["calibration_bins"]
        )
        calibration.extend(
            {"method": group[0], "mask_fraction": group[1], **r} for r in metrics.pop("calibration")
        )
        report[f"{group[0]}@{group[1]}"] = {
            "method": group[0],
            "mask_fraction": group[1],
            "expected_targets": len(truth),
            "attempted": len(attempted[group]),
            "invalid_outputs": invalid[group],
            "missing_outputs": len(truth) - len(attempted[group]),
            "completion_rate": len(values) / len(truth),
            **metrics,
        }
    comparisons = []
    compare = compare or ("structured_confidence", "structured_without_confidence")
    for mask in sorted({mask for _, mask in attempted}):
        a, b = (compare[0], mask), (compare[1], mask)
        if a not in attempted or b not in attempted:
            continue
        comparisons.append(
            {
                "left": compare[0],
                "right": compare[1],
                "mask_fraction": mask,
                **paired_brier(
                    groups[a],
                    groups[b],
                    truth,
                    cfg["evaluation"]["bootstrap_replicates"],
                    cfg["evaluation"]["bootstrap_seed"],
                ),
            }
        )
    output = resolve(cfg["evaluation"]["output_dir"])
    payload = {
        "prepared_identity": manifest["identity"],
        "predictions_sha256": sha256(resolve(predictions)),
        "split": split,
        "methods": report,
        "paired_brier": comparisons,
        "notes": [
            "p_success is probability of correctness, never self-reported confidence.",
            "Invalid outputs are excluded and counted, never clipped or silently imputed.",
            "Negative paired Brier difference favors the left method.",
            "Student-cluster bootstrap retains all common targets per sampled student.",
            "AUROC is undefined for single-class labels; confidence intervals need two students.",
        ],
    }
    write_json(output / f"{split}-metrics.json", payload)
    with (output / f"{split}-calibration.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "mask_fraction",
                "bin",
                "lower",
                "upper",
                "count",
                "mean_probability",
                "accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(calibration)
    lines = [
        "# Future-answer probability evaluation",
        "",
        "| Method | Mask | Valid / expected | Brier | AUROC | ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    def show(value):
        return "N/A" if value is None else f"{value:.5f}"

    for row in report.values():
        lines.append(
            f"| {row['method']} | {row['mask_fraction']} | {row['count']} / {row['expected_targets']} | "
            + " | ".join(show(row[k]) for k in ("brier", "auroc", "ece"))
            + " |"
        )
    lines += [
        "",
        "Layer-only model comparisons do not establish an improvement to full ExeGen.",
        "",
    ]
    (output / f"{split}-report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload
