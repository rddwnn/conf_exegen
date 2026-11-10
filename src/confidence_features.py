"""Concept summaries from historical evidence only; no target outcome features."""

import math
from collections import defaultdict
from itertools import groupby

from common import fingerprint, read_json, resolve, sha256, write_json
from confidence_data import dataset_summary, history_boundary, load_attempts, partition_students

CORRECTNESS_FEATURES = [
    "concept_count_mean",
    "concept_success_rate_mean",
    "concept_unseen_fraction",
    "student_count",
    "student_success_rate",
]
CONFIDENCE_FEATURES = CORRECTNESS_FEATURES + [
    "reported_count_mean",
    "confidence_mean",
    "confidence_missing_rate_mean",
    "high_confidence_error_rate_mean",
    "low_confidence_correct_rate_mean",
]


def masked(confidence, student, attempt_id, fraction, seed):
    if confidence is None:
        return None
    threshold = int(fingerprint([seed, student, attempt_id])[:16], 16) / 2**64
    return None if threshold < fraction else confidence


class KnowledgeHistory:
    def __init__(self, settings, mask_fraction=0.0, mask_seed=42):
        self.settings = settings
        self.mask_fraction = mask_fraction
        self.mask_seed = mask_seed
        self.records = []
        self.concepts = defaultdict(list)

    def add(self, attempt):
        self.records.append(attempt)
        for concept in attempt.concept_ids:
            self.concepts[concept].append(attempt)

    def concept_summary(self, concept):
        records = self.concepts.get(concept, [])
        n = len(records)
        success = sum(r.is_correct for r in records)
        reported = [
            (
                r,
                masked(
                    r.confidence, r.student_id, r.attempt_id, self.mask_fraction, self.mask_seed
                ),
            )
            for r in records
        ]
        reported = [(r, c) for r, c in reported if c is not None]
        count = len(reported)
        return {
            "count": n,
            "success_rate": success / n if n else 0.5,
            "smoothed_success_probability": (success + 1) / (n + 2),
            "reported_count": count,
            # Zero is an explicit sentinel; reported_count/missing_rate distinguish missingness.
            "confidence_mean": sum(c for _, c in reported) / count if count else 0.0,
            "confidence_missing_rate": 1 - count / n if n else 1.0,
            "high_confidence_error_rate": sum(
                c == self.settings["high_confidence"] and not r.is_correct for r, c in reported
            )
            / count
            if count
            else 0.0,
            "low_confidence_correct_rate": sum(
                c == self.settings["low_confidence"] and r.is_correct for r, c in reported
            )
            / count
            if count
            else 0.0,
        }

    def features(self, concepts):
        summaries = [self.concept_summary(c) for c in concepts]

        def mean(key):
            return sum(s[key] for s in summaries) / len(summaries)

        n = len(self.records)
        values = {
            "concept_count_mean": mean("count"),
            "concept_success_rate_mean": mean("success_rate"),
            "concept_unseen_fraction": sum(s["count"] == 0 for s in summaries) / len(summaries),
            "student_count": n,
            "student_success_rate": sum(r.is_correct for r in self.records) / n if n else 0.5,
            "reported_count_mean": mean("reported_count"),
            "confidence_mean": mean("confidence_mean"),
            "confidence_missing_rate_mean": mean("confidence_missing_rate"),
            "high_confidence_error_rate_mean": mean("high_confidence_error_rate"),
            "low_confidence_correct_rate_mean": mean("low_confidence_correct_rate"),
        }
        return {
            "correctness": [values[k] for k in CORRECTNESS_FEATURES],
            "confidence": [values[k] for k in CONFIDENCE_FEATURES],
            "simple_probability": mean("smoothed_success_probability"),
        }


def validate_settings(cfg):
    settings = cfg["data"]
    if settings["minimum_attempts"] < 2:
        raise ValueError("minimum_attempts must be at least two")
    if not 0 < settings["history_fraction"] < 1:
        raise ValueError("history_fraction must be between zero and one")
    levels = settings["confidence_levels"]
    if (
        not levels
        or len(set(levels)) != len(levels)
        or any(not isinstance(x, int) or x <= 0 for x in levels)
    ):
        raise ValueError("Confidence levels must be distinct positive ordinal codes")
    if settings["low_confidence"] not in levels or settings["high_confidence"] not in levels:
        raise ValueError("Low/high confidence codes must be documented confidence levels")
    masks = cfg["features"]["mask_fractions"]
    if (
        0.0 not in masks
        or len(set(masks)) != len(masks)
        or any(not math.isfinite(f) or not 0 <= f <= 1 for f in masks)
    ):
        raise ValueError("mask_fractions must contain zero and distinct values in [0,1]")


def prepare_confidence(cfg):
    validate_settings(cfg)
    source = resolve(cfg["data"]["input"])
    students = load_attempts(source, cfg["data"])
    splits, excluded = partition_students(students, cfg["data"])
    destination = resolve(cfg["data"]["prepared_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    paths = {split: destination / f"{split}.jsonl" for split in ("train", "validation", "test")}
    paths["contexts"] = destination / "contexts.jsonl"
    counts = defaultdict(int)
    boundaries = {}
    handles = {
        name: path.with_suffix(path.suffix + ".part").open("w", encoding="utf-8")
        for name, path in paths.items()
    }

    def emit(name, value):
        import json

        handles[name].write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        counts[name] += 1

    def pair(record, state, variants):
        values = state.features(record.concept_ids)
        return {
            "student_id": record.student_id,
            "attempt_id": record.attempt_id,
            "question_id": record.question_id,
            "concept_ids": list(record.concept_ids),
            "y_success": record.is_correct,
            "features_correctness": values["correctness"],
            "simple_probability": values["simple_probability"],
            "features_confidence": {
                str(mask): history.features(record.concept_ids)["confidence"]
                for mask, history in variants.items()
            },
        }

    try:
        for student, split in splits.items():
            records = students[student]
            if split == "train":
                state = KnowledgeHistory(cfg["data"])
                for _, group in groupby(records, key=lambda r: r.timestamp):
                    group = list(group)
                    # Every target in a timestamp group sees only strictly earlier answers.
                    if state.records:
                        for record in group:
                            emit("train", pair(record, state, {0.0: state}))
                    for record in group:
                        state.add(record)
            else:
                cut = history_boundary(records, cfg["data"]["history_fraction"])
                boundaries[student] = {
                    "history_count": cut,
                    "target_count": len(records) - cut,
                    "last_history_time": records[cut - 1].timestamp.isoformat(),
                    "first_target_time": records[cut].timestamp.isoformat(),
                }
                variants = {
                    mask: KnowledgeHistory(cfg["data"], mask, cfg["features"]["mask_seed"])
                    for mask in cfg["features"]["mask_fractions"]
                }
                for record in records[:cut]:
                    for state in variants.values():
                        state.add(record)
                for record in records[cut:]:
                    emit(split, pair(record, variants[0.0], variants))
                emit(
                    "contexts",
                    {
                        "student_id": student,
                        "split": split,
                        "history": [r.history_record(confidence=True) for r in records[:cut]],
                        "targets": [r.target_record() for r in records[cut:]],
                    },
                )
    finally:
        for handle in handles.values():
            handle.close()
    for name, path in paths.items():
        path.with_suffix(path.suffix + ".part").replace(path)
    manifest = {
        "protocol": "confidence-prefix-v1",
        "data_sha256": sha256(source),
        "settings": {"data": cfg["data"], "features": cfg["features"]},
        "summary": dataset_summary(students),
        "splits": splits,
        "excluded_students": excluded,
        "boundaries": boundaries,
        "records": dict(counts),
        "feature_names": {"correctness": CORRECTNESS_FEATURES, "confidence": CONFIDENCE_FEATURES},
        "files": {name: sha256(path) for name, path in paths.items()},
        "notes": [
            "Training uses strictly prior timestamp groups only.",
            "Validation/test suffixes use a fixed prefix; no suffix updates.",
            "Ordinal confidence is a feature, never a probability of correctness.",
            "Question features average all linked concept summaries.",
            "Masking applies to held-out history only; fit and selection use mask 0.",
        ],
    }
    manifest["identity"] = fingerprint(manifest)
    write_json(destination / "manifest.json", manifest)
    return manifest


def verified_prepared(cfg, names):
    folder = resolve(cfg["data"]["prepared_dir"])
    manifest = read_json(folder / "manifest.json")
    if manifest["settings"] != {"data": cfg["data"], "features": cfg["features"]}:
        raise ValueError("Confidence preparation settings changed; run prepare again")
    for name in names:
        if sha256(folder / f"{name}.jsonl") != manifest["files"][name]:
            raise ValueError(f"Prepared confidence artifact modified: {name}")
    return folder, manifest
