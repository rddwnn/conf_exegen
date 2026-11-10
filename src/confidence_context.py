"""Leakage-safe historical contexts and an optional bridge to the author agents."""

import copy
from datetime import datetime

from common import resolve, write_json
from confidence_data import Attempt
from confidence_features import KnowledgeHistory, verified_prepared
from confidence_model import load_models, success_probability
from prepare import json_lines

CONTEXT_VARIANTS = (
    "ordinary_exegen",
    "raw_confidence",
    "structured_confidence",
    "structured_without_confidence",
)


def attach_confidence(author_input, context):
    """Add evidence to the shared agent input, preserving every author prompt literal."""
    result = copy.deepcopy(author_input)
    if context is None:
        return result
    mode = context.get("mode")
    if mode == "raw_confidence":
        result["historical_confidence"] = context["values"]
    elif mode in ("structured_confidence", "structured_without_confidence"):
        result["confidence_profile"] = context
    else:
        raise ValueError(f"Unsupported confidence context: {mode}")
    return result


def context_payload(record, variant, cfg, models=None, mask_fraction=0.0):
    if variant not in CONTEXT_VARIANTS:
        raise ValueError(f"Unknown confidence variant: {variant}")
    state = KnowledgeHistory(cfg["data"], mask_fraction, cfg["features"]["mask_seed"])
    for row in record["history"]:
        state.add(
            Attempt(
                row["attempt_id"],
                record["student_id"],
                row["question_id"],
                datetime.fromisoformat(row["timestamp"]),
                row["is_correct"],
                row["confidence"],
                tuple(row["concept_ids"]),
            )
        )
    history = [r.history_record() for r in state.records]
    context = None
    if variant == "raw_confidence":
        from confidence_features import masked

        context = {
            "mode": variant,
            "values": [
                {
                    "attempt_id": r.attempt_id,
                    "confidence": masked(
                        r.confidence,
                        r.student_id,
                        r.attempt_id,
                        mask_fraction,
                        cfg["features"]["mask_seed"],
                    ),
                }
                for r in state.records
            ],
        }
    elif variant.startswith("structured_"):
        if models is None:
            raise ValueError("Structured profiles require fitted confidence models")
        profile = []
        limit = cfg["features"]["evidence_limit"]
        if limit < 1:
            raise ValueError("evidence_limit must be positive")
        for concept in sorted(state.concepts):
            values = state.features([concept])
            summary = state.concept_summary(concept)
            evidence = [r.attempt_id for r in state.concepts[concept]]
            prediction_features = (
                values["confidence"]
                if variant == "structured_confidence"
                else values["correctness"]
            )
            item = {
                "concept_id": concept,
                "error_risk": 1 - success_probability(models[variant], prediction_features),
                "observation_count": summary["count"],
                "success_rate": summary["success_rate"],
                "evidence_attempt_ids": evidence[-limit:],
                "evidence_truncated": len(evidence) > limit,
            }
            if variant == "structured_confidence":
                item["confidence"] = {
                    k: summary[k]
                    for k in (
                        "reported_count",
                        "confidence_mean",
                        "confidence_missing_rate",
                        "high_confidence_error_rate",
                        "low_confidence_correct_rate",
                    )
                }
            profile.append(item)
        profile.sort(key=lambda item: (-item["error_risk"], item["concept_id"]))
        context = {
            "mode": variant,
            "concepts": profile,
            "note": "Risks are fitted statistical estimates, not evidence of a misconception.",
        }
    # Targets never carry observed correctness, confidence, hints, or answer time.
    targets = [
        {k: row[k] for k in ("attempt_id", "question_id", "timestamp", "concept_ids")}
        for row in record["targets"]
    ]
    return {
        "student_id": record["student_id"],
        "split": record["split"],
        "variant": variant,
        "mask_fraction": mask_fraction,
        "history": history,
        "targets": targets,
        "confidence_context": context,
    }


def export_contexts(cfg, variants=None, mask_fraction=0.0):
    import json

    variants = variants or list(CONTEXT_VARIANTS)
    if set(variants) - set(CONTEXT_VARIANTS):
        raise ValueError("Unknown context variant")
    if mask_fraction not in cfg["features"]["mask_fractions"]:
        raise ValueError("Requested mask is not part of the frozen feature configuration")
    folder, manifest = verified_prepared(cfg, ["contexts"])
    models = (
        load_models(cfg, manifest["identity"])
        if any(v.startswith("structured_") for v in variants)
        else None
    )
    destination = resolve(cfg["evaluation"]["output_dir"]) / "contexts" / str(float(mask_fraction))
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for variant in variants:
        path = destination / (variant + ".jsonl")
        temporary = path.with_suffix(".jsonl.part")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in json_lines(folder / "contexts.jsonl"):
                payload = context_payload(record, variant, cfg, models, mask_fraction)
                handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)
        paths[variant] = str(path)
    write_json(
        destination / "manifest.json",
        {"prepared_identity": manifest["identity"], "paths": paths, "mask_fraction": mask_fraction},
    )
    return paths
