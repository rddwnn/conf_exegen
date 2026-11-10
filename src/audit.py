"""Audit prepared data before inference, using only the Python standard library."""

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime

from common import read_json, resolve, sha256, write_json
from prepare import answer_list
from runtime import RETRIEVAL_METHODS, required_artifacts

HISTORY_COLUMNS = {
    "user_id",
    "problem_id",
    "content",
    "option",
    "right_answer",
    "concept_id",
    "is_correct",
    "submit_time",
    "split",
}


def audit_data(cfg, methods=None):
    """Report data integrity separately from availability of later inference artifacts."""
    methods = methods or cfg["experiment"]["methods"]
    required_artifacts(methods)
    prepared = resolve(cfg["data"]["prepared_dir"])
    findings = {}

    def finding(code, severity, detail):
        entry = findings.setdefault(code, {"severity": severity, "count": 0, "examples": []})
        entry["count"] += 1
        if len(entry["examples"]) < 10:
            entry["examples"].append(detail)

    students = {}
    concepts, problems = set(), set()
    totals = Counter()
    split_totals = defaultdict(Counter)
    path = prepared / "histories.csv"
    hashes = {}
    if not path.is_file():
        finding("missing_history", "error", "Run prepare to create histories.csv")
    else:
        hashes["histories.csv"] = sha256(path)
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = HISTORY_COLUMNS - set(reader.fieldnames or [])
            if missing:
                finding("missing_history_columns", "error", ", ".join(sorted(missing)))
            else:
                for line, row in enumerate(reader, 2):
                    totals["records"] += 1
                    label = f"histories.csv:{line}"
                    if None in row or any(value is None for value in row.values()):
                        finding("malformed_csv_row", "error", label)
                        continue
                    student, split = row["user_id"], row["split"]
                    if not student:
                        finding("missing_student", "error", label)
                        continue
                    if split not in ("train", "validation", "test"):
                        finding("invalid_split", "error", label)
                    state = students.setdefault(
                        student,
                        {
                            "split": split,
                            "count": 0,
                            "errors": 0,
                            "last_time": None,
                            "logs": set(),
                        },
                    )
                    if split != state["split"]:
                        finding("student_split_leakage", "error", student)
                    state["count"] += 1
                    split_totals[split]["records"] += 1
                    if row["is_correct"] not in ("0", "1"):
                        finding("invalid_correctness", "error", label)
                    elif row["is_correct"] == "0":
                        totals["errors"] += 1
                        state["errors"] += 1
                        split_totals[split]["errors"] += 1
                    try:
                        timestamp = datetime.fromisoformat(row["submit_time"])
                        if state["last_time"] is not None and timestamp < state["last_time"]:
                            finding("unordered_history", "error", label)
                        state["last_time"] = timestamp
                    except (ValueError, TypeError):
                        finding("invalid_or_incompatible_timestamp", "error", label)
                    log = row.get("log_id")
                    if log:
                        if log in state["logs"]:
                            finding("repeated_log_id", "warning", label)
                        state["logs"].add(log)
                    if not row["content"].strip() or not row["problem_id"].strip():
                        finding("missing_question", "error", label)
                    try:
                        options = json.loads(row["option"])
                    except ValueError:
                        options = None
                    answers = answer_list(row["right_answer"])
                    if (
                        not isinstance(options, dict)
                        or len(options) < 2
                        or len(answers) != 1
                        or answers[0] not in options
                    ):
                        finding("invalid_single_choice", "error", label)
                    if not row["concept_id"].strip():
                        finding("missing_concept", "error", label)
                    else:
                        concepts.add(row["concept_id"])
                    problems.add(row["problem_id"])
        if not totals["records"]:
            finding("empty_history", "error", "No history records")

    for student, state in students.items():
        split_totals[state["split"]]["students"] += 1
        if state["count"] != cfg["data"]["history_length"]:
            finding("unexpected_history_length", "error", f"{student}: {state['count']} records")
        if not state["errors"]:
            split_totals[state["split"]]["students_without_errors"] += 1
    selected_split = cfg["experiment"]["split"]
    if not split_totals[selected_split]["students"]:
        finding("empty_selected_split", "error", selected_split)

    uses_retrieval = bool(set(methods) & RETRIEVAL_METHODS)
    demonstration_count = None
    if uses_retrieval:
        bank = prepared / "demonstrations.csv"
        if not bank.is_file():
            finding("missing_training_bank", "error", "Run prepare to create demonstrations.csv")
        else:
            hashes["demonstrations.csv"] = sha256(bank)
            demonstration_count = 0
            with bank.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if not {"user_id", "concept_id", "is_correct"} <= set(reader.fieldnames or []):
                    finding("missing_bank_columns", "error", "demonstrations.csv")
                else:
                    for line, row in enumerate(reader, 2):
                        demonstration_count += 1
                        student = row.get("user_id")
                        if student not in students or students[student]["split"] != "train":
                            finding(
                                "demonstration_student_leakage",
                                "error",
                                f"demonstrations.csv:{line}",
                            )
                        if not row.get("concept_id") or row.get("is_correct") not in ("0", "1"):
                            finding("invalid_demonstration", "error", f"demonstrations.csv:{line}")
            if demonstration_count == 0:
                finding("empty_training_bank", "error", "No training demonstrations")

    manifest_path = prepared / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = read_json(manifest_path)
            if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
                raise ValueError("Missing files mapping")
            # A paper-only audit must not validate unrelated retrieval artifacts.
            for name, actual in hashes.items():
                expected = manifest["files"].get(name)
                if expected is not None and actual != expected:
                    finding("prepared_checksum_mismatch", "error", name)
        except (ValueError, TypeError, AttributeError) as exc:
            finding("invalid_preparation_manifest", "warning", str(exc))
    else:
        finding("missing_preparation_manifest", "warning", "Input hashes recorded in this audit")

    errors = sum(entry["count"] for entry in findings.values() if entry["severity"] == "error")
    availability = {}
    for method in methods:
        missing = [name for name in required_artifacts([method]) if not (prepared / name).is_file()]
        availability[method] = {"artifacts_present": not missing, "missing": missing}
    report = {
        "schema_version": 1,
        "data_valid": errors == 0,
        "error_count": errors,
        "methods": methods,
        "selected_split": selected_split,
        "hashes": hashes,
        "statistics": {
            "records": totals["records"],
            "students": len(students),
            "problems": len(problems),
            "concepts": len(concepts),
            "historical_errors": totals["errors"],
            "demonstrations": demonstration_count,
        },
        "splits": {name: dict(counts) for name, counts in sorted(split_totals.items())},
        "findings": findings,
        "inference_artifact_availability": availability,
        "notes": [
            "No model, BERT, or GPU is loaded by this audit.",
            "Artifact presence does not validate embeddings or explanation model provenance.",
            "run performs method-specific inference validation before executing jobs.",
            "Students without historical errors are retained; their Recall is undefined.",
        ],
    }
    write_json(prepared / "audit.json", report)
    lines = [
        "# Prepared data audit",
        "",
        f"Data valid: {report['data_valid']}",
        f"Blocking findings: {errors}",
        "",
        "| Split | Students | Records | Errors | Students without errors |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, values in report["splits"].items():
        lines.append(
            f"| {name} | "
            + " | ".join(
                str(values.get(k, 0))
                for k in ("students", "records", "errors", "students_without_errors")
            )
            + " |"
        )
    lines.extend(["", "## Findings", ""])
    for code, entry in findings.items():
        lines.append(f"- {entry['severity']}: {code} ({entry['count']})")
    if not findings:
        lines.append("No findings.")
    lines.extend(["", "## Inference artifacts", ""])
    for method, state in availability.items():
        lines.append(
            f"- {method}: "
            + (
                "present (run still validates provenance)"
                if not state["missing"]
                else "missing " + ", ".join(state["missing"])
            )
        )
    lines.append("")
    (prepared / "audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return report
