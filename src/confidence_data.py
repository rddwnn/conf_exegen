"""Explicit confidence coding, normalized records, and student/time-safe partitions."""

import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from common import resolve, sha256, write_json

DBE_API = (
    "https://dataverse.ada.edu.au/api/datasets/:persistentId/?persistentId=doi:10.26193/6DZWOH"
)
DBE_FILES = {
    "transaction.csv",
    "questions.csv",
    "question_choices.csv",
    "kcs.csv",
    "question_kc_relationships.csv",
    "kc_relationships.csv",
}
COLUMNS = [
    "attempt_id",
    "student_id",
    "question_id",
    "timestamp",
    "is_correct",
    "confidence",
    "concept_ids",
]


@dataclass(frozen=True)
class Attempt:
    attempt_id: str
    student_id: str
    question_id: str
    timestamp: datetime
    is_correct: int
    confidence: int | None
    concept_ids: tuple[str, ...]

    def history_record(self, confidence=False):
        record = {
            "attempt_id": self.attempt_id,
            "question_id": self.question_id,
            "timestamp": self.timestamp.isoformat(),
            "is_correct": self.is_correct,
            "concept_ids": list(self.concept_ids),
        }
        if confidence:
            record["confidence"] = self.confidence
        return record

    def target_record(self):
        # A strict allowlist prevents outcome/confidence/hints/time leakage to LLM inputs.
        return {
            "attempt_id": self.attempt_id,
            "question_id": self.question_id,
            "timestamp": self.timestamp.isoformat(),
            "concept_ids": list(self.concept_ids),
        }


def confidence_value(value, settings):
    if value is None or str(value).strip() in settings["missing_confidence"]:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown confidence code: {value!r}") from exc
    if number == 0 and "0" in settings["missing_confidence"]:
        return None
    if not math.isfinite(number) or number not in settings["confidence_levels"]:
        raise ValueError(f"Unknown confidence code: {value!r}")
    return int(number)


def load_attempts(path, settings):
    students = defaultdict(list)
    seen = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing normalized columns: {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Malformed CSV row")
                if any(not row[key].strip() for key in ("attempt_id", "student_id", "question_id")):
                    raise ValueError("Missing record identifier")
                key = (row["student_id"], row["attempt_id"])
                if key in seen:
                    raise ValueError(f"Duplicate student/attempt key: {key}")
                seen.add(key)
                concepts = json.loads(row["concept_ids"])
                if (
                    not isinstance(concepts, list)
                    or not concepts
                    or any(not isinstance(c, str) or not c.strip() for c in concepts)
                    or len(set(concepts)) != len(concepts)
                ):
                    raise ValueError("concept_ids must be a nonempty JSON list of unique strings")
                if row["is_correct"] not in ("0", "1"):
                    raise ValueError("is_correct must be 0 or 1")
                timestamp = datetime.fromisoformat(row["timestamp"])
                # Naive dataset timestamps use one explicit shared UTC interpretation.
                timestamp = (
                    timestamp.replace(tzinfo=timezone.utc)
                    if timestamp.tzinfo is None
                    else timestamp.astimezone(timezone.utc)
                )
                confidence = confidence_value(row["confidence"], settings)
                students[row["student_id"]].append(
                    Attempt(
                        row["attempt_id"],
                        row["student_id"],
                        row["question_id"],
                        timestamp,
                        int(row["is_correct"]),
                        confidence,
                        tuple(concepts),
                    )
                )
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{line}: {exc}") from exc
    for records in students.values():
        records.sort(key=lambda r: (r.timestamp, r.attempt_id))
    if not students:
        raise ValueError("No attempts in normalized input")
    return dict(students)


def history_boundary(records, fraction):
    if not 0 < fraction < 1:
        raise ValueError("history_fraction must be between zero and one")
    cut = int(len(records) * fraction)
    if cut == 0 or cut == len(records):
        return 0
    # Move the whole boundary timestamp group into the held-out suffix.
    while cut > 0 and records[cut - 1].timestamp == records[cut].timestamp:
        cut -= 1
    return cut


def partition_students(students, settings):
    eligible, excluded = [], {}
    for student in sorted(students):
        records = students[student]
        if len(records) < settings["minimum_attempts"]:
            excluded[student] = "too_few_attempts"
        elif not history_boundary(records, settings["history_fraction"]):
            excluded[student] = "no_nonempty_time_separated_history"
        else:
            eligible.append(student)
    if len(eligible) < 5:
        raise ValueError("Need at least five eligible students for a 60/20/20 split")
    random.Random(settings["split_seed"]).shuffle(eligible)
    first, second = int(len(eligible) * 0.6), int(len(eligible) * 0.8)
    splits = {
        student: "train" if i < first else "validation" if i < second else "test"
        for i, student in enumerate(eligible)
    }
    return splits, excluded


def inspect_dbe(cfg):
    folder = resolve(cfg["data"]["raw_dir"])
    report = {}
    for path in sorted(folder.glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            count = 0
            for _ in reader:
                count += 1
        report[path.name] = {"columns": reader.fieldnames, "records": count, "sha256": sha256(path)}
    if not report:
        raise ValueError(f"No CSV files at {folder}; obtain the DBE-KT22 archive first")
    write_json(folder / "inspection.json", report)
    return report


def import_dbe(cfg):
    mapping = cfg["import"]
    keys = [
        "attempt_id",
        "student_id",
        "question_id",
        "timestamp",
        "is_correct",
        "confidence",
        "relation_question_id",
        "relation_concept_id",
    ]
    if any(not mapping[k] for k in keys):
        raise ValueError(
            "Run inspect, then configure all import column mappings; DBE headers are not assumed"
        )
    folder = resolve(cfg["data"]["raw_dir"])
    concepts = defaultdict(list)
    relation_path = folder / mapping["relations_file"]
    with relation_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {mapping["relation_question_id"], mapping["relation_concept_id"]}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Relation columns not found: {required}")
        for row in reader:
            question, concept = (
                row[mapping["relation_question_id"]],
                row[mapping["relation_concept_id"]],
            )
            if not question or not concept:
                raise ValueError("Empty question/concept relationship")
            if concept not in concepts[question]:
                concepts[question].append(concept)
    target = resolve(cfg["data"]["input"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".csv.part")
    transactions = folder / mapping["transactions_file"]
    try:
        with (
            transactions.open(encoding="utf-8-sig", newline="") as source,
            temporary.open("w", encoding="utf-8", newline="") as output,
        ):
            reader = csv.DictReader(source)
            if not {mapping[k] for k in keys[:6]} <= set(reader.fieldnames or []):
                raise ValueError("Mapped transaction columns not found")
            writer = csv.DictWriter(output, fieldnames=COLUMNS)
            writer.writeheader()
            for line, row in enumerate(reader, 2):
                item = {key: row[mapping[key]] for key in keys[:6]}
                correct = mapping["correctness_values"].get(item["is_correct"])
                if correct not in (0, 1):
                    raise ValueError(f"Unmapped correctness code at transaction line {line}")
                item["is_correct"] = str(correct)
                if item["question_id"] not in concepts:
                    raise ValueError(f"Unmapped question at transaction line {line}")
                value = confidence_value(item["confidence"], cfg["data"])
                item["confidence"] = "" if value is None else str(value)
                item["concept_ids"] = json.dumps(concepts[item["question_id"]], ensure_ascii=False)
                writer.writerow(item)
        load_attempts(temporary, cfg["data"])
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(
        target.with_suffix(".source.json"),
        {
            "doi": "10.26193/6DZWOH",
            "mapping": mapping,
            "inputs": {str(path): sha256(path) for path in (transactions, relation_path)},
            "normalized_sha256": sha256(target),
            "note": "Column mappings supplied after inspecting the actual archive; no headers guessed.",
        },
    )


def fetch_dbe(cfg):
    """Use the official public archive; fail explicitly on HTML/bot challenges."""
    folder = resolve(cfg["data"]["raw_dir"])
    with urlopen(DBE_API, timeout=30) as response:
        try:
            metadata = json.load(response)
        except ValueError as exc:
            raise ValueError(
                "ADA returned HTML/access protection instead of metadata. Obtain the archive "
                "from doi:10.26193/6DZWOH in a browser, place CSVs in raw_dir, then run inspect."
            ) from exc
    if metadata.get("status") != "OK":
        raise ValueError("ADA metadata request failed")
    files = metadata["data"]["latestVersion"]["files"]
    selected = [f for f in files if f["label"] in DBE_FILES]
    if {f["label"] for f in selected} != DBE_FILES:
        raise ValueError(
            "Official file inventory differs from documented DBE tables; inspect metadata"
        )
    from downloads import download

    for entry in selected:
        if entry.get("restricted"):
            raise ValueError(f"Archive requires access for {entry['label']}; no automatic bypass")
        checksum = entry["dataFile"].get("checksum", {})
        expected = checksum.get("value") if checksum.get("type", "").upper() == "SHA-256" else None
        target = folder / entry["label"]
        download(
            f"https://dataverse.ada.edu.au/api/access/datafile/{quote(str(entry['dataFile']['id']))}?format=original",
            target,
            expected,
        )
        with target.open("rb") as handle:
            prefix = handle.read(1024).lstrip(b"\xef\xbb\xbf \r\n\t").lower()
        if prefix.startswith((b"<html", b"<!doctype html")):
            target.unlink()
            target.with_name(target.name + ".download.json").unlink(missing_ok=True)
            raise ValueError(
                f"Archive returned access-protection HTML for {entry['label']}, not CSV"
            )
    write_json(folder / "archive-metadata.json", metadata)


def dataset_summary(students):
    counts = Counter(
        "missing" if r.confidence is None else str(r.confidence)
        for records in students.values()
        for r in records
    )
    return {
        "students": len(students),
        "attempts": sum(map(len, students.values())),
        "confidence_codes": dict(counts),
        "questions": len({r.question_id for records in students.values() for r in records}),
        "concepts": len(
            {c for records in students.values() for r in records for c in r.concept_ids}
        ),
    }
