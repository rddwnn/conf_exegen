"""Reconstruct absent upstream CSVs with bounded memory and explicit provenance."""

import ast
import csv
import json
import random
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

from common import fingerprint, read_json, resolve, sha256, write_json
from downloads import DATA_FILES

FIELDS = [
    "problem_id",
    "user_id",
    "content",
    "option",
    "right_answer",
    "concept_id",
    "course_name",
    "knowledge_chain",
    "is_correct",
    "explanation",
    "tokens",
    "knowledge_evidence",
    "submit_time",
    "log_id",
    "split",
]


def json_lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON at {path}:{number}") from exc


def entity_id(value, prefix):
    value = str(value)
    return value if value.startswith(prefix) else prefix + value


def answer_list(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    if not isinstance(value, str):
        return []
    for candidate in (value, value.replace('\\"', '"')):
        try:
            result = json.loads(candidate)
        except (ValueError, TypeError):
            try:
                result = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
        if isinstance(result, list):
            return [str(x) for x in result]
    return []


def write_csv(path, rows, fields=FIELDS):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prepare_data(cfg):
    settings = cfg["data"]
    raw, destination = resolve(settings["raw_dir"]), resolve(settings["prepared_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    missing = [name for name in DATA_FILES if not (raw / name).is_file()]
    if missing:
        raise ValueError(f"Missing raw files: {missing}. Run download-data first.")
    # Download manifests already contain full-file hashes; verify those on download/reuse.
    sources = {}
    for name in DATA_FILES:
        meta = raw / (name + ".download.json")
        digest = sha256(raw / name)
        sources[name] = (
            read_json(meta) if meta.exists() else {"sha256": digest, "origin": "provided locally"}
        )
        if sources[name]["sha256"] != digest:
            raise ValueError(f"Raw file no longer matches its download manifest: {name}")
    field = settings["field"]
    courses = {
        entity_id(r["course_id"], "C_"): r["course_name"]
        for r in json_lines(raw / "relations/course-field.json")
        if field in r["field"]
    }
    exercise_course = {}
    for row in json_lines(raw / "entities/course.json"):
        if row["id"] in courses:
            for resource in row["resource"]:
                if resource["resource_id"].startswith("Ex_"):
                    exercise_course.setdefault(resource["resource_id"], courses[row["id"]])
    concepts = {}
    with (raw / "relations/concept-problem.txt").open(encoding="utf-8") as handle:
        for line in handle:
            concept, problem = line.rstrip("\n").split("\t")
            if concept.endswith("_" + field):
                concepts.setdefault(problem, concept)
    stats = Counter()
    problems = {}
    for row in json_lines(raw / "entities/problem.json"):
        problem_id = entity_id(row.get("problem_id", row.get("id")), "Pm_")
        if row.get("exercise_id") not in exercise_course or problem_id not in concepts:
            continue
        answers = answer_list(row.get("answer"))
        options = row.get("option")
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except ValueError:
                options = None
        if (
            row.get("type") != 1
            or len(answers) != 1
            or not isinstance(options, dict)
            or len(options) < 2
            or answers[0] not in options
            or not isinstance(row.get("content"), str)
            or not row["content"].strip()
        ):
            stats["excluded_problem_format"] += 1
            continue
        concept = concepts[problem_id]
        course = exercise_course[row["exercise_id"]]
        problems[problem_id] = {
            "problem_id": problem_id,
            "content": row["content"],
            "option": json.dumps(options, ensure_ascii=False),
            "right_answer": json.dumps(answers, ensure_ascii=False),
            "concept_id": concept,
            "course_name": course,
            "knowledge_chain": "",
            # The active upstream code concatenates these strings without relation labels.
            "knowledge_evidence": concept + ", " + concept + course + ", ",
            "explanation": "",
            "tokens": "",
        }
    if not problems:
        raise ValueError("No usable CS single-choice problems; inspect raw schema and field filter")
    database = destination / "interactions.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA cache_size=-4096")
        db.execute("DROP TABLE IF EXISTS interactions")
        db.execute(
            "CREATE TABLE interactions (student TEXT, problem TEXT, correct INTEGER, "
            "timestamp TEXT, record TEXT, source_order INTEGER)"
        )
        batch = []
        for source_order, row in enumerate(json_lines(raw / "relations/user-problem.json")):
            stats["raw_interactions"] += 1
            if row.get("problem_id") not in problems:
                continue
            if row.get("is_correct") not in (0, 1) or not row.get("user_id"):
                stats["excluded_invalid_interaction"] += 1
                continue
            try:
                timestamp = datetime.fromisoformat(row["submit_time"]).isoformat()
            except (ValueError, TypeError, KeyError):
                stats["excluded_timestamp"] += 1
                continue
            batch.append(
                (
                    row["user_id"],
                    row["problem_id"],
                    int(row["is_correct"]),
                    timestamp,
                    str(row.get("log_id", source_order)),
                    source_order,
                )
            )
            stats["eligible_interactions"] += 1
            if len(batch) >= 5000:
                db.executemany("INSERT INTO interactions VALUES (?,?,?,?,?,?)", batch)
                db.commit()
                batch.clear()
        db.executemany("INSERT INTO interactions VALUES (?,?,?,?,?,?)", batch)
        db.execute(
            "CREATE INDEX student_time ON interactions(student,timestamp,record,source_order)"
        )
        history_length = settings["history_length"]
        students = [
            r[0]
            for r in db.execute(
                "SELECT student FROM interactions GROUP BY student HAVING count(*)>=?",
                (history_length,),
            )
        ]
        students.sort(key=lambda s: fingerprint([settings["split_seed"], s]))
        students = students[: settings["max_students"]]
        if len(students) < 5:
            raise ValueError(
                f"Only {len(students)} eligible students; need at least five for splits"
            )
        random.Random(settings["split_seed"]).shuffle(students)
        train_end, validation_end = int(len(students) * 0.6), int(len(students) * 0.8)
        splits = {
            s: "train" if i < train_end else "validation" if i < validation_end else "test"
            for i, s in enumerate(students)
        }
        histories = []
        for student in students:
            records = db.execute(
                "SELECT problem,correct,timestamp,record FROM interactions WHERE student=? "
                "ORDER BY timestamp,record,source_order LIMIT ?",
                (student, history_length),
            )
            for problem, correct, timestamp, record in records:
                histories.append(
                    {
                        **problems[problem],
                        "user_id": student,
                        "is_correct": correct,
                        "submit_time": timestamp,
                        "log_id": record,
                        "split": splits[student],
                    }
                )
    # Preserve first positive/negative examples, with a bounded bank for the local pilot.
    bank = {}
    for row in histories:
        if row["split"] == "train":
            bank.setdefault((row["concept_id"], row["is_correct"]), row)
    demonstrations = [dict(r, explanation="No explanation provided.") for r in bank.values()]
    demonstrations = demonstrations[: settings["max_demonstrations"]]
    if not demonstrations:
        raise ValueError("No training demonstrations")
    write_csv(destination / "histories.csv", histories)
    write_csv(destination / "demonstrations.csv", demonstrations)
    edges = (r for r in json_lines(raw / "prerequisites/cs.json") if r["ground_truth"] == 1)
    write_csv(
        destination / "concept_relationship_filtered.csv", edges, ["c1", "c2", "ground_truth"]
    )
    manifest = {
        "protocol": "reconstructed-single-choice-v1",
        "settings": settings,
        "sources": sources,
        "statistics": dict(stats),
        "problems": len(problems),
        "students": len(students),
        "splits": dict(Counter(splits.values())),
        "demonstrations": len(demonstrations),
        "notes": [
            "First source-order concept per problem; aggregated attempts retained as records.",
            "No 70/30 future-answer prediction in the generation experiment.",
            "knowledge_chain is empty; active upstream knowledge_evidence concatenation retained.",
        ],
        "files": {
            name: sha256(destination / name)
            for name in ("histories.csv", "demonstrations.csv", "concept_relationship_filtered.csv")
        },
    }
    write_json(destination / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "students": len(students),
                "demonstrations": len(demonstrations),
                "statistics": dict(stats),
            },
            indent=2,
        )
    )


def embed(cfg):
    import pandas as pd
    import torch
    from transformers import BertModel, BertTokenizer

    destination = resolve(cfg["data"]["prepared_dir"])
    model_dir = resolve(cfg["data"]["bert_dir"])
    if not (model_dir / "source.json").exists():
        raise ValueError("Run download-bert before embedding")
    # Remote inference configurations need no local LLM runtime parameters.
    torch.set_num_threads(cfg["model"].get("threads", 3))
    tokenizer = BertTokenizer(vocab_file=str(model_dir / "vocab.txt"))
    model = BertModel.from_pretrained(model_dir, local_files_only=True).eval().to("cpu")
    # Re-embedding resets explanation inputs; stale sessions must not label new explanations.
    for name in ("explanation-session.json", "explanation-manifest.json"):
        (destination / name).unlink(missing_ok=True)
    cache_path = destination / "embedding-cache.jsonl"
    identity = read_json(model_dir / "source.json")
    cache = {r["key"]: r["tokens"] for r in json_lines(cache_path)} if cache_path.exists() else {}
    from common import append_json

    for source, target in (
        ("histories.csv", "stuRec_1000_with_tokens.csv"),
        ("demonstrations.csv", "examples_with_explanation_with_tokens.csv"),
    ):
        frame = pd.read_csv(destination / source, keep_default_na=False)
        for index, row in frame.iterrows():
            # Same concatenation and sum-pooling as the commented upstream BERT code.
            sentence = "".join(
                str(row[k])
                for k in (
                    "content",
                    "option",
                    "right_answer",
                    "is_correct",
                    "concept_id",
                    "course_name",
                    "knowledge_chain",
                )
            )
            key = fingerprint([identity, sentence])
            if key not in cache:
                inputs = tokenizer.encode_plus(
                    sentence,
                    add_special_tokens=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                if inputs["input_ids"].shape[1] > model.config.max_position_embeddings:
                    raise ValueError(
                        f"BERT input exceeds positional limit for {row['problem_id']}; "
                        "no silent truncation is applied"
                    )
                with torch.inference_mode():
                    tokens = model(**inputs)[0].squeeze(0).sum(dim=0).tolist()
                cache[key] = tokens
                append_json(cache_path, {"key": key, "tokens": tokens})
            frame.at[index, "tokens"] = json.dumps(cache[key])
        frame.to_csv(destination / target, index=False)
    write_json(
        destination / "embedding-manifest.json",
        {
            "bert": identity,
            "pooling": "last_hidden_state.sum(dim=0)",
            "inputs": {n: sha256(destination / n) for n in ("histories.csv", "demonstrations.csv")},
            "outputs": {
                n: sha256(destination / n)
                for n in (
                    "stuRec_1000_with_tokens.csv",
                    "examples_with_explanation_with_tokens.csv",
                )
            },
        },
    )
