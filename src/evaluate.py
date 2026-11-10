"""Deterministic exercise parsing, concept metrics, and run/resource analysis."""

import ast
import csv
import json
import random
import re
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

from common import read_json, write_json


def _exercise(item):
    if not isinstance(item, dict):
        return None
    values = {str(k).lower(): v for k, v in item.items()}
    question = values.get("exercise", values.get("content", values.get("question")))
    options = values.get("options", values.get("option"))
    answer = values.get("answer", values.get("right_answer"))
    concept = values.get("concept", values.get("concept_id"))
    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
        except (ValueError, SyntaxError):
            return None
    if isinstance(answer, str):
        try:
            answer = ast.literal_eval(answer)
        except (ValueError, SyntaxError):
            answer = [answer]
    if (
        not isinstance(question, str)
        or not question.strip()
        or not isinstance(options, dict)
        or len(options) < 2
        or not isinstance(answer, list)
        or len(answer) != 1
        or str(answer[0]) not in options
        or not isinstance(concept, str)
        or not concept.strip()
    ):
        return None
    return {"exercise": question, "options": options, "answer": answer, "concept": concept.strip()}


def parse_exercises(text):
    if not isinstance(text, str):
        return []
    candidates = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL) + [text]
    for candidate in reversed(candidates):
        try:
            result = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(result, dict):
            result = result.get("exercises", result.get("result", result.get("list", [])))
        if isinstance(result, list):
            return [valid for item in result if (valid := _exercise(item)) is not None]
    # Paper ICL+CoT asks for plain exercises, not JSON; parse its documented fields.
    plain = text.replace("**", "")
    blocks = re.split(r"(?im)^\s*(?:\d+[.)]\s*|[-*]\s*)?Exercise\s*:\s*", plain)[1:]
    output = []
    for block in blocks:
        match = re.search(
            r"(?s)^(.*?)\s*[-*]?\s*Options\s*:\s*(.*?)\s*[-*]?\s*Answer\s*:\s*(.*?)"
            r"\s*[-*]?\s*Concept\s*:\s*([^\n]+)",
            block,
        )
        if not match:
            continue
        question, options, answer, concept = match.groups()
        options = options.strip()
        if not options.startswith("{"):
            options = "{" + options + "}"
        item = _exercise(
            {
                "exercise": question.strip(),
                "options": options,
                "answer": answer.strip(),
                "concept": concept.strip(),
            }
        )
        if item:
            output.append(item)
    return output


def concept_metrics(history, exercises):
    known = {row["concept_id"] for row in history}
    errors = {row["concept_id"] for row in history if int(row["is_correct"]) == 0}
    generated_list = [item["concept"] for item in exercises]
    generated = set(generated_list)
    hit = generated & errors
    return {
        "hr": len(hit) / len(generated) if generated else None,
        "recall": len(hit) / len(errors) if errors else None,
        "author_ecr": sum(c in errors for c in generated_list) / len(generated_list)
        if generated_list
        else None,
        "unknown_concepts": sorted(generated - known),
        "generated_unique_concepts": len(generated),
        "historical_error_concepts": len(errors),
    }


def load_lines(path):
    if not Path(path).exists():
        return []
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except ValueError:
                # A killed process can leave one unfinished final log line.
                rows.append({"truncated_record": True})
    return rows


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def maximum(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def analyze_run(folder):
    folder = Path(folder)
    requests = load_lines(folder / "requests.jsonl")
    samples = load_lines(folder / "resources.jsonl")
    good = [r for r in requests if r.get("status") == "ok"]

    def tokens(key):
        values = [r.get(key) for r in good]
        return sum(values) if values and all(v is not None for v in values) else None

    latencies = [r["elapsed_seconds"] for r in requests if "elapsed_seconds" in r]
    completion_tokens = tokens("completion_tokens")
    request_seconds = sum(latencies)
    gpus = [gpu for row in samples for gpu in row.get("gpus", [])]
    # Device energy only over observed intervals, not attributable process energy.
    energy = 0.0
    intervals = 0
    previous = {}
    for sample in samples:
        for gpu in sample.get("gpus", []):
            watts = gpu.get("power_watts")
            key = gpu["uuid"]
            if watts is not None and key in previous:
                timestamp, power = previous[key]
                energy += (sample["timestamp"] - timestamp) * (power + watts) / 2
                intervals += 1
            if watts is not None:
                previous[key] = sample["timestamp"], watts

    def process_values(scope, field):
        return [(r.get(scope) or {}).get(field) for r in samples]

    stages = {}
    for stage in sorted({r.get("stage", "other") for r in requests}):
        selected = [r for r in requests if r.get("stage", "other") == stage]
        stages[stage] = {
            "calls": len(selected),
            "seconds": sum(r.get("elapsed_seconds", 0) for r in selected),
            "prompt_tokens": sum(r["prompt_tokens"] for r in selected)
            if all(r.get("prompt_tokens") is not None for r in selected)
            else None,
            "completion_tokens": sum(r["completion_tokens"] for r in selected)
            if all(r.get("completion_tokens") is not None for r in selected)
            else None,
        }
    return {
        "calls": len(requests),
        "successful_calls": len(good),
        "failed_calls": sum(r.get("status") == "error" for r in requests),
        "prompt_tokens": tokens("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "total_tokens": tokens("total_tokens"),
        "request_seconds": request_seconds,
        "stages": stages,
        "regeneration_calls": max(stages.get("exercise_generation", {}).get("calls", 0) - 1, 0),
        "max_prompt_tokens": maximum([r.get("prompt_tokens") for r in good]),
        "p50_request_seconds": statistics.median(latencies) if latencies else None,
        "p95_request_seconds": sorted(latencies)[max(0, int(len(latencies) * 0.95 + 0.999) - 1)]
        if latencies
        else None,
        "mean_request_seconds": mean(latencies),
        "max_request_seconds": maximum(latencies),
        "effective_output_tokens_per_second": completion_tokens / request_seconds
        if completion_tokens is not None and request_seconds
        else None,
        "ttft_seconds": None,
        "worker_peak_rss_bytes": maximum(process_values("worker", "rss_bytes")),
        "server_peak_rss_bytes": maximum(process_values("server", "rss_bytes")),
        "server_mean_cpu_percent": mean(process_values("server", "cpu_percent")),
        "gpu_device_peak_memory_mib": maximum([g.get("memory_used_mib") for g in gpus]),
        "gpu_device_mean_utilization_percent": mean([g.get("utilization_percent") for g in gpus]),
        "gpu_device_mean_power_watts": mean([g.get("power_watts") for g in gpus]),
        "gpu_device_observed_energy_joules": energy if intervals else None,
        "resource_samples": len(samples),
        "gpu_monitor_failures": sum(bool(s.get("gpu_error")) for s in samples),
        "truncated_request_records": sum(bool(r.get("truncated_record")) for r in requests),
        "notes": [
            "Output tokens / full request time includes prompt processing and HTTP overhead.",
            "TTFT is unavailable for the preserved non-streaming author calls.",
            "GPU utilization, memory, and energy are device-wide, not process-exclusive.",
            "CPU percent uses psutil convention: 100 percent is one CPU core.",
            "RAM is sampled RSS, which can double-count shared pages in process trees.",
        ],
    }


def paired_intervals(rows, first="exegen_code", second="icl_cot_paper"):
    by_key = {(r["method"], r["student"], r["seed"]): r for r in rows if r["valid_set"]}
    output = {}
    for metric in ("hr", "recall"):
        per_student = defaultdict(list)
        for (method, student, seed), row in by_key.items():
            other = by_key.get((second, student, seed))
            if method == first and other and row[metric] is not None and other[metric] is not None:
                per_student[student].append(row[metric] - other[metric])
        differences = [mean(values) for values in per_student.values()]
        if not differences:
            output[metric] = {"students": 0, "difference": None, "ci95": None}
            continue
        rng = random.Random(42)
        boot = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(2000))
        output[metric] = {
            "students": len(differences),
            "difference": mean(differences),
            "ci95": [boot[49], boot[1949]] if len(differences) > 1 else None,
        }
    return {"comparison": f"{first} minus {second}", "metrics": output}


def evaluate_experiment(experiment):
    experiment = Path(experiment)
    rows = []
    manifest = read_json(experiment / "manifest.json")
    for job_entry in manifest["jobs"]:
        folder = experiment / job_entry["relative_dir"]
        attempts = sorted(folder.glob("attempt-*/status.json"))
        job = read_json(folder / "job.json")
        # Report the latest attempt, including failure, without hiding earlier costs.
        attempt = attempts[-1].parent if attempts else None
        status = read_json(attempt / "status.json") if attempt else {"status": "not_started"}
        result = (
            read_json(attempt / "result.json")
            if attempt and (attempt / "result.json").exists()
            else {}
        )
        exercises = result.get("exercises", [])
        valid = status["status"] == "complete" and len(exercises) == job["exercise_count"]
        metrics = concept_metrics(job["history"], exercises)
        if not valid:
            metrics["hr"] = metrics["recall"] = metrics["author_ecr"] = None
        telemetry = analyze_run(attempt) if attempt else {}
        all_attempts = [analyze_run(p.parent) for p in attempts]
        telemetry["all_attempt_calls"] = sum(a["calls"] for a in all_attempts)
        telemetry["all_attempt_request_seconds"] = sum(a["request_seconds"] for a in all_attempts)
        telemetry["all_attempt_total_tokens"] = (
            sum(a["total_tokens"] for a in all_attempts)
            if all(a["total_tokens"] is not None for a in all_attempts)
            else None
        )
        telemetry["attempts"] = len(attempts)
        row = {
            "method": job["method"],
            "student": job["student"],
            "seed": job["seed"],
            "status": status["status"],
            "valid_set": valid,
            "exercise_count": len(exercises),
            "elapsed_seconds": status.get("elapsed_seconds"),
            "termination": result.get("termination"),
            **metrics,
            **telemetry,
        }
        rows.append(row)
        if attempt:
            write_json(attempt / "analysis.json", row)
    summaries = {}
    for method in sorted({r["method"] for r in rows}):
        selected = [r for r in rows if r["method"] == method]
        keys = (
            "hr",
            "recall",
            "author_ecr",
            "elapsed_seconds",
            "calls",
            "total_tokens",
            "effective_output_tokens_per_second",
            "server_peak_rss_bytes",
            "gpu_device_peak_memory_mib",
            "gpu_device_mean_utilization_percent",
        )
        summaries[method] = {
            "scheduled": len(selected),
            "valid_sets": sum(r["valid_set"] for r in selected),
            "workflow_completions": sum(r["status"] == "complete" for r in selected),
            "means": {key: mean([r.get(key) for r in selected]) for key in keys},
            "non_null_counts": {key: sum(r.get(key) is not None for r in selected) for key in keys},
            "all_attempt_calls": sum(r["all_attempt_calls"] for r in selected),
            "all_attempt_request_seconds": sum(r["all_attempt_request_seconds"] for r in selected),
        }
    destination = experiment / "metrics"
    write_json(destination / "runs.json", rows)
    write_json(
        destination / "summary.json", {"methods": summaries, "paired": paired_intervals(rows)}
    )
    if rows:
        with (destination / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
            )
            writer.writeheader()
            writer.writerows(
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                    for k, v in row.items()
                }
                for row in rows
            )
    report = [
        "# Run analysis",
        "",
        "GPU statistics are device-wide; missing values are not zeros.",
        "",
        "| Method | Scheduled | Valid sets | HR | Recall | Mean calls | Mean seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def display(value):
        return "N/A" if value is None else f"{value:.3f}"

    for method, summary in summaries.items():
        means = summary["means"]
        report.append(
            f"| {method} | {summary['scheduled']} | {summary['valid_sets']} | "
            + " | ".join(display(means[k]) for k in ("hr", "recall", "calls", "elapsed_seconds"))
            + " |"
        )
    report.extend(
        [
            "",
            "Detailed request traces, server timings, sampling coverage, and resource "
            "statistics are in runs.json / runs.csv and each attempt's analysis.json.",
            "",
        ]
    )
    (destination / "report.md").write_text("\n".join(report), encoding="utf-8")
    # Blind review material: method mapping is separate from the reviewer-facing CSV.
    rng = random.Random(42)
    review, mapping = [], {}
    for method in summaries:
        candidates = []
        for entry in manifest["jobs"]:
            folder = experiment / entry["relative_dir"]
            job = read_json(folder / "job.json")
            latest = sorted(folder.glob("attempt-*/status.json"))
            if job["method"] == method and latest:
                attempt = latest[-1].parent
                if (
                    read_json(latest[-1])["status"] == "complete"
                    and (attempt / "result.json").exists()
                ):
                    items = read_json(attempt / "result.json").get("exercises", [])
                    if len(items) == job["exercise_count"]:
                        candidates.extend(items)
        for item in rng.sample(candidates, min(100, len(candidates))):
            review.append((method, item))
    rng.shuffle(review)
    with (destination / "blind-review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "review_id",
                "exercise",
                "options",
                "answer",
                "concept",
                "correct",
                "unambiguous",
                "aligned",
                "notes",
            ]
        )
        for i, (method, item) in enumerate(review, 1):
            key = f"R{i:04d}"
            mapping[key] = method
            writer.writerow(
                [
                    key,
                    item["exercise"],
                    json.dumps(item["options"], ensure_ascii=False),
                    json.dumps(item["answer"]),
                    item["concept"],
                    "",
                    "",
                    "",
                    "",
                ]
            )
    write_json(destination / "review-key.json", mapping)
    print("\n".join(report))
    return summaries


def export_experiment(experiment, destination):
    experiment, destination = Path(experiment).resolve(), Path(destination).resolve()
    if destination.is_relative_to(experiment):
        raise ValueError("Export destination must be outside the experiment directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        archive.add(experiment, arcname=experiment.name)
