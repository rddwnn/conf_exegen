import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from common import ROOT, config, read_json
from confidence_context import attach_confidence, export_contexts
from confidence_data import (
    COLUMNS,
    Attempt,
    confidence_value,
    history_boundary,
    import_dbe,
    load_attempts,
)
from confidence_evaluation import evaluate_predictions, paired_brier, probability_metrics
from confidence_features import KnowledgeHistory, masked, prepare_confidence
from confidence_model import fit_models, predict_models, success_probability
from prepare import json_lines


def save_attempts(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def cohort(tmp_path):
    cfg = config("configs/confidence.toml")
    cfg["data"].update(
        input=str(tmp_path / "attempts.csv"),
        prepared_dir=str(tmp_path / "prepared"),
        raw_dir=str(tmp_path / "raw"),
    )
    cfg["model"].update(output_dir=str(tmp_path / "models"), c_values=[0.1, 1.0])
    cfg["evaluation"].update(output_dir=str(tmp_path / "outputs"), bootstrap_replicates=100)
    cfg["features"].update(mask_fractions=[0.0, 0.5, 1.0], evidence_limit=2)
    rows = []
    for student in range(10):
        for i in range(10):
            correct = (student + i) % 2
            rows.append(
                {
                    "attempt_id": f"A_{i:02d}",
                    "student_id": f"U_{student:02d}",
                    "question_id": f"Q_{i % 3}",
                    "timestamp": f"2020-01-{i + 1:02d}T12:00:00",
                    "is_correct": correct,
                    "confidence": "0" if i % 4 == 0 else "3" if not correct else "1",
                    "concept_ids": json.dumps(["C_A", "C_B"] if i % 3 == 0 else ["C_A"]),
                }
            )
    save_attempts(Path(cfg["data"]["input"]), rows)
    return cfg, rows


def test_missing_confidence_is_not_low_and_unknown_codes_fail(cohort):
    cfg, _ = cohort
    assert confidence_value("0", cfg["data"]) is None
    assert confidence_value("0.0", cfg["data"]) is None
    assert confidence_value("", cfg["data"]) is None
    assert confidence_value("1", cfg["data"]) == 1
    for value in ("-1", "4", "nan", "inf", "guessed"):
        with pytest.raises(ValueError, match="Unknown confidence"):
            confidence_value(value, cfg["data"])


def test_normalization_rejects_duplicate_attempts_and_concepts(cohort):
    cfg, rows = cohort
    path = Path(cfg["data"]["input"])
    save_attempts(path, [rows[0], rows[0]])
    with pytest.raises(ValueError, match="Duplicate"):
        load_attempts(path, cfg["data"])
    rows[0]["concept_ids"] = '["a","a"]'
    save_attempts(path, rows[:1])
    with pytest.raises(ValueError, match="unique strings"):
        load_attempts(path, cfg["data"])


def test_concept_statistics_missingness_and_multiconcept_aggregation(cohort):
    cfg, _ = cohort
    state = KnowledgeHistory(cfg["data"])
    for i, (correct, confidence) in enumerate([(1, 1), (0, 3), (1, None)]):
        state.add(
            Attempt(str(i), "s", "q", datetime.now(timezone.utc), correct, confidence, ("a",))
        )
    summary = state.concept_summary("a")
    assert summary["reported_count"] == 2
    assert summary["confidence_mean"] == 2
    assert summary["confidence_missing_rate"] == pytest.approx(1 / 3)
    assert summary["high_confidence_error_rate"] == 0.5
    assert summary["low_confidence_correct_rate"] == 0.5
    combined = state.features(["a", "unseen"])
    assert combined["correctness"][0] == 1.5
    assert combined["correctness"][2] == 0.5
    assert state.concept_summary("unseen")["confidence_missing_rate"] == 1


def test_masking_is_stable_nested_and_never_changes_correctness_features(cohort):
    cfg, _ = cohort
    students = load_attempts(cfg["data"]["input"], cfg["data"])
    records = next(iter(students.values()))
    before, after = KnowledgeHistory(cfg["data"]), KnowledgeHistory(cfg["data"], 1.0)
    for record in records:
        before.add(record)
        after.add(record)
    assert before.features(["C_A"])["correctness"] == after.features(["C_A"])["correctness"]
    assert after.concept_summary("C_A")["reported_count"] == 0
    for record in records:
        a = masked(record.confidence, record.student_id, record.attempt_id, 0.25, 42)
        b = masked(record.confidence, record.student_id, record.attempt_id, 0.75, 42)
        if a is None:
            assert b is None


def test_student_splits_are_disjoint_and_deterministic(cohort):
    cfg, _ = cohort
    first = prepare_confidence(cfg)
    second = prepare_confidence(cfg)
    assert first == second
    assert list(first["splits"].values()).count("train") == 6
    assert list(first["splits"].values()).count("validation") == 2
    assert list(first["splits"].values()).count("test") == 2
    folder = Path(cfg["data"]["prepared_dir"])
    membership = {
        split: {r["student_id"] for r in json_lines(folder / f"{split}.jsonl")}
        for split in ("train", "validation", "test")
    }
    assert not membership["train"] & membership["test"]
    assert not membership["validation"] & membership["test"]
    assert not membership["train"] & membership["validation"]


def test_equal_timestamp_groups_are_never_split_or_seen_during_training(cohort):
    cfg, rows = cohort
    for row in rows:
        index = int(row["attempt_id"].split("_")[1])
        if index in (0, 1):
            row["timestamp"] = "2020-01-01T12:00:00"
        if 4 <= index <= 7:
            row["timestamp"] = "2020-01-05T12:00:00"
    save_attempts(Path(cfg["data"]["input"]), rows)
    students = load_attempts(cfg["data"]["input"], cfg["data"])
    assert history_boundary(next(iter(students.values())), 0.7) == 4
    manifest = prepare_confidence(cfg)
    assert all(v["history_count"] == 4 for v in manifest["boundaries"].values())
    train = list(json_lines(Path(cfg["data"]["prepared_dir"]) / "train.jsonl"))
    assert not any(r["attempt_id"] in ("A_00", "A_01") for r in train)
    assert all(r["features_correctness"][3] == 2 for r in train if r["attempt_id"] == "A_02")
    assert all(
        r["features_correctness"][3] == 4
        for r in train
        if r["attempt_id"] in ("A_04", "A_05", "A_06", "A_07")
    )


def test_future_outcomes_and_confidence_do_not_change_heldout_features(cohort):
    cfg, rows = cohort
    manifest = prepare_confidence(cfg)
    folder = Path(cfg["data"]["prepared_dir"])
    before = {
        split: list(json_lines(folder / f"{split}.jsonl")) for split in ("validation", "test")
    }
    for row in rows:
        if (
            manifest["splits"][row["student_id"]] != "train"
            and int(row["attempt_id"].split("_")[1]) >= 7
        ):
            row["is_correct"] = 1 - row["is_correct"]
            row["confidence"] = "3"
    save_attempts(Path(cfg["data"]["input"]), rows)
    prepare_confidence(cfg)
    for split in before:
        after = list(json_lines(folder / f"{split}.jsonl"))
        for a, b in zip(before[split], after, strict=True):
            assert a["features_correctness"] == b["features_correctness"]
            assert a["features_confidence"] == b["features_confidence"]
            assert a["simple_probability"] == b["simple_probability"]
            assert a["y_success"] != b["y_success"]
        assert {r["features_correctness"][3] for r in after} == {7}


def test_ordinary_and_raw_context_exports_do_not_require_models(cohort):
    cfg, _ = cohort
    prepare_confidence(cfg)
    paths = export_contexts(cfg, ["ordinary_exegen", "raw_confidence"], 1.0)
    ordinary = list(json_lines(paths["ordinary_exegen"]))
    raw = list(json_lines(paths["raw_confidence"]))
    assert ordinary[0]["confidence_context"] is None
    assert ordinary[0]["history"] == raw[0]["history"]
    assert all("confidence" not in row for row in ordinary[0]["history"])
    assert all(v["confidence"] is None for v in raw[0]["confidence_context"]["values"])
    assert all(
        not {"confidence", "is_correct", "hint_used", "time_taken"} & target.keys()
        for item in raw
        for target in item["targets"]
    )
    author = {"examples": [], "tasks": [{"content": "Question"}]}
    context = raw[0]["confidence_context"]
    assert attach_confidence(author, None) == author
    injected = attach_confidence(author, context)
    assert injected["tasks"] == author["tasks"]
    assert "historical_confidence" not in author


def test_fit_predict_evaluate_and_export_structured_profiles(cohort):
    pytest.importorskip("sklearn")
    cfg, _ = cohort
    manifest = prepare_confidence(cfg)
    # fit must not open any test features or labels.
    test_path = Path(cfg["data"]["prepared_dir"]) / "test.jsonl"
    test_data = test_path.read_bytes()
    test_path.unlink()
    result = fit_models(cfg)
    test_path.write_bytes(test_data)
    assert set(result) == {"structured_confidence", "structured_without_confidence"}
    model = read_json(Path(cfg["model"]["output_dir"]) / "structured_confidence.json")
    training_rows = list(json_lines(Path(cfg["data"]["prepared_dir"]) / "train.jsonl"))
    expected_means = [
        sum(row["features_confidence"]["0.0"][i] for row in training_rows) / len(training_rows)
        for i in range(len(model["feature_names"]))
    ]
    assert model["mean"] == pytest.approx(expected_means)
    assert set(model["train_students"]).isdisjoint(model["validation_students"])
    assert set(model["train_students"]) == {
        k for k, v in manifest["splits"].items() if v == "train"
    }
    predictions = predict_models(cfg)
    report = evaluate_predictions(cfg, predictions)
    assert len(report["methods"]) == 9
    assert all(row["completion_rate"] == 1 for row in report["methods"].values())
    assert report["paired_brier"][0]["students"] == 2
    assert report["paired_brier"][0]["ci95"] is not None
    unmasked = report["methods"]["structured_without_confidence@0.0"]["brier"]
    assert unmasked == report["methods"]["structured_without_confidence@1.0"]["brier"]
    paths = export_contexts(cfg)
    structured = list(json_lines(paths["structured_confidence"]))[0]
    ablated = list(json_lines(paths["structured_without_confidence"]))[0]
    assert structured["history"] == ablated["history"]
    assert "confidence" in structured["confidence_context"]["concepts"][0]
    assert "confidence" not in ablated["confidence_context"]["concepts"][0]
    assert len(structured["confidence_context"]["concepts"][0]["evidence_attempt_ids"]) <= 2
    for item in structured["confidence_context"]["concepts"]:
        assert 0 <= item["error_risk"] <= 1
        assert not any(int(a.split("_")[1]) >= 7 for a in item["evidence_attempt_ids"])


def test_json_probability_inference_has_correct_error_orientation():
    model = {"feature_names": ["x"], "mean": [0], "scale": [1], "coefficients": [1], "intercept": 0}
    assert success_probability(model, [0]) == 0.5
    assert success_probability(model, [1000]) == 0
    assert success_probability(model, [-1000]) == 1


def test_probability_metrics_and_cluster_weighting():
    pytest.importorskip("sklearn")
    assert probability_metrics([(1, 0.9), (0, 0.1)])["brier"] == pytest.approx(0.01)
    assert probability_metrics([(1, 0.5)])["auroc"] is None
    assert probability_metrics([])["brier"] is None
    truth = {("one", "a"): 1, ("two", "b"): 0, ("two", "c"): 0}
    a, b = {k: 0.9 for k in truth}, {k: 0.5 for k in truth}
    result = paired_brier(a, b, truth, replicates=100)
    assert result["students"] == 2 and result["targets"] == 3
    assert result["difference"] == pytest.approx((0.01 + 0.81 + 0.81) / 3 - 0.25)


def test_invalid_llm_probabilities_are_counted_and_labels_come_from_truth(cohort, tmp_path):
    pytest.importorskip("sklearn")
    cfg, _ = cohort
    prepare_confidence(cfg)
    rows = list(json_lines(Path(cfg["data"]["prepared_dir"]) / "test.jsonl"))
    path = tmp_path / "llm.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "mask_fraction",
                "student_id",
                "attempt_id",
                "p_success",
                "y_success",
            ],
        )
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow(
                {
                    "method": "llm",
                    "mask_fraction": 0,
                    "student_id": row["student_id"],
                    "attempt_id": row["attempt_id"],
                    "p_success": "nan" if i == 0 else 0.5,
                    "y_success": 999,
                }
            )
    report = evaluate_predictions(cfg, path)["methods"]["llm@0.0"]
    assert report["invalid_outputs"] == 1
    assert report["count"] == len(rows) - 1
    assert report["brier"] == 0.25
    with path.open("a") as handle:
        handle.write(f"llm,0,{rows[0]['student_id']},{rows[0]['attempt_id']},0.5,999\n")
    with pytest.raises(ValueError, match="Duplicate prediction"):
        evaluate_predictions(cfg, path)


def test_import_requires_verified_column_mapping_and_records_missing_confidence(cohort):
    cfg, _ = cohort
    with pytest.raises(ValueError, match="column mappings"):
        import_dbe(cfg)
    folder = Path(cfg["data"]["raw_dir"])
    folder.mkdir()
    mapping = cfg["import"]
    for name in (
        "attempt_id",
        "student_id",
        "question_id",
        "timestamp",
        "is_correct",
        "confidence",
    ):
        mapping[name] = name
    mapping.update(relation_question_id="question_id", relation_concept_id="concept_id")
    (folder / mapping["relations_file"]).write_text("question_id,concept_id\nq,a\nq,b\n")
    (folder / mapping["transactions_file"]).write_text(
        "attempt_id,student_id,question_id,timestamp,is_correct,confidence\na,s,q,2020-01-01,0,0\n"
    )
    import_dbe(cfg)
    row = next(iter(load_attempts(cfg["data"]["input"], cfg["data"]).values()))[0]
    assert row.confidence is None and row.concept_ids == ("a", "b")
    assert Path(cfg["data"]["input"]).with_suffix(".source.json").is_file()


def test_archive_html_failure_is_explicit_and_not_saved_as_csv(cohort, monkeypatch):
    import io

    import confidence_data

    cfg, _ = cohort
    monkeypatch.setattr(
        confidence_data,
        "urlopen",
        lambda *args, **kwargs: io.StringIO("<html>Access denied</html>"),
    )
    with pytest.raises(ValueError, match="access protection"):
        confidence_data.fetch_dbe(cfg)
    assert not Path(cfg["data"]["raw_dir"]).exists()


def test_confidence_cli_end_to_end_without_gpu(cohort, tmp_path):
    pytest.importorskip("sklearn")
    import subprocess
    import sys

    cfg, _ = cohort
    text = []
    for section in ("data", "features", "model", "evaluation"):
        text.append(f"[{section}]")
        text.extend(f"{key} = {json.dumps(value)}" for key, value in cfg[section].items())
    path = tmp_path / "confidence.toml"
    path.write_text("\n".join(text), encoding="utf-8")
    cli = [sys.executable, str(ROOT / "experiments/confidence.py"), "--config", str(path)]
    commands = [
        ["prepare"],
        ["fit"],
        ["predict"],
        ["evaluate", str(Path(cfg["evaluation"]["output_dir"]) / "test-predictions.csv")],
        ["export-contexts"],
    ]
    for command in commands:
        result = subprocess.run(cli + command, capture_output=True, text=True)
        assert result.returncode == 0, (command, result.stdout, result.stderr)
    report = read_json(Path(cfg["evaluation"]["output_dir"]) / "test-metrics.json")
    assert report["split"] == "test"
    assert report["methods"]["structured_confidence@0.0"]["completion_rate"] == 1
