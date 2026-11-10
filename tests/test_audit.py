import csv
import json
import subprocess
import sys

import pytest

from audit import audit_data
from common import ROOT, config, read_json, sha256, write_json


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def prepared(tmp_path):
    cfg = config("configs/reproduction.toml")
    cfg["data"].update(prepared_dir=str(tmp_path), history_length=2)
    rows = []
    for split in ("train", "validation", "test"):
        for i in (1, 2):
            rows.append(
                {
                    "user_id": "U_" + split,
                    "problem_id": f"Pm_{i}",
                    "content": "Question?",
                    "option": '{"A":"One","B":"Two"}',
                    "right_answer": '["A"]',
                    "concept_id": "K_test_CS",
                    "is_correct": "0" if i == 1 and split != "validation" else "1",
                    "submit_time": f"2020-01-0{i}T12:00:00",
                    "split": split,
                    "log_id": f"{split}_{i}",
                }
            )
    write_rows(tmp_path / "histories.csv", rows)
    return cfg, rows


def test_paper_audit_ignores_retrieval_files_and_needs_no_model(prepared, tmp_path):
    cfg, _ = prepared
    (tmp_path / "demonstrations.csv").write_text("corrupt")
    (tmp_path / "embedding-manifest.json").write_text("corrupt")
    (tmp_path / "explanation-manifest.json").write_text("corrupt")
    report = audit_data(cfg, ["icl_cot_paper"])
    assert report["data_valid"]
    assert report["statistics"]["students"] == 3
    assert report["splits"]["validation"]["students_without_errors"] == 1
    assert set(report["hashes"]) == {"histories.csv"}
    assert report["inference_artifact_availability"]["icl_cot_paper"]["artifacts_present"]
    assert read_json(tmp_path / "audit.json") == report
    assert "Data valid: True" in (tmp_path / "audit.md").read_text()


@pytest.mark.parametrize(
    "change,code",
    [
        (lambda rows: rows[0].update(right_answer='["Z"]'), "invalid_single_choice"),
        (lambda rows: rows[0].update(is_correct="2"), "invalid_correctness"),
        (lambda rows: rows[0].update(submit_time="invalid"), "invalid_or_incompatible_timestamp"),
        (lambda rows: rows[0].update(submit_time="2020-02-01"), "unordered_history"),
        (lambda rows: rows[1].update(split="test"), "student_split_leakage"),
        (lambda rows: rows.pop(), "unexpected_history_length"),
        (lambda rows: rows[0].update(concept_id=""), "missing_concept"),
    ],
)
def test_audit_blocks_invalid_prepared_histories(prepared, tmp_path, change, code):
    cfg, rows = prepared
    change(rows)
    write_rows(tmp_path / "histories.csv", rows)
    report = audit_data(cfg, ["icl_cot_paper"])
    assert not report["data_valid"]
    assert report["findings"][code]["severity"] == "error"


def test_retrieval_audit_checks_training_bank_before_embeddings(prepared, tmp_path):
    cfg, rows = prepared
    write_rows(tmp_path / "demonstrations.csv", rows[:1])
    report = audit_data(cfg, ["exegen_code", "icl_cot_paper"])
    assert report["data_valid"]
    assert report["statistics"]["demonstrations"] == 1
    assert not report["inference_artifact_availability"]["exegen_code"]["artifacts_present"]
    write_rows(tmp_path / "demonstrations.csv", rows[-1:])
    report = audit_data(cfg, ["exegen_code"])
    assert not report["data_valid"]
    assert "demonstration_student_leakage" in report["findings"]


def test_audit_reports_missing_inputs_and_caps_examples(prepared, tmp_path):
    cfg, rows = prepared
    rows = [dict(rows[-1], concept_id="") for _ in range(30)]
    write_rows(tmp_path / "histories.csv", rows)
    report = audit_data(cfg, ["icl_cot_paper"])
    assert report["findings"]["missing_concept"]["count"] == 30
    assert len(report["findings"]["missing_concept"]["examples"]) == 10
    (tmp_path / "histories.csv").unlink()
    report = audit_data(cfg, ["icl_cot_paper"])
    assert not report["data_valid"]
    assert report["findings"]["missing_history"]["count"] == 1


def test_audit_cli_works_without_any_site_packages(prepared, tmp_path):
    _, rows = prepared
    text = (
        (ROOT / "configs/reproduction.toml")
        .read_text()
        .replace('prepared_dir = "data/prepared"', f"prepared_dir = {json.dumps(str(tmp_path))}")
        .replace("history_length = 10", "history_length = 2")
    )
    path = tmp_path / "config.toml"
    path.write_text(text)
    command = [
        sys.executable,
        "-S",
        str(ROOT / "experiments/reproduce.py"),
        "--config",
        str(path),
        "audit-data",
        "--methods",
        "icl_cot_paper",
    ]
    success = subprocess.run(command, capture_output=True, text=True)
    assert success.returncode == 0, success.stderr
    assert "Data valid: True" in success.stdout
    rows[0]["right_answer"] = '["invalid"]'
    write_rows(tmp_path / "histories.csv", rows)
    failure = subprocess.run(command, capture_output=True, text=True)
    assert failure.returncode == 1
    assert "invalid_single_choice" in failure.stdout


def test_audit_detects_preparation_checksum_drift(prepared, tmp_path):
    cfg, rows = prepared
    write_json(
        tmp_path / "manifest.json",
        {
            "files": {
                "histories.csv": sha256(tmp_path / "histories.csv"),
                "demonstrations.csv": "unrelated for paper-only audit",
            }
        },
    )
    assert audit_data(cfg, ["icl_cot_paper"])["data_valid"]
    rows[0]["content"] = "Changed but still structurally valid question"
    write_rows(tmp_path / "histories.csv", rows)
    report = audit_data(cfg, ["icl_cot_paper"])
    assert not report["data_valid"]
    assert report["findings"]["prepared_checksum_mismatch"]["examples"] == ["histories.csv"]
