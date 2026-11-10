"""Real local MLflow stores exercise persistence and failure handling, without an LLM."""

from pathlib import Path

import pytest

from common import config, write_json
from tracking import ExperimentTracker, client_and_experiment, log_preparation

pytest.importorskip("mlflow")


def setup_experiment(tmp_path):
    cfg = config("configs/remote.toml")
    cfg["tracking"].update(directory=str(tmp_path / "mlruns"), backend="sqlite", mlflow=True)
    destination = tmp_path / "outputs" / "experiment"
    write_json(
        destination / "manifest.json",
        {
            "identity": {
                "selection": cfg["experiment"],
                "methods": ["icl_cot_paper"],
                "model_sha256": "abc",
                "binary_sha256": None,
            },
            "provenance": {},
        },
    )
    (destination / "config.toml").write_text("test = true\n")
    job = {
        "method": "icl_cot_paper",
        "student": "synthetic-student",
        "seed": 42,
        "run_dir": str(destination / "runs" / "student" / "42" / "attempt-0001"),
    }
    write_json(Path(job["run_dir"]).parent / "job.json", job)
    return cfg, destination, job


@pytest.mark.parametrize("error,status", [(RuntimeError, "FAILED"), (KeyboardInterrupt, "KILLED")])
def test_tracking_records_failed_and_interrupted_attempts(tmp_path, error, status):
    cfg, destination, job = setup_experiment(tmp_path)
    with pytest.raises(error):
        with ExperimentTracker(cfg, destination) as tracker:
            with tracker.attempt(job):
                raise error("synthetic interruption")
    client, experiment_id, _ = client_and_experiment(cfg, cfg["experiment"]["name"])
    runs = client.search_runs([experiment_id])
    assert len(runs) == 2
    assert {run.info.status for run in runs} == {status}
    child = next(run for run in runs if run.data.tags["exegen.kind"] == "attempt")
    assert child.data.tags["mlflow.parentRunId"] in {run.info.run_id for run in runs}


def test_preparation_logging_is_idempotent_and_preserves_artifacts(tmp_path):
    cfg, _, _ = setup_experiment(tmp_path)
    prepared = tmp_path / "prepared"
    cfg["data"]["prepared_dir"] = str(prepared)
    write_json(
        prepared / "manifest.json",
        {
            "settings": {"split_seed": 42},
            "students": 5,
            "demonstrations": 2,
            "problems": 4,
            "splits": {"train": 3, "validation": 1, "test": 1},
            "statistics": {"raw_interactions": 99},
        },
    )
    write_json(prepared / "embedding-manifest.json", {"bert": {"revision": "synthetic"}})
    write_json(prepared / "audit.json", {"data_valid": True})
    (prepared / "audit.md").write_text("Synthetic data audit passed.\n")
    first = log_preparation(cfg)
    assert log_preparation(cfg) == first
    client, experiment_id, _ = client_and_experiment(cfg, "exegen-data-preparation")
    runs = client.search_runs([experiment_id])
    assert len(runs) == 1
    assert runs[0].data.metrics["students"] == 5
    assert runs[0].info.status == "FINISHED"
    assert len(client.list_artifacts(first["run_id"], "preparation")) == 4
    write_json(prepared / "audit.json", {"data_valid": False})
    with pytest.raises(ValueError, match="pass audit"):
        log_preparation(cfg)
