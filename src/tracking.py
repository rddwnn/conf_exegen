"""Optional local MLflow tracking; experiment files remain the source of truth."""

import json
import math
from contextlib import contextmanager

from common import fingerprint, read_json, resolve, write_json


def local_store(cfg):
    settings = cfg["tracking"]
    directory = resolve(settings["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    backend = settings.get("backend", "file")
    if backend not in {"file", "sqlite"}:
        raise ValueError("tracking.backend must be file or sqlite")
    uri = f"sqlite:///{directory / 'tracking.db'}" if backend == "sqlite" else directory.as_uri()
    return uri, directory / "artifacts"


def client_and_experiment(cfg, name):
    from mlflow import MlflowClient

    uri, artifacts = local_store(cfg)
    artifacts.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(name)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(
            name, artifact_location=artifacts.joinpath(fingerprint(name)[:16]).as_uri()
        )
    )
    return client, experiment_id, uri


def numeric_metrics(values, prefix=""):
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from numeric_metrics(value, name + ".")
        elif isinstance(value, (int, float)) and math.isfinite(value):
            yield name, float(value)


class ExperimentTracker:
    def __init__(self, cfg, destination):
        self.cfg = cfg
        self.destination = destination
        self.client = None
        self.failed = False

    def __enter__(self):
        if not self.cfg["tracking"]["mlflow"]:
            return self
        manifest = read_json(self.destination / "manifest.json")
        selection = manifest["identity"]["selection"]
        self.client, self.experiment_id, uri = client_and_experiment(self.cfg, selection["name"])
        self.state = self.destination / f"mlflow-{fingerprint(uri)[:12]}.json"
        if self.state.exists():
            self.run_id = read_json(self.state)["run_id"]
            self.client.update_run(self.run_id, status="RUNNING")
        else:
            run = self.client.create_run(
                self.experiment_id,
                tags={
                    "mlflow.runName": self.destination.name,
                    "exegen.kind": "experiment",
                    "exegen.output_directory": str(self.destination),
                },
            )
            self.run_id = run.info.run_id
            write_json(self.state, {"tracking_uri": uri, "run_id": self.run_id})
        identity = manifest["identity"]
        params = {
            "experiment_id": self.destination.name,
            "methods": ",".join(identity["methods"]),
            "model_sha256": identity["model_sha256"],
            "binary_sha256": identity["binary_sha256"],
            "inference_mode": self.cfg.get("inference", {}).get("mode", "local"),
            "git_commit": manifest["provenance"].get("git", {}).get("commit", "unknown"),
        }
        for group, values in [
            ("model", self.cfg["model"]),
            ("selection", selection),
            ("data", self.cfg["data"]),
        ]:
            for key, value in values.items():
                if isinstance(value, (str, int, float, bool)):
                    params[f"{group}.{key}"] = value
                elif isinstance(value, dict):
                    params[f"{group}.{key}"] = json.dumps(value, sort_keys=True)
        params["seeds"] = ",".join(map(str, selection["seeds"]))
        try:
            for key, value in params.items():
                self.client.log_param(self.run_id, key, str(value))
            for name in ("manifest.json", "config.toml"):
                self.client.log_artifact(self.run_id, str(self.destination / name), "provenance")
        except BaseException:
            self.client.set_terminated(self.run_id, "FAILED")
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.client:
            status = (
                "KILLED"
                if exc_type and issubclass(exc_type, KeyboardInterrupt)
                else ("FAILED" if exc_type or self.failed else "FINISHED")
            )
            self.client.set_terminated(self.run_id, status)

    @contextmanager
    def attempt(self, job):
        if not self.client:
            yield
            return
        folder = resolve(job["run_dir"])
        run = self.client.create_run(
            self.experiment_id,
            tags={
                "mlflow.runName": f"{job['method']}/{job['student']}/{job['seed']}/{folder.name}",
                "mlflow.parentRunId": self.run_id,
                "exegen.kind": "attempt",
            },
        )
        child_id = run.info.run_id
        status = "FAILED"
        try:
            write_json(folder / "mlflow.json", {"run_id": child_id, "parent_run_id": self.run_id})
            for key in ("method", "student", "seed"):
                self.client.log_param(child_id, key, str(job[key]))
            self.client.log_artifact(child_id, str(folder.parent / "job.json"), "provenance")
            yield
            if (folder / "status.json").exists():
                result = read_json(folder / "status.json")
                status = "FINISHED" if result["status"] == "complete" else "FAILED"
        except KeyboardInterrupt:
            status = "KILLED"
            raise
        finally:
            try:
                from evaluate import analyze_run, concept_metrics

                for key, value in numeric_metrics(analyze_run(folder)):
                    self.client.log_metric(child_id, key, value)
                if (folder / "result.json").exists():
                    exercises = read_json(folder / "result.json").get("exercises", [])
                    valid = status == "FINISHED" and len(exercises) == job["exercise_count"]
                    metrics = {"valid_set": valid, "exercise_count": len(exercises)}
                    if valid:
                        metrics.update(concept_metrics(job["history"], exercises))
                    for key, value in numeric_metrics(metrics):
                        self.client.log_metric(child_id, key, value)
                for name in (
                    "status.json",
                    "result.json",
                    "requests.jsonl",
                    "resources.jsonl",
                    "worker.log",
                    "analysis.json",
                ):
                    if (folder / name).is_file():
                        self.client.log_artifact(child_id, str(folder / name), "attempt")
            finally:
                self.client.set_terminated(child_id, status)

    def summary(self, summaries):
        if not self.client:
            return
        for key, value in numeric_metrics(summaries):
            self.client.log_metric(self.run_id, key, value)
        self.client.log_artifacts(self.run_id, str(self.destination / "metrics"), "metrics")


def log_preparation(cfg):
    prepared = resolve(cfg["data"]["prepared_dir"])
    manifest = read_json(prepared / "manifest.json")
    embedding = read_json(prepared / "embedding-manifest.json")
    audit = read_json(prepared / "audit.json")
    if not audit["data_valid"]:
        raise ValueError("Prepared data must pass audit before logging preparation")
    client, experiment_id, uri = client_and_experiment(cfg, "exegen-data-preparation")
    identity = fingerprint([manifest, embedding])
    state = prepared / f"mlflow-preparation-{fingerprint(uri)[:12]}.json"
    if state.exists() and read_json(state)["identity"] == identity:
        return read_json(state)
    run = client.create_run(
        experiment_id,
        tags={
            "mlflow.runName": f"mooccubex-{identity[:12]}",
            "exegen.kind": "data_preparation",
        },
    )
    run_id = run.info.run_id
    try:
        for key, value in manifest["settings"].items():
            client.log_param(run_id, key, str(value))
        client.log_param(run_id, "bert_revision", embedding["bert"]["revision"])
        for key in ("students", "demonstrations", "problems", "splits", "statistics"):
            for name, value in numeric_metrics({key: manifest[key]}):
                client.log_metric(run_id, name, value)
        for name in ("manifest.json", "embedding-manifest.json", "audit.json", "audit.md"):
            client.log_artifact(run_id, str(prepared / name), "preparation")
    except BaseException:
        client.set_terminated(run_id, "FAILED")
        raise
    client.set_terminated(run_id, "FINISHED")
    result = {"tracking_uri": uri, "run_id": run_id, "identity": identity}
    write_json(state, result)
    return result
