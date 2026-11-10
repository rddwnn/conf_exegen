"""Local llama-server lifecycle and resumable experiment orchestration."""

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from common import ROOT, fingerprint, git_state, read_json, resolve, sha256, write_json

METHODS = {"exegen_code", "icl_cot_paper", "baseline_code"}


def api_url(cfg, suffix):
    base = cfg["model"]["base_url"].rstrip("/")
    parsed = urlparse(base)
    if parsed.hostname not in ("localhost", "127.0.0.1", "::1") or parsed.scheme != "http":
        raise ValueError("This reproduction requires a local HTTP inference server")
    return base.removesuffix("/v1") + suffix


def api_get(cfg, suffix):
    with urlopen(api_url(cfg, suffix), timeout=10) as response:
        return json.load(response)


def server_command(cfg):
    settings = cfg["model"]
    if not settings["file"]:
        raise ValueError("Set model.file to a downloaded GGUF file")
    model, binary = resolve(settings["file"]), resolve(settings["server_binary"])
    if not model.is_file() or not binary.is_file():
        raise ValueError("Missing GGUF or llama-server binary; use download-model / build-server")
    parsed = urlparse(api_url(cfg, "/health"))
    if not 1 <= settings["threads"] <= 3:
        raise ValueError("model.threads must be between 1 and 3 for the allocated machine")
    return [
        str(binary),
        "--model",
        str(model),
        "--alias",
        settings["name"],
        "--host",
        parsed.hostname,
        "--port",
        str(parsed.port or 8080),
        "--ctx-size",
        str(settings["context_size"]),
        "--parallel",
        "1",
        "--threads",
        str(settings["threads"]),
        "--threads-batch",
        str(settings["threads"]),
        "--gpu-layers",
        str(settings["gpu_layers"]),
        "--flash-attn",
        "auto",
        "--fit",
        "off",
        "--no-context-shift",
        "--cache-ram",
        str(settings["cache_ram_mib"]),
        "--metrics",
    ]


def serve(cfg):
    command = server_command(cfg)
    metadata = ROOT / "runtime/server.json"
    metadata.parent.mkdir(exist_ok=True)
    if metadata.exists():
        import psutil

        old = read_json(metadata)
        try:
            process = psutil.Process(old["pid"])
            if process.create_time() == old["create_time"] and process.is_running():
                raise ValueError("A managed server is already running")
        except psutil.Error:
            pass
    with (metadata.parent / "server.log").open("a", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        import psutil

        write_json(
            metadata,
            {
                "pid": child.pid,
                "create_time": psutil.Process(child.pid).create_time(),
                "command": command,
                "model_sha256": sha256(resolve(cfg["model"]["file"])),
                "binary_sha256": sha256(command[0]),
                "model_config": cfg["model"],
                "version": subprocess.check_output(
                    [command[0], "--version"], text=True, stderr=subprocess.STDOUT
                ),
            },
        )
        try:
            print(
                f"Local server PID {child.pid}; logs: {metadata.parent / 'server.log'}", flush=True
            )
            code = child.wait()
            if code:
                raise RuntimeError(f"llama-server exited with {code}; inspect runtime/server.log")
        except KeyboardInterrupt:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        finally:
            if child.poll() is None:
                child.terminate()


def code_provenance():
    files = []
    for folder, pattern in (("src", "*.py"), ("prompts", "*"), ("scripts", "*.py")):
        files.extend(p for p in (ROOT / folder).rglob(pattern) if p.is_file())
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}


def check(cfg, offline=False, smoke=False):
    manifest = read_json(ROOT / "third_party/manifest.json")
    for name, expected in manifest["files"].items():
        if sha256(ROOT / "third_party/exegen" / name) != expected:
            raise ValueError(f"Upstream file changed: {name}")
    dependencies = {}
    for name in ("pyautogen", "openai", "httpx", "pandas", "numpy", "psutil"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"Missing {name}; install the reproduction extra") from exc
    if not (ROOT / "src/author/process.py").exists():
        raise ValueError("Missing working copies; run scripts/patch_upstream.py")
    destination = resolve(cfg["experiment"]["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    # Verify actual write access rather than trusting permission flags.
    import tempfile

    with tempfile.TemporaryFile(dir=destination):
        pass
    result = {
        "upstream_revision": manifest["revision"],
        "dependencies": dependencies,
        "git": git_state(),
        "free_disk_bytes": shutil.disk_usage(destination).free,
        "mode": "offline" if offline else "local_server",
        "python": sys.version,
    }
    if not offline:
        server_command(cfg)
        health = api_get(cfg, "/health")
        models = api_get(cfg, "/v1/models")
        if cfg["model"]["name"] not in {item["id"] for item in models["data"]}:
            raise ValueError("Configured model alias is not served")
        receipt = ROOT / "runtime/server.json"
        if not receipt.exists():
            raise ValueError("Start the server with the serve command to record model provenance")
        server = read_json(receipt)
        import psutil

        process = psutil.Process(server["pid"])
        if process.create_time() != server["create_time"]:
            raise ValueError("Stale server receipt")
        if server["model_config"] != cfg["model"]:
            raise ValueError("Running server configuration differs; restart it with this config")
        if sha256(resolve(cfg["model"]["file"])) != server["model_sha256"]:
            raise ValueError("GGUF changed after the server started")
        result.update({"health": health, "server": server, "properties": api_get(cfg, "/props")})
        if smoke:
            from openai import OpenAI

            client = OpenAI(base_url=cfg["model"]["base_url"], api_key="local", max_retries=0)
            response = client.chat.completions.create(
                model=cfg["model"]["name"],
                messages=[{"role": "user", "content": "Reply OK."}],
                max_tokens=8,
                temperature=0,
                seed=42,
            )
            result["smoke_response"] = response.model_dump(mode="json")
    return result


def llm_config(cfg, seed):
    model = cfg["model"]
    return {
        "cache_seed": None,
        "timeout": model["timeout_seconds"],
        "config_list": [
            {
                "model": model["name"],
                "base_url": model["base_url"],
                "api_key": "local",
                "price": [0, 0],
                "max_retries": 0,
                "seed": seed,
                "temperature": model["temperature"],
                "top_p": model["top_p"],
                "max_tokens": model["max_tokens"],
            }
        ],
    }


def worker_environment():
    return {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "AUTOGEN_USE_DOCKER": "False",
    }


def run_worker(job, replay=None):
    target = Path(job["run_dir"])
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "job.json", job)
    command = [sys.executable, str(ROOT / "src/worker.py"), str(target / "job.json")]
    if replay:
        command += ["--replay", str(replay)]
    with (target / "worker.log").open("w", encoding="utf-8") as handle:
        child = subprocess.Popen(
            command, stdout=handle, stderr=subprocess.STDOUT, env=worker_environment(), cwd=ROOT
        )
        try:
            code = child.wait()
        except KeyboardInterrupt:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            write_json(
                target / "status.json", {"status": "interrupted", "error": "Parent interrupted"}
            )
            raise
    if code and not (target / "status.json").exists():
        write_json(
            target / "status.json", {"status": "failed", "error": f"Worker exit code {code}"}
        )
    return code


def base_job(cfg, server):
    prepared = resolve(cfg["data"]["prepared_dir"])
    monitor = dict(cfg["monitor"])
    if not monitor["server_pid"]:
        monitor["server_pid"] = server["pid"]
    return {
        "demonstrations_csv": str(prepared / "examples_with_explanation_with_tokens.csv"),
        "relations_csv": str(prepared / "concept_relationship_filtered.csv"),
        "j_output.txt": str(ROOT / "prompts/j_output.txt"),
        "n_output.txt": str(ROOT / "prompts/n_output.txt"),
        "monitor": monitor,
        "exercise_count": cfg["experiment"]["exercise_count"],
    }


def explain(cfg):
    import pandas as pd

    provenance = check(cfg)
    prepared = resolve(cfg["data"]["prepared_dir"])
    path = prepared / "examples_with_explanation_with_tokens.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    origin = fingerprint(
        {
            "model": cfg["model"],
            "server": provenance["server"]["model_sha256"],
            "prompts": code_provenance(),
            "inputs": read_json(prepared / "embedding-manifest.json"),
        }
    )
    session = prepared / "explanation-session.json"
    if session.exists() and read_json(session)["identity"] != origin:
        raise ValueError(
            "Explanation settings changed. Re-run embed to reset demonstrations "
            "before explaining with a new model/configuration"
        )
    write_json(session, {"identity": origin})
    for index, row in frame.iterrows():
        folder = prepared / "explanation-runs" / origin[:16] / str(index)
        if row["explanation"] != "No explanation provided.":
            continue
        if (folder / "status.json").exists() and read_json(folder / "status.json")[
            "status"
        ] == "complete":
            explanation = read_json(folder / "result.json")["explanation"]
        else:
            # Preserve upstream input attributes, not preparation-only metadata.
            keys = [
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
            ]
            job = {
                **base_job(cfg, provenance["server"]),
                "method": "explanation",
                "record": {k: row.to_dict()[k] for k in keys},
                "run_dir": str(folder),
                "llm_config": llm_config(cfg, 42),
            }
            if run_worker(job):
                raise ValueError(f"Explanation failed; see {folder / 'worker.log'}")
            explanation = read_json(folder / "result.json")["explanation"]
        frame.at[index, "explanation"] = explanation
        temporary = path.with_suffix(".csv.part")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    write_json(
        prepared / "explanation-manifest.json",
        {
            "identity": origin,
            "model": cfg["model"],
            "server": provenance["server"],
            "demonstrations_sha256": sha256(path),
            "records": len(frame),
        },
    )


def run_experiment(cfg, config_path, methods=None, limit=None, split=None, seeds=None):
    import pandas as pd

    provenance = check(cfg)
    exp = dict(cfg["experiment"])
    methods = methods or exp["methods"]
    if set(methods) - METHODS:
        raise ValueError(f"Unknown methods: {set(methods) - METHODS}")
    if exp["exercise_count"] != 10:
        raise ValueError("Paper prompt and manager are fixed to ten exercises")
    if limit is not None:
        exp["student_limit"] = limit
    if split:
        exp["split"] = split
    if seeds:
        exp["seeds"] = seeds
    prepared = resolve(cfg["data"]["prepared_dir"])
    files = [
        "stuRec_1000_with_tokens.csv",
        "examples_with_explanation_with_tokens.csv",
        "concept_relationship_filtered.csv",
        "manifest.json",
        "embedding-manifest.json",
        "explanation-manifest.json",
    ]
    missing = [name for name in files if not (prepared / name).exists()]
    if missing:
        raise ValueError(f"Missing prepared artifacts: {missing}; run prepare, embed, explain")
    embeddings = read_json(prepared / "embedding-manifest.json")
    for name, expected in embeddings["inputs"].items():
        if sha256(prepared / name) != expected:
            raise ValueError(f"Stale embeddings: {name} changed; run embed and explain again")
    if sha256(prepared / files[0]) != embeddings["outputs"][files[0]]:
        raise ValueError("History embeddings were modified after preparation")
    demos = pd.read_csv(prepared / files[1], keep_default_na=False)
    if demos.empty or (demos["explanation"] == "No explanation provided.").any():
        raise ValueError("Demonstrations are empty or explanations are incomplete")
    explanation_manifest = read_json(prepared / "explanation-manifest.json")
    if explanation_manifest["demonstrations_sha256"] != sha256(prepared / files[1]):
        raise ValueError("Demonstration file changed after explanation preparation")
    if (
        explanation_manifest["model"] != cfg["model"]
        or explanation_manifest["server"]["model_sha256"] != provenance["server"]["model_sha256"]
    ):
        raise ValueError("Demonstrations were generated with a different model/configuration")
    histories = pd.read_csv(prepared / files[0], keep_default_na=False)
    histories = histories[histories["split"] == exp["split"]]
    students = list(dict.fromkeys(histories["user_id"]))[: exp["student_limit"]]
    if not students:
        raise ValueError("No students in requested split")
    if set(students) & set(demos["user_id"]):
        raise ValueError("Demonstration/student leakage")
    identity = {
        "config": cfg,
        "selection": exp,
        "methods": methods,
        "code": code_provenance(),
        "inputs": {n: sha256(prepared / n) for n in files},
        "model_sha256": provenance["server"]["model_sha256"],
        "binary_sha256": provenance["server"]["binary_sha256"],
    }
    experiment_id = fingerprint(identity)[:16]
    destination = resolve(exp["output_dir"]) / f"{exp['name']}-{experiment_id}"
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / "manifest.json").exists():
        shutil.copyfile(config_path, destination / "config.toml")
        entries = []
        for method in methods:
            for student in students:
                for seed in exp["seeds"]:
                    folder = destination / "runs" / method / student / str(seed)
                    job = {
                        **base_job(cfg, provenance["server"]),
                        "method": method,
                        "student": student,
                        "seed": seed,
                        "history": histories[histories["user_id"] == student].to_dict(
                            orient="records"
                        ),
                        "llm_config": llm_config(cfg, seed),
                    }
                    write_json(folder / "job.json", job)
                    entries.append({"relative_dir": str(folder.relative_to(destination))})
        write_json(
            destination / "manifest.json",
            {
                "identity": identity,
                "provenance": provenance,
                "jobs": entries,
                "created_at": time.time(),
                "config_path": str(config_path),
            },
        )
    failures = 0
    for entry in read_json(destination / "manifest.json")["jobs"]:
        folder = destination / entry["relative_dir"]
        attempts = sorted(folder.glob("attempt-*"))
        if attempts and (attempts[-1] / "status.json").exists():
            if read_json(attempts[-1] / "status.json")["status"] == "complete":
                continue
        job = read_json(folder / "job.json")
        # Refresh PID after a server restart; scientific identity does not include PID.
        job["monitor"] = base_job(cfg, provenance["server"])["monitor"]
        job["run_dir"] = str(folder / f"attempt-{len(attempts) + 1:04d}")
        print(f"Running {job['method']} {job['student']} seed={job['seed']}", flush=True)
        failures += bool(run_worker(job))
    from evaluate import evaluate_experiment

    summaries = evaluate_experiment(destination)
    if cfg["tracking"]["mlflow"]:
        import mlflow

        mlflow.set_tracking_uri(resolve(cfg["tracking"]["directory"]).as_uri())
        mlflow.set_experiment(exp["name"])
        with mlflow.start_run(run_name=destination.name):
            mlflow.log_params({"model": cfg["model"]["name"], "experiment_id": experiment_id})
            for method, summary in summaries.items():
                mlflow.log_metrics(
                    {f"{method}.{k}": v for k, v in summary["means"].items() if v is not None}
                )
            mlflow.log_artifacts(str(destination / "metrics"))
    print(f"Saved experiment: {destination} ({failures} failed jobs)")
    return 1 if failures else 0
