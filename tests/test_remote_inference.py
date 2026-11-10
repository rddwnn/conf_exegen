import copy
import json
import os
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError

import pandas as pd
import pytest

from common import ROOT, config, fingerprint, read_json, sha256, write_json
from inference import (
    api_get,
    inference_mode,
    remote_server,
    server_identity,
)
from runtime import base_job, check, run_experiment, serve
from telemetry import ResourceMonitor, request_stage


def completion(content="OK", finish="stop", reasoning=None):
    return {
        "id": "test-local-http",
        "object": "chat.completion",
        "created": 0,
        "model": "exegen-local",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                },
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


@pytest.fixture
def remote(tmp_path):
    cfg = config("configs/remote.toml")
    cfg["inference"]["receipt"] = str(tmp_path / "remote.json")
    cfg["experiment"].update(output_dir=str(tmp_path / "outputs"), seeds=[42], student_limit=1)
    cfg["data"]["prepared_dir"] = str(tmp_path / "prepared")
    receipt = {
        "schema_version": 1,
        "created_at": "2026-01-01T00:00:00Z",
        "model": {
            "name": "Qwen3-14B",
            "alias": "exegen-local",
            "gguf_sha256": "b" * 64,
            "quantization": "Q4_K_M",
        },
        "server": {
            "version": "version: 999 (aaaaaaaa)",
            "commit": "a" * 40,
            "binary_sha256": "c" * 64,
            "pid": 987654321,
            "runtime_parameters": {
                "host": "127.0.0.1",
                "port": 8080,
                "context_size": 16384,
                "parallel": 1,
                "threads": 3,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        },
        "generation": {
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
        },
    }
    write_json(Path(cfg["inference"]["receipt"]), receipt)
    state = {"posts": [], "overrides": {}, "generated": False}

    class Handler(BaseHTTPRequestHandler):
        def emit(self, status, payload):
            encoded = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            defaults = {
                "/health": (200, {"status": "ok"}),
                "/v1/models": (200, {"data": [{"id": "exegen-local"}]}),
                "/props": (
                    200,
                    {
                        "default_generation_settings": {"n_ctx": 16384},
                        "chat_template_kwargs": {"enable_thinking": False},
                        "build_info": "b999-aaaaaaaa",
                    },
                ),
            }
            self.emit(*state["overrides"].get(self.path, defaults.get(self.path, (404, {}))))

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["posts"].append(body)
            if self.path in state["overrides"]:
                self.emit(*state["overrides"][self.path])
                return
            if body["messages"][-1]["content"] == "Reply with the word OK.":
                reply = "OK"
            else:
                stage = request_stage(body["messages"])
                exercise = {
                    "Exercise": "Which?",
                    "Options": {"A": "One", "B": "Two"},
                    "Answer": ["A"],
                    "Concept": "K_Test_CS",
                }
                exercises = "```json\n" + json.dumps({"exercises": [exercise] * 10}) + "\n```"
                if stage == "knowledge_perception":
                    state["generated"] = False
                    reply = "Weak concept K_Test_CS."
                elif stage == "manager":
                    reply = (
                        "stopChat\n" + exercises
                        if state["generated"]
                        else "ExerciseGenerator next."
                    )
                elif stage == "speaker_selection":
                    reply = "ExerciseGenerator"
                elif stage == "summary":
                    reply = "Final summary."
                else:
                    if body["messages"][0].get("content", ""):
                        state["generated"] = True
                    reply = exercises
            self.emit(200, completion(reply))

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    cfg["model"]["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield cfg, receipt, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_remote_check_needs_no_local_binary_weights_receipt_or_process(remote, monkeypatch):
    import psutil

    import runtime

    cfg, _, state = remote
    assert "file" not in cfg["model"] and "server_binary" not in cfg["model"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Remote validation must not use local server lifecycle/PID")

    monkeypatch.setattr(runtime, "server_command", forbidden)
    monkeypatch.setattr(psutil, "Process", forbidden)
    result = check(cfg)
    assert result["mode"] == "remote_server"
    assert result["server"]["pid"] == 0
    assert "not recomputed locally" in result["server"]["verification"]
    assert result["smoke_response"]["choices"][0]["message"]["content"] == "OK"
    assert state["posts"][0]["max_tokens"] == 128
    assert state["posts"][0]["chat_template_kwargs"]["enable_thinking"] is False


def test_remote_offline_check_does_not_require_receipt_or_network(remote, monkeypatch):
    import inference

    cfg, _, state = remote
    Path(cfg["inference"]["receipt"]).unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline validation must not contact the server")

    monkeypatch.setattr(inference, "urlopen", forbidden)
    assert check(cfg, offline=True)["mode"] == "offline"
    assert not state["posts"]


def test_remote_serve_is_rejected_and_legacy_mode_defaults_local(remote):
    cfg, _, _ = remote
    with pytest.raises(ValueError, match="existing GPU server"):
        serve(cfg)
    assert inference_mode({}) == "local"
    with pytest.raises(ValueError, match="local or remote"):
        inference_mode({"inference": {"mode": "anything"}})


@pytest.mark.parametrize(
    "path,status,payload,expected",
    [
        ("/health", 401, {}, "authentication"),
        ("/health", 403, {}, "authentication"),
        ("/health", 503, {}, "not ready"),
        ("/health", 200, {"status": "loading"}, "not ready"),
        ("/v1/models", 200, {"data": [{"id": "wrong"}]}, "model mismatch"),
        ("/v1/models", 200, b"not json", "Invalid inference response"),
        ("/props", 200, {"default_generation_settings": {"n_ctx": 2048}}, "context mismatch"),
        ("/v1/chat/completions", 401, {"error": {"message": "bad key"}}, "authentication"),
        (
            "/v1/chat/completions",
            404,
            {"error": {"message": "no chat"}},
            "chat-completions failure",
        ),
        ("/v1/chat/completions", 200, {"choices": []}, "no choices"),
        (
            "/v1/chat/completions",
            200,
            completion("", "length", "Okay, the user just said"),
            "reasoning_chars=",
        ),
        ("/v1/chat/completions", 200, completion(" ", "stop"), "message.content"),
        (
            "/v1/chat/completions",
            200,
            completion("OK", "stop", "thinking anyway"),
            "Thinking tokens",
        ),
    ],
)
def test_remote_failures_have_specific_diagnostics(remote, path, status, payload, expected):
    cfg, _, state = remote
    state["overrides"][path] = status, payload
    with pytest.raises(ValueError, match=expected):
        check(cfg, smoke=True)


def test_missing_optional_props_does_not_fabricate_metadata(remote):
    cfg, _, state = remote
    state["overrides"]["/props"] = 404, {}
    result = check(cfg)
    assert result["properties"] is None
    assert result["context_validation"] == "receipt_only_not_verified_via_api"
    assert result["server"]["model_sha256"] == "b" * 64


@pytest.mark.parametrize(
    "reason,expected",
    [(ConnectionRefusedError("refused"), "connectivity"), (socket.timeout("timed out"), "timeout")],
)
def test_tunnel_connectivity_and_timeout_are_distinct(remote, monkeypatch, reason, expected):
    import inference

    cfg, _, _ = remote

    def fail(*args, **kwargs):
        raise URLError(reason)

    monkeypatch.setattr(inference, "urlopen", fail)
    with pytest.raises(ValueError, match=expected):
        api_get(cfg, "/health")


@pytest.mark.parametrize(
    "field,value,expected",
    [("gguf_sha256", "", "GGUF SHA-256"), ("alias", "wrong", "alias mismatch")],
)
def test_remote_receipt_is_required_and_validated(remote, field, value, expected):
    cfg, receipt, _ = remote
    receipt["model"][field] = value
    write_json(Path(cfg["inference"]["receipt"]), receipt)
    with pytest.raises(ValueError, match=expected):
        remote_server(cfg)


def test_receipt_context_thinking_commit_and_generation_mismatch(remote):
    cfg, receipt, state = remote
    path = Path(cfg["inference"]["receipt"])
    for mutate, expected in [
        (lambda r: r["server"]["runtime_parameters"].update(context_size=8192), "context length"),
        (
            lambda r: r["server"]["runtime_parameters"].update(
                chat_template_kwargs={"enable_thinking": True}
            ),
            "template settings mismatch",
        ),
        (lambda r: r["generation"].update(top_p=0.5), "generation setting mismatch"),
    ]:
        changed = copy.deepcopy(receipt)
        mutate(changed)
        write_json(path, changed)
        with pytest.raises(ValueError, match=expected):
            remote_server(cfg)
    write_json(path, receipt)
    state["overrides"]["/props"] = 200, {"build_info": "b999-dddddddd"}
    with pytest.raises(ValueError, match="commit mismatch"):
        check(cfg)


def test_receipt_identity_ignores_restart_time_but_includes_scientific_settings(remote):
    cfg, receipt, _ = remote
    path = Path(cfg["inference"]["receipt"])
    before = fingerprint(server_identity(remote_server(cfg)))
    receipt["created_at"] = "later"
    receipt["server"]["pid"] = 123
    write_json(path, receipt)
    assert fingerprint(server_identity(remote_server(cfg))) == before
    receipt["model"]["gguf_sha256"] = "d" * 64
    write_json(path, receipt)
    assert fingerprint(server_identity(remote_server(cfg))) != before
    receipt["model"]["gguf_sha256"] = "b" * 64
    receipt["server"]["binary_sha256"] = None
    write_json(path, receipt)
    assert remote_server(cfg)["binary_sha256"] is None
    assert fingerprint(server_identity(remote_server(cfg))) != before


def test_remote_monitor_never_samples_gpu_or_server_pid(remote, tmp_path, monkeypatch):
    cfg, receipt, _ = remote
    cfg["monitor"].update(server_pid=receipt["server"]["pid"], gpu_indices=[0])
    job = base_job(cfg, remote_server(cfg))
    assert job["monitor"]["server_pid"] == 0
    monitor = ResourceMonitor(tmp_path / "resources.jsonl", job["monitor"])
    seen = []

    def sample(pid):
        seen.append(pid)
        return {"pid": pid, "rss_bytes": 100, "cpu_percent": None}

    monkeypatch.setattr(monitor, "process_sample", sample)

    def forbidden(*args, **kwargs):
        raise AssertionError("Remote mode must not run local nvidia-smi")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monitor.sample()
    assert seen == [os.getpid()]
    record = json.loads((tmp_path / "resources.jsonl").read_text())
    assert record["server"] is None and record["gpus"] == []
    assert record["inference_mode"] == "remote"


def prepare_fixture(cfg):
    folder = Path(cfg["data"]["prepared_dir"])
    folder.mkdir()
    row = {
        "user_id": "U_test",
        "problem_id": "Pm_1",
        "content": "Question?",
        "option": '{"A":"One","B":"Two"}',
        "right_answer": '["A"]',
        "concept_id": "K_Test_CS",
        "is_correct": 0,
        "explanation": "",
        "knowledge_evidence": "K_Test_CS",
        "tokens": "[1,0]",
        "split": "validation",
    }
    pd.DataFrame([row]).to_csv(folder / "histories.csv", index=False)
    pd.DataFrame([row]).to_csv(folder / "stuRec_1000_with_tokens.csv", index=False)
    pd.DataFrame(
        [dict(row, user_id="U_train", split="train", explanation="Training explanation")]
    ).to_csv(folder / "examples_with_explanation_with_tokens.csv", index=False)
    pd.DataFrame(columns=["c1", "c2", "ground_truth"]).to_csv(
        folder / "concept_relationship_filtered.csv", index=False
    )
    write_json(folder / "manifest.json", {})
    write_json(
        folder / "embedding-manifest.json",
        {
            "inputs": {"histories.csv": sha256(folder / "histories.csv")},
            "outputs": {
                "stuRec_1000_with_tokens.csv": sha256(folder / "stuRec_1000_with_tokens.csv")
            },
        },
    )
    write_json(
        folder / "explanation-manifest.json",
        {
            "model": cfg["model"],
            "server": {"model_sha256": "b" * 64},
            "demonstrations_sha256": sha256(folder / "examples_with_explanation_with_tokens.csv"),
        },
    )


def test_remote_pipeline_uses_same_agents_settings_and_resume(remote, tmp_path):
    cfg, receipt, state = remote
    prepare_fixture(cfg)
    assert (
        run_experiment(cfg, ROOT / "configs/remote.toml", split="validation", limit=1, seeds=[42])
        == 0
    )
    root = Path(cfg["experiment"]["output_dir"])
    experiment = next(root.iterdir())
    manifest = read_json(experiment / "manifest.json")
    assert manifest["identity"]["inference_identity"]["commit"] == receipt["server"]["commit"]
    assert (
        read_json(experiment / "metrics/summary.json")["methods"]["exegen_code"]["valid_sets"] == 1
    )
    assert all(r["chat_template_kwargs"] == {"enable_thinking": False} for r in state["posts"])
    assert all(r["model"] == "exegen-local" for r in state["posts"])
    calls = len(state["posts"])
    receipt["created_at"] = "restart"
    receipt["server"]["pid"] = 999
    write_json(Path(cfg["inference"]["receipt"]), receipt)
    assert (
        run_experiment(cfg, ROOT / "configs/remote.toml", split="validation", limit=1, seeds=[42])
        == 0
    )
    assert len(state["posts"]) == calls + 1  # readiness smoke only; completed workers skipped
    assert len(list(root.iterdir())) == 1
    receipt["model"]["gguf_sha256"] = "d" * 64
    write_json(Path(cfg["inference"]["receipt"]), receipt)
    with pytest.raises(ValueError, match="different model"):
        run_experiment(cfg, ROOT / "configs/remote.toml", split="validation", limit=1, seeds=[42])


def test_standalone_launcher_checks_flags_hashes_and_loopback(tmp_path):
    binary = tmp_path / "fake-server"
    flags = "--model --alias --host --port --ctx-size --parallel --threads --threads-batch --gpu-layers --flash-attn --fit --no-context-shift --cache-ram --jinja --chat-template-kwargs --predict --temp --top-p --top-k --min-p --log-disable"
    binary.write_text(
        f"#!{sys.executable}\nimport sys\nif '--help' in sys.argv: print({flags!r})\nelif '--version' in sys.argv: print('version: 999 (aaaaaaaa)')\n"
    )
    binary.chmod(0o755)
    weights = tmp_path / "test weights.gguf"
    weights.write_bytes(b"synthetic test weights")
    receipt = tmp_path / "receipt.json"
    command = [
        sys.executable,
        "-S",
        str(ROOT / "scripts/remote_inference.py"),
        "--binary",
        str(binary),
        "--model",
        str(weights),
        "--model-name",
        "Qwen3-14B",
        "--quantization",
        "Q4_K_M",
        "--commit",
        "a" * 40,
        "--receipt",
        str(receipt),
        "--receipt-only",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    metadata = read_json(receipt)
    assert metadata["model"]["gguf_sha256"] == sha256(weights)
    assert metadata["server"]["binary_sha256"] == sha256(binary)
    args = metadata["command"]
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert json.loads(args[args.index("--chat-template-kwargs") + 1]) == {"enable_thinking": False}
    assert "--log-disable" in args
    binary.write_text(f"#!{sys.executable}\nprint('--help --version')\n")
    failure = subprocess.run(command, capture_output=True, text=True)
    assert failure.returncode == 1 and "lacks required options" in failure.stderr


def test_local_check_and_smoke_remain_supported_without_inference_section(
    remote, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import psutil

    import runtime

    cfg, _, _ = remote
    cfg.pop("inference")
    binary, model = tmp_path / "binary", tmp_path / "weights.gguf"
    binary.touch()
    model.write_bytes(b"local test model")
    cfg["model"].update(
        file=str(model), server_binary=str(binary), threads=3, gpu_layers=99, cache_ram_mib=0
    )
    root = tmp_path / "local-checkout"
    root.mkdir()
    (root / "third_party").symlink_to(ROOT / "third_party", target_is_directory=True)
    (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    write_json(
        root / "runtime/server.json",
        {
            "pid": 123,
            "create_time": 42,
            "model_config": cfg["model"],
            "model_sha256": sha256(model),
            "binary_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(runtime, "ROOT", root)
    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 42))
    result = check(cfg, smoke=True)
    assert result["mode"] == "local_server"
    assert result["server"]["pid"] == 123
    assert result["smoke_response"]["choices"][0]["message"]["content"] == "OK"


def test_demonstrations_can_move_to_tunnel_only_with_same_generation_identity(remote):
    from runtime import experiment_inputs

    cfg, _, _ = remote
    prepare_fixture(cfg)
    path = Path(cfg["data"]["prepared_dir"]) / "explanation-manifest.json"
    manifest = read_json(path)
    manifest["model"].update(file="/old-gpu/weights.gguf", base_url="http://127.0.0.1:8080/v1")
    write_json(path, manifest)
    experiment_inputs(cfg, ["exegen_code"], remote_server(cfg))
    manifest["model"]["chat_template_kwargs"] = {"enable_thinking": True}
    write_json(path, manifest)
    with pytest.raises(ValueError, match="different model/configuration"):
        experiment_inputs(cfg, ["exegen_code"], remote_server(cfg))
