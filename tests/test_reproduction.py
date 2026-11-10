import ast
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from common import ROOT, config, read_json, sha256, write_json
from evaluate import analyze_run, concept_metrics, paired_intervals, parse_exercises
from prepare import answer_list, entity_id, prepare_data
from runtime import llm_config, run_worker, server_command


def completion(content, prompt_tokens=20, completion_tokens=10):
    return {
        "id": "test-replay",
        "object": "chat.completion",
        "created": 0,
        "model": "exegen-local",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def exercise(concept="K_Test_CS"):
    return {
        "Exercise": "Which option is correct?",
        "Options": {"A": "One", "B": "Two"},
        "Answer": ["A"],
        "Concept": concept,
    }


def history():
    return [
        {
            "problem_id": "Pm_1",
            "user_id": "U_test",
            "content": "A question?",
            "option": '{"A":"One","B":"Two"}',
            "right_answer": '["A"]',
            "concept_id": "K_Test_CS",
            "is_correct": 0,
            "explanation": "",
            "knowledge_evidence": "K_Test_CS, ",
            "tokens": "[1.0, 0.0]",
        }
    ]


def job(tmp_path, method):
    cfg = config("configs/reproduction.toml")
    rows = [dict(history()[0], user_id="U_train", explanation="A training explanation")]
    pd.DataFrame(rows).to_csv(tmp_path / "demonstrations.csv", index=False)
    pd.DataFrame(columns=["c1", "c2", "ground_truth"]).to_csv(
        tmp_path / "relations.csv", index=False
    )
    return {
        "method": method,
        "student": "U_test",
        "seed": 42,
        "history": history(),
        "exercise_count": 10,
        "llm_config": llm_config(cfg, 42),
        "run_dir": str(tmp_path / method),
        "demonstrations_csv": str(tmp_path / "demonstrations.csv"),
        "relations_csv": str(tmp_path / "relations.csv"),
        "j_output.txt": str(ROOT / "prompts/j_output.txt"),
        "n_output.txt": str(ROOT / "prompts/n_output.txt"),
        "monitor": {"interval_seconds": 0.05, "server_pid": 0, "gpu_indices": []},
    }


def replay_worker(tmp_path, method, contents):
    replay = tmp_path / "replay.json"
    write_json(replay, [completion(content) for content in contents])
    settings = job(tmp_path, method)
    code = run_worker(settings, replay)
    folder = Path(settings["run_dir"])
    assert code == 0, (folder / "worker.log").read_text()
    return folder


def test_upstream_snapshot_and_prompt_literals():
    manifest = read_json(ROOT / "third_party/manifest.json")
    for name, expected in manifest["files"].items():
        assert sha256(ROOT / "third_party/exegen" / name) == expected
    for source, working in [
        ("process.py", "process.py"),
        ("experiment/baseline.py", "baseline.py"),
    ]:
        original = ast.parse((ROOT / "third_party/exegen" / source).read_text())
        patched = ast.parse((ROOT / "src/author" / working).read_text())

        def prompts(tree):
            return [
                ast.dump(keyword.value)
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                for keyword in node.keywords
                if keyword.arg == "system_message"
            ]

        assert prompts(original) == prompts(patched)
    expected = read_json(ROOT / "prompts/provenance.json")["icl_cot_paper.txt"]["sha256"]
    assert sha256(ROOT / "prompts/icl_cot_paper.txt") == expected


def test_retrieval_functions_are_copied_and_retain_selection_order():
    original = ast.parse((ROOT / "third_party/exegen/experiment/create_prompt.py").read_text())
    patched = ast.parse((ROOT / "src/author/create_prompt.py").read_text())

    def functions(tree):
        return [ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)]

    assert functions(original) == functions(patched)
    namespace = {}
    exec(compile(patched, "create_prompt.py", "exec"), namespace)
    bank = pd.DataFrame(
        [
            {"concept_id": "a", "is_correct": 0, "tokens": "[1, 0]", "name": "first wrong"},
            {"concept_id": "a", "is_correct": 0, "tokens": "[1, 0]", "name": "second wrong"},
            {"concept_id": "a", "is_correct": 1, "tokens": "[0, 1]", "name": "correct"},
        ]
    )
    records = pd.DataFrame(
        [{"concept_id": "a", "tokens": "[1, 0]"}, {"concept_id": "missing", "tokens": "[0, 1]"}]
    )
    hard, remaining = namespace["get_examples_by_concept"](bank, records)
    assert list(hard["name"]) == ["first wrong", "correct"]
    assert list(remaining["concept_id"]) == ["missing"]
    soft = namespace["get_examples_by_similarity"](bank.reset_index(drop=True), remaining)
    assert list(soft["name"]) == ["correct"]


def test_metrics_unique_concepts_and_original_list_weighting():
    records = [{"concept_id": "a", "is_correct": 0}, {"concept_id": "b", "is_correct": 1}]
    generated = [{"concept": "a"}, {"concept": "a"}, {"concept": "b"}]
    result = concept_metrics(records, generated)
    assert result["hr"] == 0.5
    assert result["recall"] == 1
    assert result["author_ecr"] == pytest.approx(2 / 3)
    assert concept_metrics(records, [])["hr"] is None
    assert concept_metrics(records[1:], generated)["recall"] is None
    assert concept_metrics(records, [{"concept": "invented"}])["unknown_concepts"] == ["invented"]


def test_json_and_paper_formats():
    text = "```json\n" + json.dumps({"exercises": [exercise()]}) + "\n```"
    assert len(parse_exercises(text)) == 1
    plain = (
        "1. Exercise: Which?\nOptions: 'A': 'One', 'B': 'Two'\nAnswer: ['A']\nConcept: K_Test_CS"
    )
    assert parse_exercises(plain)[0]["concept"] == "K_Test_CS"
    invalid = exercise()
    invalid["Answer"] = ["A", "B"]
    assert parse_exercises(json.dumps([invalid])) == []


@pytest.mark.parametrize("method", ["baseline_code", "icl_cot_paper"])
def test_single_agent_replay(tmp_path, method):
    reply = "```json\n" + json.dumps({"exercises": [exercise()] * 10}) + "\n```"
    folder = replay_worker(tmp_path, method, [reply])
    assert read_json(folder / "status.json")["status"] == "complete"
    assert len(read_json(folder / "result.json")["exercises"]) == 10
    analysis = analyze_run(folder)
    assert analysis["calls"] == 1
    assert analysis["total_tokens"] == 30
    assert analysis["ttft_seconds"] is None


def test_exegen_group_chat_replay_counts_selection_and_summary(tmp_path):
    reply = "```json\n" + json.dumps({"exercises": [exercise()] * 10}) + "\n```"
    folder = replay_worker(
        tmp_path,
        "exegen_code",
        [
            "The student's weak concept is K_Test_CS.",
            "ExerciseGenerator should speak next.",
            "ExerciseGenerator",
            reply,
            "stopChat\n" + reply,
            "Final exercise recommendation summary.",
        ],
    )
    assert read_json(folder / "status.json")["status"] == "complete"
    assert len(read_json(folder / "result.json")["exercises"]) == 10
    assert analyze_run(folder)["calls"] == 6
    assert (folder / "groupchat.json").exists()


def test_failed_replay_is_recorded(tmp_path):
    replay = tmp_path / "empty.json"
    write_json(replay, [])
    settings = job(tmp_path, "baseline_code")
    assert run_worker(settings, replay) != 0
    folder = Path(settings["run_dir"])
    assert read_json(folder / "status.json")["status"] == "failed"
    assert analyze_run(folder)["failed_calls"] == 1


def test_explanation_uses_copied_prompt(tmp_path):
    settings = job(tmp_path, "explanation")
    settings["record"] = history()[0]
    replay = tmp_path / "explanation.json"
    write_json(replay, [completion('{"explanation":"A test explanation."}')])
    assert run_worker(settings, replay) == 0
    assert (
        read_json(Path(settings["run_dir"]) / "result.json")["explanation"] == "A test explanation."
    )


def test_resource_analysis_missing_fields_and_energy(tmp_path):
    rows = [
        {
            "timestamp": 10,
            "worker": {"rss_bytes": 100},
            "server": {"rss_bytes": 200},
            "gpus": [{"uuid": "gpu0", "power_watts": 40, "memory_used_mib": 2000}],
        },
        {
            "timestamp": 12,
            "worker": {"rss_bytes": 120},
            "server": {"rss_bytes": 220},
            "gpus": [{"uuid": "gpu0", "power_watts": 60, "memory_used_mib": 2100}],
        },
    ]
    (tmp_path / "resources.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    analysis = analyze_run(tmp_path)
    assert analysis["server_peak_rss_bytes"] == 220
    assert analysis["gpu_device_observed_energy_joules"] == 100
    assert analysis["gpu_device_peak_memory_mib"] == 2100
    assert analysis["total_tokens"] is None
    assert analysis["gpu_device_mean_utilization_percent"] is None


def test_paired_bootstrap_groups_generation_seeds_by_student():
    rows = [
        {
            "method": method,
            "student": str(student),
            "seed": seed,
            "valid_set": True,
            "hr": 0.8 if method == "exegen_code" else 0.3,
            "recall": None,
        }
        for method in ("exegen_code", "icl_cot_paper")
        for student in range(3)
        for seed in (17, 42, 73)
    ]
    result = paired_intervals(rows)["metrics"]
    assert result["hr"]["students"] == 3
    assert result["hr"]["difference"] == pytest.approx(0.5)
    assert result["recall"]["ci95"] is None


def test_server_does_not_silently_shrink_context(tmp_path):
    cfg = config("configs/reproduction.toml")
    binary, weights = tmp_path / "llama-server", tmp_path / "model.gguf"
    binary.touch()
    weights.touch()
    cfg["model"].update(file=str(weights), server_binary=str(binary))
    command = server_command(cfg)
    assert command[command.index("--parallel") + 1] == "1"
    assert command[command.index("--cache-ram") + 1] == "0"
    assert "--no-context-shift" in command
    assert command[command.index("--fit") + 1] == "off"


def test_preparation_streams_real_schema_and_prevents_student_leakage(tmp_path):
    cfg = config("configs/reproduction.toml")
    raw, target = tmp_path / "raw", tmp_path / "prepared"
    cfg["data"].update(raw_dir=str(raw), prepared_dir=str(target), history_length=2)

    def jsonl(name, records):
        path = raw / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8"
        )

    jsonl(
        "relations/course-field.json",
        [{"course_id": 1, "course_name": "CS", "field": [cfg["data"]["field"]]}],
    )
    jsonl("entities/course.json", [{"id": "C_1", "resource": [{"resource_id": "Ex_1"}]}])
    jsonl(
        "entities/problem.json",
        [
            {
                "problem_id": i,
                "exercise_id": "Ex_1",
                "type": 1,
                "content": "Question",
                "option": {"A": "One", "B": "Two"},
                "answer": '[\\"A\\"]',
            }
            for i in (1, 2)
        ],
    )
    concept = "K_Test_" + cfg["data"]["field"]
    for name, contents in [
        ("relations/concept-problem.txt", f"{concept}\tPm_1\n{concept}\tPm_2\n"),
        ("relations/concept-course.txt", ""),
        ("entities/concept.json", ""),
    ]:
        path = raw / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    jsonl("prerequisites/cs.json", [{"c1": "a", "c2": "b", "ground_truth": 1}])
    jsonl(
        "relations/user-problem.json",
        [
            {
                "user_id": f"U_{student}",
                "problem_id": f"Pm_{i}",
                "is_correct": i % 2,
                "submit_time": f"2020-01-0{i} 10:00:00",
                "log_id": f"{student}_{i}",
            }
            for student in range(10)
            for i in (1, 2)
        ],
    )
    prepare_data(cfg)
    histories = pd.read_csv(target / "histories.csv")
    demos = pd.read_csv(target / "demonstrations.csv")
    assert len(histories) == 20
    assert set(demos["user_id"]) <= set(histories[histories["split"] == "train"]["user_id"])
    assert not set(demos["user_id"]) & set(histories[histories["split"] == "test"]["user_id"])
    assert answer_list('[\\"A\\"]') == ["A"]
    assert entity_id(42, "Pm_") == "Pm_42"


def test_cli_help_without_server():
    result = subprocess.run(
        [sys.executable, str(ROOT / "experiments/reproduce.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "download-data" in result.stdout and "evaluate" in result.stdout


def test_all_experts_and_regeneration_are_preserved(tmp_path):
    reply = "```json\n" + json.dumps({"exercises": [exercise()] * 10}) + "\n```"
    folder = replay_worker(
        tmp_path,
        "exegen_code",
        [
            "Weak concept: K_Test_CS.",
            "ExerciseGenerator next.",
            "ExerciseGenerator",
            reply,
            "Linguistic_Fluency_discriminator next.",
            "Linguistic_Fluency_discriminator",
            "Fluency approved.",
            "Knowledge_Concept_Coverage_discriminator next.",
            "Knowledge_Concept_Coverage_discriminator",
            "Coverage approved.",
            "Correctness_and_Reasonableness_discriminator next.",
            "Correctness_and_Reasonableness_discriminator",
            "Reject: fix option A.",
            "ExerciseGenerator next; revise option A.",
            "ExerciseGenerator",
            reply,
            "stopChat\n" + reply,
            "Final summary.",
        ],
    )
    analysis = analyze_run(folder)
    assert analysis["calls"] == 18
    assert analysis["regeneration_calls"] == 1
    assert analysis["stages"]["fluency_review"]["calls"] == 1
    assert analysis["stages"]["coverage_review"]["calls"] == 1
    assert analysis["stages"]["correctness_review"]["calls"] == 1
    assert read_json(folder / "result.json")["termination"] == "stopChat"


def test_actual_sdk_connects_to_local_http_model(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen = []
    reply = "```json\n" + json.dumps({"exercises": [exercise()] * 10}) + "\n```"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            payload = json.dumps(completion(reply)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        settings = job(tmp_path, "baseline_code")
        settings["llm_config"]["config_list"][0]["base_url"] = (
            f"http://127.0.0.1:{server.server_port}/v1"
        )
        code = run_worker(settings)
        folder = Path(settings["run_dir"])
        assert code == 0, (folder / "worker.log").read_text()
        assert len(seen) == 1
        assert seen[0]["model"] == "exegen-local"
        assert analyze_run(folder)["total_tokens"] == 30
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_downloader_resumes_and_verifies_checksum(tmp_path):
    import hashlib
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from downloads import download

    data = b"a small immutable download"
    ranges = []

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("ETag", '"version1"')
            self.end_headers()

        def do_GET(self):
            header = self.headers.get("Range")
            ranges.append(header)
            offset = int(header.removeprefix("bytes=").removesuffix("-")) if header else 0
            self.send_response(206 if header else 200)
            self.send_header("Content-Length", str(len(data) - offset))
            if header:
                self.send_header("Content-Range", f"bytes {offset}-{len(data) - 1}/{len(data)}")
            self.end_headers()
            self.wfile.write(data[offset:])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/file"
        target = tmp_path / "download"
        target.with_name("download.part").write_bytes(data[:5])
        write_json(
            target.with_name("download.part.json"),
            {"url": url, "validator": '"version1"', "size": len(data)},
        )
        expected = hashlib.sha256(data).hexdigest()
        assert download(url, target, expected)["sha256"] == expected
        assert target.read_bytes() == data
        assert ranges == ["bytes=5-"]
        download(url, target, expected)
        assert len(ranges) == 1
        target.write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="Checksum mismatch"):
            download(url, target, expected)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("tracking_enabled", [False, True])
def test_experiment_resume_and_offline_evaluation(tmp_path, monkeypatch, tracking_enabled):
    import runtime
    from evaluate import evaluate_experiment, export_experiment

    cfg = config("configs/reproduction.toml")
    if tracking_enabled:
        pytest.importorskip("mlflow")
    cfg["tracking"].update(
        mlflow=tracking_enabled, backend="sqlite", directory=str(tmp_path / "mlruns")
    )
    cfg["data"]["prepared_dir"] = str(tmp_path / "prepared")
    cfg["experiment"].update(output_dir=str(tmp_path / "outputs"), seeds=[42], student_limit=1)
    prepared = Path(cfg["data"]["prepared_dir"])
    prepared.mkdir()
    rows = [dict(row, split="test") for row in history()]
    pd.DataFrame(rows).to_csv(prepared / "stuRec_1000_with_tokens.csv", index=False)
    pd.DataFrame(
        [dict(row, user_id="U_train", explanation="Training", split="train") for row in history()]
    ).to_csv(prepared / "examples_with_explanation_with_tokens.csv", index=False)
    pd.DataFrame(columns=["c1", "c2", "ground_truth"]).to_csv(
        prepared / "concept_relationship_filtered.csv", index=False
    )
    write_json(prepared / "manifest.json", {"test": True})
    write_json(
        prepared / "embedding-manifest.json",
        {
            "inputs": {},
            "outputs": {
                "stuRec_1000_with_tokens.csv": sha256(prepared / "stuRec_1000_with_tokens.csv")
            },
        },
    )
    server = {"pid": 123, "model_sha256": "test", "binary_sha256": "test"}
    write_json(
        prepared / "explanation-manifest.json",
        {
            "model": cfg["model"],
            "server": server,
            "demonstrations_sha256": sha256(prepared / "examples_with_explanation_with_tokens.csv"),
        },
    )
    monkeypatch.setattr(runtime, "check", lambda cfg: {"server": server})
    calls = []

    def fake_worker(settings):
        calls.append(settings["method"])
        folder = Path(settings["run_dir"])
        write_json(folder / "status.json", {"status": "complete", "elapsed_seconds": 1.0})
        write_json(
            folder / "result.json", {"exercises": parse_exercises(json.dumps([exercise()] * 10))}
        )
        return 0

    monkeypatch.setattr(runtime, "run_worker", fake_worker)
    path = ROOT / "configs/reproduction.toml"
    assert runtime.run_experiment(cfg, path) == 0
    assert calls == ["icl_cot_paper", "exegen_code"]
    runtime.run_experiment(cfg, path)
    assert len(calls) == 2
    experiment = next((tmp_path / "outputs").iterdir())
    if tracking_enabled:
        from tracking import client_and_experiment

        client, experiment_id, _ = client_and_experiment(cfg, cfg["experiment"]["name"])
        runs = client.search_runs([experiment_id])
        assert len(runs) == 3  # one experiment and two attempts, including after resume
        assert all(run.info.status == "FINISHED" for run in runs)
        parent = next(run for run in runs if run.data.tags["exegen.kind"] == "experiment")
        assert parent.data.metrics["exegen_code.means.hr"] == 1
        assert client.list_artifacts(parent.info.run_id, "metrics")
    summaries = evaluate_experiment(experiment)
    assert summaries["exegen_code"]["means"]["hr"] == 1
    export_experiment(experiment, tmp_path / "export.tar.gz")
    assert (tmp_path / "export.tar.gz").stat().st_size > 0


@pytest.mark.parametrize("retrieval_files", ["absent", "corrupt"])
def test_paper_experiment_needs_only_plain_history(tmp_path, monkeypatch, retrieval_files):
    import runtime

    cfg = config("configs/reproduction.toml")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    cfg["data"]["prepared_dir"] = str(prepared)
    cfg["experiment"].update(output_dir=str(tmp_path / "outputs"), seeds=[42], student_limit=1)
    plain = [
        {
            k: row[k]
            for k in ("user_id", "content", "option", "right_answer", "concept_id", "is_correct")
        }
        for row in history()
    ]
    pd.DataFrame([dict(row, split="test") for row in plain]).to_csv(
        prepared / "histories.csv", index=False
    )
    unrelated = [
        "stuRec_1000_with_tokens.csv",
        "examples_with_explanation_with_tokens.csv",
        "concept_relationship_filtered.csv",
        "manifest.json",
        "embedding-manifest.json",
        "explanation-manifest.json",
    ]
    if retrieval_files == "corrupt":
        for name in unrelated:
            (prepared / name).write_text("deliberately invalid")
    server = {"pid": 0, "model_sha256": "test", "binary_sha256": "test"}
    monkeypatch.setattr(runtime, "check", lambda cfg: {"server": server})
    replay = tmp_path / "replay.json"
    write_json(replay, [completion(json.dumps([exercise()] * 10))])
    worker = runtime.run_worker
    calls = []

    def replay_worker(settings):
        calls.append(settings)
        assert "tokens" not in settings["history"][0]
        assert (
            not {"demonstrations_csv", "relations_csv", "j_output.txt", "n_output.txt"}
            & settings.keys()
        )
        return worker(settings, replay)

    monkeypatch.setattr(runtime, "run_worker", replay_worker)
    assert (
        runtime.run_experiment(cfg, ROOT / "configs/reproduction.toml", methods=["icl_cot_paper"])
        == 0
    )
    experiment = next((tmp_path / "outputs").iterdir())
    assert set(read_json(experiment / "manifest.json")["identity"]["inputs"]) == {"histories.csv"}
    # Retrieval artifacts must not affect paper-only validation, fingerprints, or resumption.
    for name in unrelated:
        (prepared / name).write_text("changed unrelated artifact")
    assert (
        runtime.run_experiment(cfg, ROOT / "configs/reproduction.toml", methods=["icl_cot_paper"])
        == 0
    )
    assert len(calls) == 1
    assert len(list((tmp_path / "outputs").iterdir())) == 1


@pytest.mark.parametrize(
    "methods",
    [
        ["exegen_code"],
        ["baseline_code"],
        ["icl_cot_paper", "exegen_code"],
        ["icl_cot_paper", "baseline_code"],
    ],
)
def test_retrieval_methods_still_require_full_preparation(tmp_path, methods):
    from runtime import experiment_inputs

    cfg = config("configs/reproduction.toml")
    cfg["data"]["prepared_dir"] = str(tmp_path)
    pd.DataFrame([dict(history()[0], split="test")]).to_csv(tmp_path / "histories.csv", index=False)
    with pytest.raises(ValueError, match="embedding-manifest.json.*explanation-manifest.json"):
        experiment_inputs(cfg, methods, {"model_sha256": "test"})


def test_paper_reports_missing_plain_history_without_retrieval_instructions(tmp_path):
    from runtime import experiment_inputs

    cfg = config("configs/reproduction.toml")
    cfg["data"]["prepared_dir"] = str(tmp_path)
    with pytest.raises(ValueError, match="histories.csv") as exc:
        experiment_inputs(cfg, ["icl_cot_paper"], {"model_sha256": "test"})
    assert "run prepare" in str(exc.value)
    assert "embed" not in str(exc.value) and "explain" not in str(exc.value)


def test_optional_performance_plot_without_gpu(tmp_path):
    pytest.importorskip("matplotlib")
    from plots import plot_run

    (tmp_path / "requests.jsonl").write_text(
        json.dumps(
            {
                "status": "ok",
                "call": 1,
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "elapsed_seconds": 0.1,
            }
        )
        + "\n"
    )
    output = plot_run(tmp_path)
    assert output.read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("config_path", ["configs/reproduction.toml", "configs/remote.toml"])
def test_cpu_bert_embedding_and_cache_with_tiny_local_checkpoint(tmp_path, config_path):
    torch = pytest.importorskip("torch")
    from transformers import BertConfig, BertModel, BertTokenizer

    from prepare import embed

    cfg = config(config_path)
    prepared, weights = tmp_path / "prepared", tmp_path / "bert"
    prepared.mkdir()
    weights.mkdir()
    cfg["data"].update(prepared_dir=str(prepared), bert_dir=str(weights))
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "question", "?"]
    (weights / "vocab.txt").write_text("\n".join(vocabulary) + "\n")
    torch.manual_seed(42)
    model = BertModel(
        BertConfig(
            vocab_size=len(vocabulary),
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    ).eval()
    model.save_pretrained(weights)
    write_json(weights / "source.json", {"repository": "synthetic-test", "revision": "test"})
    frame = pd.DataFrame([dict(history()[0], course_name="CS", knowledge_chain="", split="test")])
    frame.to_csv(prepared / "histories.csv", index=False)
    frame.to_csv(prepared / "demonstrations.csv", index=False)
    embed(cfg)
    result = pd.read_csv(prepared / "stuRec_1000_with_tokens.csv", keep_default_na=False)
    sentence = "".join(
        str(frame.iloc[0][k])
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
    tokenizer = BertTokenizer(vocab_file=str(weights / "vocab.txt"))
    with torch.inference_mode():
        expected = (
            model(**tokenizer.encode_plus(sentence, return_tensors="pt"))[0].squeeze(0).sum(dim=0)
        )
    assert json.loads(result.iloc[0]["tokens"]) == pytest.approx(expected.tolist(), abs=1e-5)
    before = (prepared / "embedding-cache.jsonl").read_bytes()
    embed(cfg)
    assert (prepared / "embedding-cache.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    "context,key",
    [
        (
            {"mode": "raw_confidence", "values": [{"attempt_id": "a", "confidence": 3}]},
            "historical_confidence",
        ),
        (
            {
                "mode": "structured_confidence",
                "concepts": [{"concept_id": "K_Test_CS", "error_risk": 0.7}],
            },
            "confidence_profile",
        ),
    ],
)
def test_exegen_receives_optional_confidence_context_in_diagnosis_and_generation(
    tmp_path, context, key
):
    reply = "```json\n" + json.dumps({"exercises": [exercise()] * 10}) + "\n```"
    settings = job(tmp_path, "exegen_code")
    settings["confidence_context"] = context
    replay = tmp_path / "replay.json"
    write_json(
        replay,
        [
            completion(text)
            for text in [
                "Weak concept K_Test_CS.",
                "ExerciseGenerator next.",
                "ExerciseGenerator",
                reply,
                "stopChat\n" + reply,
                "Final summary.",
            ]
        ],
    )
    code = run_worker(settings, replay)
    folder = Path(settings["run_dir"])
    assert code == 0, (folder / "worker.log").read_text()
    assert key in read_json(folder / "retrieved_input.json")
    calls = [json.loads(line) for line in (folder / "requests.jsonl").read_text().splitlines()]
    for stage in ("knowledge_perception", "exercise_generation"):
        selected = [r for r in calls if r["stage"] == stage]
        assert len(selected) == 1
        assert key in json.dumps(selected[0]["request"]["messages"])
