"""Per-request traces and bounded process/device sampling for local inference."""

import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from common import append_json


def request_stage(messages):
    """Classify preserved author prompts without modifying their contents."""
    system = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    last = str(messages[-1].get("content", "")) if messages else ""
    if "Selects the next speaking agent" in system or "select the next role" in system.lower():
        return "speaker_selection"
    if last.startswith("Summarize the takeaway from the conversation."):
        return "summary"
    checks = [
        ("You are an interpreter", "demonstration_explanation"),
        ("You are a knowledge tracking expert", "knowledge_perception"),
        ("You are the moderator", "manager"),
        ("specializing in assessing linguistic fluency", "fluency_review"),
        ("specializing in assessing the coverage", "coverage_review"),
        ("specializing in assessing the correctness", "correctness_review"),
        ("You are an exercise generation expert", "exercise_generation"),
    ]
    for text, stage in checks:
        if text in system:
            return stage
    if "You are an expert in generating single-choice exercises" in last:
        return "exercise_generation"
    return "other"


class ResourceMonitor:
    """GPU measurements are device-wide; RSS/CPU are explicitly per process tree."""

    def __init__(self, path, settings):
        self.path = Path(path)
        self.settings = settings
        self.stop_event = threading.Event()
        self.thread = None
        self.processes = {}

    def process_sample(self, pid):
        import psutil

        try:
            root = psutil.Process(pid)
            members = [root, *root.children(recursive=True)]
            rss, cpu, initialized = 0, 0.0, True
            for process in members:
                key = (process.pid, process.create_time())
                tracked = self.processes.setdefault(key, process)
                # First psutil observation initializes its CPU delta; report CPU only later.
                first = not hasattr(tracked, "_exegen_seen")
                percent = tracked.cpu_percent()
                tracked._exegen_seen = True
                rss += tracked.memory_info().rss
                cpu += percent
                initialized = initialized and not first
            return {
                "pid": pid,
                "rss_bytes": rss,
                "cpu_percent": cpu if initialized else None,
                "process_count": len(members),
            }
        except (psutil.Error, OSError) as exc:
            return {"pid": pid, "rss_bytes": None, "cpu_percent": None, "error": str(exc)}

    def sample(self):
        timestamp = time.time()
        result = {
            "timestamp": timestamp,
            "worker": self.process_sample(os.getpid()),
            "server": None,
            "gpus": [],
            "gpu_error": None,
        }
        import psutil

        memory = psutil.virtual_memory()
        result["host_memory"] = {"available_bytes": memory.available, "total_bytes": memory.total}
        pid = self.settings.get("server_pid", 0)
        if pid:
            result["server"] = self.process_sample(pid)
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
            for line in completed.stdout.splitlines():
                values = [part.strip() for part in line.split(",")]
                if int(values[0]) not in self.settings.get("gpu_indices", [0]):
                    continue

                def number(value):
                    try:
                        return float(value)
                    except ValueError:
                        return None

                result["gpus"].append(
                    {
                        "index": int(values[0]),
                        "uuid": values[1],
                        "scope": "device",
                        "utilization_percent": number(values[2]),
                        "memory_used_mib": number(values[3]),
                        "memory_total_mib": number(values[4]),
                        "power_watts": number(values[5]),
                    }
                )
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            result["gpu_error"] = str(exc)
        append_json(self.path, result)

    def __enter__(self):
        def work():
            while not self.stop_event.is_set():
                try:
                    self.sample()
                except Exception as exc:
                    append_json(self.path, {"timestamp": time.time(), "monitor_error": str(exc)})
                self.stop_event.wait(self.settings.get("interval_seconds", 1.0))

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=5)


@contextmanager
def trace_completions(path, replay_path=None):
    """Intercept AutoGen's actual SDK calls, including speaker selection and summary."""
    from openai.resources.chat.completions import Completions
    from openai.types.chat import ChatCompletion

    original = Completions.create
    replay = None
    if replay_path:
        replay = iter(json.loads(Path(replay_path).read_text(encoding="utf-8")))
    count = 0

    def create(client, *args, **kwargs):
        nonlocal count
        count += 1
        started = time.time()
        timer = time.perf_counter()
        record = {
            "call": count,
            "started_at": started,
            "request": kwargs,
            "ttft_seconds": None,
            "streaming": False,
            "stage": request_stage(kwargs.get("messages", [])),
        }
        try:
            if kwargs.get("stream"):
                raise ValueError("Streaming is disabled to preserve the author's request path")
            if replay is None:
                response = original(client, *args, **kwargs)
            else:
                response = ChatCompletion.model_validate(next(replay))
            elapsed = time.perf_counter() - timer
            payload = response.model_dump(mode="json")
            usage = payload.get("usage") or {}
            completion_tokens = usage.get("completion_tokens")
            record.update(
                {
                    "status": "ok",
                    "elapsed_seconds": elapsed,
                    "response": payload,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": completion_tokens,
                    "total_tokens": usage.get("total_tokens"),
                    "effective_output_tokens_per_second": (
                        completion_tokens / elapsed
                        if completion_tokens is not None and elapsed
                        else None
                    ),
                    "server_timings": payload.get("timings"),
                }
            )
            append_json(path, record)
            return response
        except BaseException as exc:
            record.update(
                {
                    "status": "error",
                    "elapsed_seconds": time.perf_counter() - timer,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            append_json(path, record)
            raise

    Completions.create = create
    try:
        yield
        if replay is not None:
            sentinel = object()
            if next(replay, sentinel) is not sentinel:
                raise ValueError("Replay contains unused responses; execution path changed")
    finally:
        Completions.create = original
