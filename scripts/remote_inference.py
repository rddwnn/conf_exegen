"""Standalone remote GPU launcher: Python stdlib + llama-server + GGUF only."""

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def version_commit(binary, supplied=None):
    version = subprocess.check_output(
        [str(binary), "--version"], text=True, stderr=subprocess.STDOUT
    ).strip()
    if not version:
        raise ValueError("llama-server --version returned no version metadata")
    match = re.search(r"(?:version:[^\n]*\(|commit[:= ]+)([a-fA-F0-9]{7,40})", version)
    reported = match.group(1).lower() if match else None
    commit, source = reported, "binary_version"
    if supplied:
        if not re.fullmatch(r"[a-fA-F0-9]{7,40}", supplied):
            raise ValueError("--commit must be a real llama.cpp revision, not a release label")
        supplied = supplied.lower()
        if reported and not (supplied.startswith(reported) or reported.startswith(supplied)):
            raise ValueError("Supplied commit differs from llama-server --version")
        commit, source = (
            supplied,
            "operator_supplied_matching_version_prefix"
            if reported
            else "operator_supplied_unverified",
        )
    if not commit:
        raise ValueError(
            "Binary does not report its commit; supply --commit from its build checkout"
        )
    return version, commit, source


def setup(args):
    binary, model = (
        Path(args.binary).expanduser().resolve(),
        Path(args.model).expanduser().resolve(),
    )
    if not binary.is_file() or not model.is_file():
        raise ValueError("The GPU host needs an existing llama-server binary and GGUF file")
    if not 1 <= args.port <= 65535 or args.context_size <= args.max_tokens or args.threads < 1:
        raise ValueError("Invalid port/context/output/thread settings")
    if (
        args.max_tokens <= 0
        or args.top_k < 1
        or not 0 < args.top_p <= 1
        or not 0 <= args.min_p <= 1
        or not math.isfinite(args.temperature)
        or args.temperature < 0
        or not args.alias.strip()
        or not args.model_name.strip()
        or not args.quantization.strip()
    ):
        raise ValueError("Invalid model identity or generation settings")
    parameters = {
        "host": "127.0.0.1",
        "port": args.port,
        "context_size": args.context_size,
        "parallel": 1,
        "threads": args.threads,
        "gpu_layers": args.gpu_layers,
        "flash_attention": "auto",
        "fit": "off",
        "context_shift": False,
        "cache_ram_mib": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "logging": False,
    }
    generation = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
    }
    command = [
        str(binary),
        "--model",
        str(model),
        "--alias",
        args.alias,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--ctx-size",
        str(args.context_size),
        "--parallel",
        "1",
        "--threads",
        str(args.threads),
        "--threads-batch",
        str(args.threads),
        "--gpu-layers",
        str(args.gpu_layers),
        "--flash-attn",
        "auto",
        "--fit",
        "off",
        "--no-context-shift",
        "--cache-ram",
        "0",
        "--jinja",
        "--chat-template-kwargs",
        json.dumps({"enable_thinking": False}),
        "--predict",
        str(args.max_tokens),
        "--temp",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--min-p",
        str(args.min_p),
        "--log-disable",
    ]
    help_text = subprocess.check_output(
        [str(binary), "--help"], text=True, stderr=subprocess.STDOUT
    )
    unsupported = [flag for flag in command[1:] if flag.startswith("--") and flag not in help_text]
    if unsupported:
        raise ValueError(
            f"Installed llama-server lacks required options: {unsupported}; use a compatible pinned build"
        )
    version, commit, source = version_commit(binary, args.commit)
    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": args.model_name,
            "alias": args.alias,
            "gguf_sha256": digest(model),
            "quantization": args.quantization,
            "path": str(model),
        },
        "server": {
            "version": version,
            "commit": commit,
            "commit_source": source,
            "binary_sha256": digest(binary),
            "runtime_parameters": parameters,
        },
        "generation": generation,
        "command": command,
        "verification": {
            "hashes": "Computed from files on this host by SHA-256",
            "name_and_quantization": "Operator supplied; retain the original GGUF publisher reference",
            "live_api_binding": "Not cryptographic attestation; copy this receipt over SSH and validate the endpoint",
        },
    }
    path = Path(args.receipt).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as output:
        json.dump(receipt, output, indent=2)
        output.write("\n")
        temporary = Path(output.name)
    temporary.replace(path)
    return command, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-name", required=True, help="Exact publisher/model name, e.g. Qwen3-14B"
    )
    parser.add_argument(
        "--quantization", required=True, help="Exact GGUF quantization, e.g. Q4_K_M"
    )
    parser.add_argument("--alias", default="exegen-local")
    parser.add_argument("--receipt", default="remote-server.json")
    parser.add_argument(
        "--commit", help="Full build revision; checked against --version when possible"
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--context-size", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument(
        "--receipt-only",
        action="store_true",
        help="Record setup without launching; not a readiness proof",
    )
    args = parser.parse_args(argv)
    command, receipt = setup(args)
    print(f"Saved remote receipt: {Path(args.receipt).resolve()}", flush=True)
    print(f"Model SHA-256: {receipt['model']['gguf_sha256']}", flush=True)
    if args.receipt_only:
        return 0
    # Never persist request bodies, student histories, or completions on the GPU host.
    # Clear ambient logging/prompt-cache settings before starting the private server.
    import os

    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("LLAMA_ARG_")
    }
    with open(os.devnull, "w") as sink:
        child = subprocess.Popen(command, stdout=sink, stderr=sink, env=environment)
        try:
            return child.wait()
        except KeyboardInterrupt:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            return 130


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
