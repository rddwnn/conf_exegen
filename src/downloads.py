"""Explicit, resumable downloads; experiments never fetch inputs implicitly."""

import re
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

from common import ROOT, resolve, sha256, write_json

DATA_BASE = "https://lfs.aminer.cn/misc/moocdata/data/mooccube2/"
DATA_FILES = (
    "relations/course-field.json",
    "entities/course.json",
    "entities/problem.json",
    "relations/concept-problem.txt",
    "prerequisites/cs.json",
    "relations/user-problem.json",
)
PAPER_URL = (
    "https://papers.neurips.cc/paper_files/paper/2025/file/"
    "85dbd2fb8b355e4231b51e454c08ec1c-Paper-Conference.pdf"
)


def download(url, target, expected_sha256=None):
    """Resume only with an unchanged HTTP validator; hash every completed download."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = target.with_name(target.name + ".download.json")
    if target.exists():
        digest = sha256(target)
        if expected_sha256 and digest != expected_sha256:
            raise ValueError(f"Checksum mismatch: {target}")
        if not metadata_path.exists():
            raise ValueError(f"Unverified existing file: {target}; remove it or supply provenance")
        from common import read_json

        previous = read_json(metadata_path)
        if previous["url"] != url or previous["sha256"] != digest:
            raise ValueError(f"Existing download provenance mismatch: {target}")
        return previous
    request = Request(url, method="HEAD")
    with urlopen(request, timeout=60) as response:
        size = int(response.headers.get("Content-Length", 0))
        validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
    part = target.with_name(target.name + ".part")
    part_meta = part.with_name(part.name + ".json")
    offset = 0
    if part.exists() and part_meta.exists() and validator:
        from common import read_json

        previous = read_json(part_meta)
        if previous == {"url": url, "validator": validator, "size": size}:
            offset = part.stat().st_size
    headers = {"Range": f"bytes={offset}-", "If-Range": validator} if offset else {}
    write_json(part_meta, {"url": url, "validator": validator, "size": size})
    print(f"Downloading {url} (resume offset {offset:,}; expected {size:,} bytes)", flush=True)
    if not (size and offset == size):
        with urlopen(Request(url, headers=headers), timeout=120) as response:
            if offset and response.status != 206:
                offset = 0
            if offset and not response.headers.get("Content-Range", "").startswith(
                f"bytes {offset}-"
            ):
                raise ValueError("Unexpected HTTP range response; refusing to append")
            with part.open("ab" if offset else "wb") as output:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
    if size and part.stat().st_size != size:
        raise ValueError(f"Incomplete download, kept for resume: {part}")
    digest = sha256(part)
    if expected_sha256 and digest != expected_sha256:
        part.unlink(missing_ok=True)
        part_meta.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch: {part}")
    metadata = {"url": url, "sha256": digest, "bytes": part.stat().st_size, "validator": validator}
    part.replace(target)
    write_json(metadata_path, metadata)
    part_meta.unlink(missing_ok=True)
    return metadata


def fetch_data(cfg, only=None):
    selected = DATA_FILES if not only else [x for x in DATA_FILES if x in only]
    if only and len(selected) != len(set(only)):
        raise ValueError("Unknown dataset filename")
    for name in selected:
        download(DATA_BASE + name, resolve(cfg["data"]["raw_dir"]) / name)


def fetch_bert(cfg):
    from huggingface_hub import HfApi, snapshot_download

    settings = cfg["data"]
    revision = (
        HfApi().model_info(settings["bert_repo"], revision=settings["bert_revision"] or "main").sha
    )
    target = resolve(settings["bert_dir"])
    snapshot_download(
        settings["bert_repo"],
        revision=revision,
        local_dir=target,
        allow_patterns=[
            "config.json",
            "vocab.txt",
            "tokenizer_config.json",
            "tokenizer.json",
            "model.safetensors",
        ],
    )
    write_json(
        target / "source.json",
        {
            "repository": settings["bert_repo"],
            "revision": revision,
            "files": {
                str(p.relative_to(target)): sha256(p)
                for p in target.iterdir()
                if p.is_file() and p.name != "source.json"
            },
        },
    )


def fetch_model(url, checksum, cfg):
    if not re.fullmatch(r"[a-fA-F0-9]{64}", checksum):
        raise ValueError("Supply the GGUF file's SHA-256 from its publisher")
    if not cfg["model"]["file"]:
        raise ValueError("Set model.file in the configuration first")
    return download(url, resolve(cfg["model"]["file"]), checksum.lower())


def build_server(cfg):
    revision = cfg["model"]["server_revision"]
    if not revision or revision in ("main", "master", "latest"):
        raise ValueError("Set model.server_revision to a pinned llama.cpp commit or release")
    target = ROOT / "runtime" / "llama.cpp"
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "https://github.com/ggml-org/llama.cpp.git",
                str(target),
            ],
            check=True,
        )
    subprocess.run(["git", "fetch", "origin", revision], cwd=target, check=True)
    subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=target, check=True)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(target),
            "-B",
            str(target / "build"),
            "-DGGML_CUDA=ON",
            "-DLLAMA_CURL=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(target / "build"),
            "--target",
            "llama-server",
            "-j",
            str(cfg["model"]["threads"]),
        ],
        check=True,
    )
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
    write_json(
        target.parent / "llama-build.json",
        {
            "revision": actual,
            "requested_revision": revision,
            "binary_sha256": sha256(target / "build/bin/llama-server"),
        },
    )
