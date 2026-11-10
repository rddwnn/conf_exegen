# ExeGen reproduction

Run the published ExeGen agents and ICL+CoT on MOOCCubeX using one local LLM.
This is an adapted reproduction: GPT-4o is replaced by an explicitly chosen GGUF
model served by `llama-server`. Confidence-aware diagnosis is a later research stage.

- [Paper and Appendix B.2](https://papers.neurips.cc/paper_files/paper/2025/file/85dbd2fb8b355e4231b51e454c08ec1c-Paper-Conference.pdf)
- [Author implementation](https://github.com/dsz532/exeGen/tree/c5a45afa9f1541a33e7034271b75b64fa8291526)
- [MOOCCubeX](https://github.com/THU-KEG/MOOCCubeX)
- [Provenance, patches, and evaluation protocol](docs/reproduction.md)
- [Model-run telemetry definitions](docs/telemetry.md)

`third_party/exegen/` is an unchanged source snapshot with SHA-256 checksums.
`src/author/` contains generated working copies and reviewable diffs. New control
code is in `src/`, prompts in `prompts/`, and the CLI in `experiments/reproduce.py`.
This uv workspace is not an installable Python package.

## New Linux machine

Install Git, uv, and a compatible NVIDIA driver. Building llama.cpp additionally
requires CMake, a C/C++ compiler, and the CUDA toolkit. Allow disk space for the
22.5 GB responses file, approximately 1.5 GB of other raw inputs, your GGUF, and
results. Processing streams the raw files rather than loading them into RAM together.

After cloning and entering the repository:

```bash
uv sync --frozen --extra reproduction --extra prepare --extra analysis
uv run --frozen --extra reproduction python experiments/reproduce.py check --offline
```

Python 3.11 and dependencies are pinned. On Linux, the `prepare` extra uses
CPU-only PyTorch for BERT; generative inference runs in the separate llama.cpp
CUDA binary. The optional `research` extra is for future work and MLflow.

Edit `configs/reproduction.toml`: set `model.file` to your intended GGUF path,
for example `models/model.gguf`, and record the chosen model/quantization. No
LLM has been silently selected. BERT and llama.cpp source revisions are pinned.
All methods must use the same model file and decoding settings.

## Downloads and server build

```bash
uv run --frozen python experiments/reproduce.py download-model --url '<GGUF_URL>' --sha256 '<SHA256>'
uv run --frozen python experiments/reproduce.py build-server
uv run --frozen python experiments/reproduce.py download-data
uv run --frozen --extra prepare python experiments/reproduce.py download-bert
```

Use a publisher-provided GGUF URL and checksum. An existing local GGUF can also
be used; the server records its actual hash. To use a compatible prebuilt server,
set `model.server_binary` instead of building. Its version and hash are recorded.

Dataset downloads use `.part` files, HTTP validators, and recorded SHA-256 hashes.
Rerun the download command to resume interruptions or verify complete files.
Use `download-data --only relations/course-field.json` to test a small download.
Models, datasets, server builds, caches, and results are ignored by Git.
Experiments do not download inputs implicitly.

## Data and demonstrations

```bash
uv run --frozen python experiments/reproduce.py prepare
uv run --frozen --extra prepare python experiments/reproduce.py embed
```

Preparation reconstructs unpublished upstream CSVs and records source/split
manifests. Embedding preserves the original BERT concatenation and sum-pooling,
on CPU. Overlength BERT inputs fail explicitly instead of being silently truncated.

Start the server in one terminal:

```bash
uv run --frozen python experiments/reproduce.py serve
```

It serves one model/request at a time on `127.0.0.1:8080`, uses at most three CPU
threads, disables automatic fitting/context shifting/additional host prompt cache,
and writes `runtime/server.json` and `runtime/server.log`. Stop it with Ctrl-C.

In another terminal:

```bash
uv run --frozen --extra reproduction python experiments/reproduce.py check --smoke
uv run --frozen --extra reproduction python experiments/reproduce.py explain
```

Explanation generation uses the copied author prompt and the same local model.
Each completed explanation is persisted. To change its model/settings, rerun
`embed` to reset the derived demonstration inputs before explaining again.
Keep the preparation logs for full cost accounting. BERT need not remain loaded.

The `openai` dependency is the HTTP client used by the authors' AutoGen version.
It connects only to the configured local server; no OpenAI service key is needed.

## Pilot and comparison

```bash
uv run --frozen --extra reproduction python experiments/reproduce.py run --split validation --limit 10 --seeds 42
```

Inspect failures, context fit, RAM/VRAM, and cost before the main run:

```bash
uv run --frozen --extra reproduction python experiments/reproduce.py run
```

Defaults: up to 100 test students, seeds 17/42/73, ten single-choice exercises,
and methods `icl_cot_paper` and `exegen_code`. The original generator baseline is
available separately with `run --methods baseline_code --split validation --limit 1 --seeds 42`.
It is not assumed equivalent to the paper's ICL+CoT.

Identical commands resume the same fingerprinted experiment: complete jobs are
skipped; failed/interrupted jobs get new attempts. Old attempts remain on disk.
Interrupted dialogs restart rather than restoring a partial AutoGen conversation.
A completed but invalid output is reported without hidden generation retries.

## Results and model-run analysis

Artifacts are saved under `outputs/<name>-<fingerprint>/`, grouped by method,
student, seed, and attempt. They include inputs, all actual API requests/responses,
resource samples, results, status, and logs. ExeGen saves the GroupChat and summary.
The manifest fingerprints code, prompts, inputs, model, binary, and configuration.

Recompute metrics without loading a model:

```bash
uv run --frozen python experiments/reproduce.py evaluate outputs/<experiment-directory>
```

Outputs include `metrics/summary.json`, `runs.json`, `runs.csv`, `report.md`, and a
blind-review CSV with a separate method key. They cover HR/Recall over unique
concepts, original list-weighted ECR, valid-set/completion counts, unknown concepts,
and paired student bootstrap intervals. Undefined metrics stay null, not zero.

Analyze an individual attempt and render a standalone performance figure:

```bash
uv run --frozen --extra analysis python experiments/reproduce.py analyze-run outputs/<experiment-directory>/runs/exegen_code/<student>/<seed>/attempt-0001 --plots
```

This creates `performance.json` and `performance.png`: input/output tokens,
latency, VRAM, process RAM, and CPU/GPU load. Analysis also includes per-role
calls/tokens/time, regeneration count, latency quantiles, maximum prompt tokens,
and sampled device power/energy. Missing measurements are explicitly unavailable.

Enable `[tracking].mlflow` and install the `research` extra for optional local
MLflow tracking. JSON/CSV artifacts remain available without a tracking server.

Archive results before releasing the compute machine:

```bash
uv run --frozen python experiments/reproduce.py export outputs/<experiment-directory> artifacts/results.tar.gz
```

The archive contains the experiment, not raw data, model weights, or preparation
logs under `data/prepared/`; retain those separately as needed.

## Verification

```bash
uv sync --frozen --extra reproduction --extra analysis --group dev
uv run --frozen --extra reproduction --extra analysis pytest -q
uv run --frozen ruff check src scripts experiments tests
uv run --frozen ruff format --check src scripts experiments tests
```

Tests use synthetic records, recorded completions, and a local HTTP fixture.
They need neither a GPU nor model/dataset downloads. Hardware fit, actual model
quality, and complete MOOCCubeX preparation still require the target-machine pilot.
