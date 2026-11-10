# Remote GPU inference with local orchestration

The workstation retains MOOCCubeX, prepared CSVs, BERT, demonstrations, AutoGen,
request traces, evaluation, and results. The rented GPU host runs only llama.cpp
and the GGUF. Requests travel through an SSH tunnel to a loopback-only API.
This is the same experiment pipeline, not a second remote experiment runner.

Do not update the checkout/environment backing a currently running experiment.
Prepare a separate checkout/worktree for this version and reuse/copy prepared
data only after checking its provenance.

## 1. Prepare the GPU host

Use a CUDA development image with an NVIDIA driver, CUDA toolkit (`nvcc`), CMake,
a C++ compiler, Git, and Python 3.10+ for the standalone setup helper. No uv,
AutoGen, Python ML environment, or educational datasets are needed remotely.

On the GPU host, verify `nvidia-smi` and `nvcc --version`, then build the same pinned
llama.cpp revision recorded in the repository's local configuration:

```bash
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout --detach 90e3841d9f35a5bf3c112b9d71502bce77de5b32
cmake -S . -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j 3
```

Place the exact Qwen3-14B Q4_K_M GGUF used for the reproduction on that machine.
Retain its publisher reference and checksum. No GGUF is committed to this repo.
Copy just the setup script from the workstation:

```bash
scp scripts/remote_inference.py gpu-user@gpu-server:~/remote_inference.py
```

Inside a persistent remote terminal such as tmux, launch (replace absolute paths):

```bash
python3 ~/remote_inference.py \
  --binary /path/to/llama.cpp/build/bin/llama-server \
  --model /path/to/Qwen3-14B-Q4_K_M.gguf \
  --model-name Qwen3-14B \
  --quantization Q4_K_M \
  --commit 90e3841d9f35a5bf3c112b9d71502bce77de5b32 \
  --receipt ~/remote-server.json
```

The script checks the installed binary's `--help` for every option it will use,
including `--jinja`, `--chat-template-kwargs`, and `--log-disable`. Unsupported
versions fail before starting the model. The supplied commit is checked against
the commit prefix in `--version` when reported. A prebuilt binary can also be
used; retain its actual version/commit rather than assuming it matches this build.

Defaults match `configs/remote.toml`: alias `exegen-local`, context 16384, output
limit 4096, temperature 0.7, top-p 0.8, top-k 20, min-p 0, one slot, and three CPU
threads. Set the corresponding script/config options together if changing them.
Automatic context fitting and context shifting are disabled.

Qwen3 non-thinking mode is configured explicitly with:

```text
--jinja --chat-template-kwargs '{"enable_thinking":false}'
```

The API always binds to **127.0.0.1:8080**. There is no public API listener.
Server logging is disabled and stdout/stderr discarded so requests/completions
are not persisted as remote project logs. Ambient `LLAMA_ARG_*` overrides are
removed. The only generated project artifact on the GPU host is setup metadata,
not student histories or experiment results. Inference necessarily handles the
request context in RAM while generating.

The script remains in the foreground; Ctrl-C stops the child. `--receipt-only`
records a setup without launching and is not proof of server readiness.
After launch, use `nvidia-smi` in a second remote terminal to confirm the model
process and its VRAM allocation before starting the overnight experiment.

## 2. Copy provenance and open the tunnel

On the workstation, with the project directory as the working directory:

```bash
mkdir -p runtime
scp gpu-user@gpu-server:~/remote-server.json runtime/remote-server.json
```

Keep the SSH forwarding session running in another persistent **local** terminal:

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18080:127.0.0.1:8080 \
  gpu-user@gpu-server
```

Both forwarding and the remote listener are explicitly loopback-bound. Do not
replace them with `0.0.0.0`, use `ssh -g`, or open port 8080 publicly. SSH handles
transport authentication; the compatible API uses the existing dummy key `local`.

## 3. Configure the workstation

Install the repository environment on the workstation only:

```bash
uv sync --frozen --extra reproduction --extra prepare --extra analysis
```

Use `--config configs/remote.toml` **before** the subcommand. The minimal mode fields
are `[inference] mode = "remote"` and `receipt = "runtime/remote-server.json"`.
The model URL is `http://127.0.0.1:18080/v1`. Remote configuration does not include
or require a local GGUF, server binary, source checkout, or remote PID.

To keep the original commands without `--config`, explicitly apply this profile
to your local `configs/reproduction.toml` (for example, copy the remote template
there). Keep that configuration with the run; no hidden mode auto-detection occurs.
Older configurations without `[inference]` remain local by default.

```bash
uv run --no-sync python experiments/reproduce.py --config configs/remote.toml check --offline
uv run --no-sync python experiments/reproduce.py --config configs/remote.toml check --smoke
```

Offline checks require neither the tunnel nor the receipt. Online remote checks:

- validate the supplied receipt and loopback URL;
- require `/health` to report `status=ok`;
- require the configured alias in `/v1/models`;
- compare actual context metadata and build/template fields from `/props` when available;
- perform a real chat completion and require a nonempty, finished text response.

The smoke test allows 128 output tokens, rejects empty/reasoning-only or truncated
answers, and rejects thinking output when non-thinking mode is configured. Chat
template kwargs also accompany **all** agent requests, including manager, speaker
selection, reviewers, and summary. Prompts themselves are not edited to add `/no_think`.
These parameters are identical for ICL+CoT and ExeGen.

The check distinguishes authentication, connection/tunnel failure, timeout, loading,
alias/context/commit mismatch, unsupported chat API, and invalid content. `/props`
may be absent on some builds; the manifest then explicitly records that context
was checked against the receipt only, not verified through live metadata.

## 4. Run the existing pipeline locally

Data download/preparation and BERT embeddings remain workstation commands. An
ICL+CoT-only run needs just `histories.csv`; it does not need BERT or explanations.
For ExeGen, complete `embed` locally and `explain` through the same tunnel:

```bash
uv run --no-sync python experiments/reproduce.py --config configs/remote.toml download-data
uv run --no-sync python experiments/reproduce.py --config configs/remote.toml download-bert
uv run --no-sync python experiments/reproduce.py --config configs/remote.toml prepare
uv run --no-sync python experiments/reproduce.py --config configs/remote.toml audit-data
uv run --no-sync python experiments/reproduce.py --config configs/remote.toml embed
uv run --no-sync python experiments/reproduce.py --config configs/remote.toml explain
```

CPU BERT embedding defaults to three threads when the remote profile omits local
model runtime settings. Downloads, prepared data, and BERT weights are ignored by
Git. Interrupted dataset downloads can be resumed by repeating `download-data`;
completed files are verified against their recorded hashes.

Existing demonstrations can be reused only if the model hash and documented
generation settings agree. Paths/ports alone no longer invalidate them. Old
thinking-mode or unrecorded-thinking demonstrations must not be silently reused
for the new non-thinking experiment.

Pilot:

```bash
uv run --no-sync python experiments/reproduce.py --config configs/remote.toml run \
  --methods icl_cot_paper --split validation --limit 1 --seeds 42

uv run --no-sync python experiments/reproduce.py --config configs/remote.toml run \
  --methods icl_cot_paper exegen_code --split validation --limit 3 --seeds 42
```

After reviewing content, completion rates, context fit, and costs, run the frozen
test configuration with the same CLI. Keep the workstation, local controller,
tunnel, and remote server running throughout the overnight experiment.

Results, every agent request/response, intermediate outputs, and evaluation are
written **locally** under `outputs/`. Evaluate/export exactly as before. The GPU
host does not need these files or the experiment configuration.

After a dropped tunnel, restore it and rerun the same command: completed jobs are
skipped and failed/interrupted jobs start new attempts. A restart with an unchanged
scientific receipt does not change identity merely because its PID/time changed.
Changes to model hash, binary/version/commit, quantization, runtime/generation
settings or configuration create a different experiment identity. Code-version
changes also change identity, as in the original implementation.

## Scientific provenance and telemetry limits

`remote-server.json` contains actual remotely computed GGUF and binary SHA-256,
exact supplied model name/quantization, server version/commit, launch command,
context, runtime parameters, and generation defaults. Preserve it with results.
Binary SHA-256 may be null for a manually supplied receipt if unavailable; model
SHA-256 and server version/commit are mandatory. No placeholder hash is generated.

**This is operator-supplied provenance, not cryptographic attestation of the live
API.** The workstation does not recompute a remote GGUF hash. Alias, available
metadata, and smoke checks detect common mistakes but cannot prove that a server
is executing the supplied bytes. That limitation is retained in the manifest.
Each attempt retains its current inference provenance as well as the experiment
manifest, including after a server restart.

Request tokens, timings, server timing extensions, local worker CPU/RAM, and local
artifacts remain available. Remote mode never inspects a remote PID with local
psutil and never reports workstation GPU readings as remote GPU load. GPU/server
CPU/RAM measurements are null unless a future explicit remote monitoring facility
is added; they are not zero-cost or zero-load measurements. End-to-end latency
includes SSH/network transport. No GPU performance claim is inferred from tests.

References: [pinned llama-server API/options](https://github.com/ggml-org/llama.cpp/blob/90e3841d9f35a5bf3c112b9d71502bce77de5b32/tools/server/README.md)
and [Qwen3-14B non-thinking mode](https://huggingface.co/Qwen/Qwen3-14B#enable_thinkingfalse).
