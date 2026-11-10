# Local MLflow tracking

The workstation stores metadata in SQLite and artifacts in `mlruns/artifacts/`.
`configs/remote.toml` enables tracking by default. GPU inference still uses the
SSH-forwarded llama.cpp API; the GPU host needs no MLflow installation.

```toml
[tracking]
mlflow = true
directory = "mlruns"
backend = "sqlite"
```

Paths are resolved against the repository root, independently of the working
directory. Older profiles without `backend` retain the original file store.
Existing file-store runs are not migrated to SQLite automatically.

## Install and start

```bash
uv sync --frozen --extra reproduction --extra prepare --extra analysis --extra tracking
uv run --no-sync python experiments/tracking.py check
uv run --no-sync python experiments/tracking.py serve
```

Open <http://127.0.0.1:5000>. The UI binds to loopback and uses one worker.
Use `serve --port 5001` if necessary. Ctrl-C stops a foreground server.
MLflow 2.22.5 and a compatible SQLAlchemy version are locked in `uv.lock`.

The experiment runner writes directly to the local SQLite store, so the UI need
not be running during inference. The UI and runner use the same absolute backend
and artifact paths. The store and generated artifacts are excluded from Git.
Retain the `tracking` extra when syncing the environment; use `--no-sync` for runs
once the environment has been installed.

## What gets recorded

Each experiment has a parent run with model and generation parameters, selection,
seeds, model/binary hashes, Git revision, configuration and provenance manifests.
Each worker attempt has a child run with method, student and seed, request/token
counts, latency, available CPU/RAM/GPU measurements, and valid-set concept metrics.
Result, status, request trace, resource trace and worker log files are attached.
Missing telemetry and undefined metrics are omitted, rather than replaced by zero.
In remote mode the existing local monitor does not measure remote GPU load.

Attempts appear when they start; metrics and artifacts are logged when the worker
returns. Parent aggregate metrics, denominators and reports are logged after
experiment evaluation. Files in `outputs/` remain the source of truth.
Worker failures mark the child and parent as failed; handled Ctrl-C marks them
killed. A hard kill cannot finalize MLflow status. Resume reuses the parent run
ID saved alongside the manifest, skips completed jobs, and creates a child only
for a newly executed attempt. Run IDs are scoped to the tracking store.

Changing tracking configuration changes the current experiment fingerprint, as
other configuration changes do. Enable tracking before starting the experiment;
this integration does not retroactively import earlier inference runs.

## Record data preparation

After `prepare`, `audit-data` and `embed`, record the actual preparation separately:

```bash
uv run --no-sync python experiments/tracking.py log-preparation
```

The `exegen-data-preparation` experiment contains split/student/problem counts,
source statistics, BERT revision, settings, manifests and the audit report.
Repeating this command for the same prepared manifests reuses its run ID.
It does not run an LLM or label preparation counts as generation quality metrics.
Explanation-generation worker traces remain under `data/prepared/explanation-runs/`.

## Back up

Stop the UI and experiment writer, then retain `mlruns/` together with `outputs/`
and preparation manifests. SQLite metadata and its artifact directory belong
together. The existing `export` command archives experiment files in `outputs/`;
it does not include the MLflow database or MLflow artifact copies.

MLflow's [tracking-server documentation](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)
explains the separation of metadata storage, artifact storage, and the UI/API.
The commands above explicitly select the backend instead of relying on defaults
that differ across MLflow versions.
