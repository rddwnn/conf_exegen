# Model-run telemetry

All actual SDK chat-completion calls are recorded in `requests.jsonl`, including
speaker selection and reflection summary. Requests remain non-streaming, as in the
author code. SDK retries are disabled; failures become explicit recorded attempts.

Each record contains the exact request/response, timestamp, full latency,
server-reported input/output/total tokens, finish reasons, and any returned
llama.cpp timing extension. Prompt-based stage classification is analysis only;
it never changes the agent workflow. Generation calls after the first are counted
as regeneration calls.

Effective output tokens/second divides output tokens by full request time,
including prefill, transport, and overhead. It is not decode-only throughput.
TTFT is unavailable without streaming, so it is null. Token counts missing from
the server response remain null rather than being estimated from characters.

## Resource samples

The worker samples approximately once a second:

- Worker and server process-tree RSS and CPU percentage via psutil.
- Host available and total RAM.
- Selected NVIDIA devices' VRAM, utilization, and power via `nvidia-smi`.

The managed server PID is read automatically. `monitor.server_pid` overrides it.
In remote mode this override is ignored: only the local worker and host memory
are sampled. No local nvidia-smi or remote-PID lookup is performed. Remote GPU and
server CPU/RAM values stay null; HTTP timing includes the SSH/network overhead.
See [remote inference](remote-inference.md) for receipt provenance and limitations.
The first process CPU reading is null while a baseline is established. CPU 100%
means one core; a three-core allocation can approach 300% per process tree.
RSS includes resident mapped weights and may double-count shared pages. Worker
RSS excludes the parent orchestration process; host available RAM is also recorded.

GPU statistics are device-wide and can include unrelated processes. Configure
`monitor.gpu_indices` for the allocated devices. Device power is integrated over
observed intervals with the trapezoid rule; this is sampled device energy, not
energy attributable exclusively to the model. Short runs may lack enough samples.
Sampling can miss brief peaks.

Unavailable `nvidia-smi`, unsupported fields, and inaccessible processes are
recorded as missing/error data without aborting the model run.

## Reports

`evaluate` writes per-job JSON/CSV and method summaries. The latest attempt
determines the reported outcome; earlier attempts remain on disk and contribute
to all-attempt call/time/token fields. Counts of available measurements are
reported with means. Preparation and explanation costs are separate.

`analyze-run <attempt> --plots` writes `performance.json` and `performance.png`:
per-stage calls/tokens/time, p50/p95 latency, peak prompt tokens, effective speed,
sampled RSS/VRAM/load/power/energy, and regeneration counts. Figures show request
tokens, latency, GPU memory, process memory, and CPU/GPU load.

Explanation traces live in `data/prepared/explanation-runs/`. Keep these separately
for complete cost accounting; experiment exports do not include all preparation
artifacts. At 15 points/hour, billing depends on the entire allocated machine
lifetime, including setup, preprocessing, idle time, and model loading.
