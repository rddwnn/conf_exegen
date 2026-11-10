"""Optional standalone performance figures; no model or GPU is needed to render them."""

from pathlib import Path

from evaluate import load_lines


def plot_run(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    requests = load_lines(folder / "requests.jsonl")
    samples = load_lines(folder / "resources.jsonl")
    valid_requests = [r for r in requests if r.get("status") == "ok"]
    valid_samples = [s for s in samples if "timestamp" in s]
    fig, axes = plt.subplots(5, 1, figsize=(11, 15), constrained_layout=True)
    if valid_requests:
        ids = [r["call"] for r in valid_requests]
        axes[0].plot(ids, [r.get("prompt_tokens") for r in valid_requests], "o-", label="Input")
        axes[0].plot(
            ids, [r.get("completion_tokens") for r in valid_requests], "o-", label="Output"
        )
        axes[0].set(xlabel="Model call", ylabel="Tokens", title="Per-request token usage")
        axes[0].legend()
        axes[1].plot(ids, [r["elapsed_seconds"] for r in valid_requests], "o-")
        axes[1].set(xlabel="Model call", ylabel="Seconds", title="Full request latency (not TTFT)")
    else:
        axes[0].text(0.5, 0.5, "No completed requests", ha="center", transform=axes[0].transAxes)
    if valid_samples:
        origin = valid_samples[0]["timestamp"]
        timestamps = [s["timestamp"] - origin for s in valid_samples]
        uuids = sorted({g["uuid"] for s in valid_samples for g in s.get("gpus", [])})
        for uuid in uuids:
            values = [
                (s["timestamp"] - origin, g)
                for s in valid_samples
                for g in s.get("gpus", [])
                if g["uuid"] == uuid
            ]
            axes[2].plot(
                [t for t, _ in values],
                [g.get("memory_used_mib") for _, g in values],
                label=uuid[:16],
            )
            axes[4].plot(
                [t for t, _ in values],
                [g.get("utilization_percent") for _, g in values],
                label="GPU utilization (device)",
            )
        for scope in ("worker", "server"):
            rss = [(s.get(scope) or {}).get("rss_bytes") for s in valid_samples]
            axes[3].plot(
                timestamps, [v / 1024**2 if v is not None else None for v in rss], label=scope
            )
        axes[4].plot(
            timestamps,
            [(s.get("server") or {}).get("cpu_percent") for s in valid_samples],
            label="Server CPU (100% = one core)",
        )
        axes[2].set(ylabel="MiB", title="GPU memory, device-wide")
        axes[3].set(ylabel="MiB", title="Sampled process-tree RSS")
        axes[4].set(ylabel="Percent", title="GPU and server CPU load")
        for axis in axes[2:]:
            axis.set_xlabel("Seconds since first resource sample")
            if axis.get_legend_handles_labels()[0]:
                axis.legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("Local model run: " + folder.name)
    output = folder / "performance.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
