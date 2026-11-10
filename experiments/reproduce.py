"""Command-line entry point; run from any working directory, without packaging."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    from common import config, resolve
    from downloads import (
        DATA_FILES,
        PAPER_URL,
        build_server,
        download,
        fetch_bert,
        fetch_data,
        fetch_model,
    )
    from evaluate import evaluate_experiment, export_experiment
    from prepare import embed, prepare_data
    from runtime import check, explain, run_experiment, serve

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reproduction.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check", help="Verify inputs and the local inference server")
    checking.add_argument("--offline", action="store_true")
    checking.add_argument("--smoke", action="store_true")
    data = commands.add_parser("download-data", help="Download required MOOCCubeX files explicitly")
    data.add_argument("--only", nargs="+", choices=DATA_FILES)
    commands.add_parser("download-bert", help="Download the pinned Chinese BERT checkpoint")
    model = commands.add_parser("download-model", help="Download an explicitly chosen GGUF")
    model.add_argument("--url", required=True)
    model.add_argument("--sha256", required=True)
    commands.add_parser("download-paper", help="Download the reference paper, not commit it")
    commands.add_parser(
        "build-server", help="Build the pinned CUDA llama-server (requires toolkit)"
    )
    commands.add_parser("serve", help="Run a single-slot local server in the foreground")
    commands.add_parser("prepare", help="Stream raw data into reconstructed upstream-format CSVs")
    audit = commands.add_parser("audit-data", help="Audit prepared data without a model or GPU")
    audit.add_argument(
        "--methods", nargs="+", choices=["exegen_code", "icl_cot_paper", "baseline_code"]
    )
    commands.add_parser("embed", help="Compute original BERT embeddings on CPU")
    commands.add_parser("explain", help="Generate demonstration explanations with the local LLM")
    run = commands.add_parser("run", help="Run or resume an experiment")
    run.add_argument(
        "--methods", nargs="+", choices=["exegen_code", "icl_cot_paper", "baseline_code"]
    )
    run.add_argument("--limit", type=int)
    run.add_argument("--split", choices=["validation", "test"])
    run.add_argument("--seeds", nargs="+", type=int, help="Use one seed for a low-cost pilot")
    evaluate = commands.add_parser(
        "evaluate", help="Recompute quality and performance reports offline"
    )
    evaluate.add_argument("experiment")
    export = commands.add_parser("export", help="Archive a saved experiment")
    export.add_argument("experiment")
    export.add_argument("destination")
    analyze = commands.add_parser("analyze-run", help="Analyze one saved attempt without a model")
    analyze.add_argument("attempt")
    analyze.add_argument("--plots", action="store_true", help="Requires the analysis extra")
    args = parser.parse_args(argv)
    cfg = config(args.config)
    if args.command == "check":
        if args.offline and args.smoke:
            parser.error("--offline and --smoke cannot be combined")
        print(json.dumps(check(cfg, args.offline, args.smoke), indent=2))
    elif args.command == "download-data":
        fetch_data(cfg, args.only)
    elif args.command == "download-bert":
        fetch_bert(cfg)
    elif args.command == "download-model":
        fetch_model(args.url, args.sha256, cfg)
    elif args.command == "download-paper":
        download(PAPER_URL, resolve("data/references/exegen.pdf"))
    elif args.command == "build-server":
        build_server(cfg)
    elif args.command == "serve":
        serve(cfg)
    elif args.command == "prepare":
        prepare_data(cfg)
    elif args.command == "audit-data":
        from audit import audit_data

        return 0 if audit_data(cfg, args.methods)["data_valid"] else 1
    elif args.command == "embed":
        embed(cfg)
    elif args.command == "explain":
        explain(cfg)
    elif args.command == "run":
        if args.limit is not None and args.limit <= 0:
            parser.error("--limit must be positive")
        return run_experiment(
            cfg, resolve(args.config), args.methods, args.limit, args.split, args.seeds
        )
    elif args.command == "evaluate":
        evaluate_experiment(resolve(args.experiment))
    elif args.command == "export":
        export_experiment(resolve(args.experiment), resolve(args.destination))
    elif args.command == "analyze-run":
        from common import write_json
        from evaluate import analyze_run

        attempt = resolve(args.attempt)
        if not attempt.is_dir():
            raise ValueError("Attempt directory does not exist")
        analysis = analyze_run(attempt)
        write_json(attempt / "performance.json", analysis)
        print(json.dumps(analysis, indent=2))
        if args.plots:
            from plots import plot_run

            print(plot_run(attempt))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
