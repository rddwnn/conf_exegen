"""Independent confidence-extension commands; do not modify ongoing reproduction jobs."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    from common import config
    from confidence_context import CONTEXT_VARIANTS, export_contexts
    from confidence_data import fetch_dbe, import_dbe, inspect_dbe
    from confidence_evaluation import evaluate_predictions
    from confidence_features import prepare_confidence
    from confidence_model import fit_models, predict_models

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/confidence.toml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "fetch-dbe", help="Fetch documented public tables, or report archive access failure"
    )
    sub.add_parser("inspect", help="Inspect real raw CSV headers before configuring import")
    sub.add_parser(
        "import-dbe", help="Normalize DBE CSVs using explicitly configured column mappings"
    )
    sub.add_parser("prepare", help="Build disjoint splits and strictly historical features")
    sub.add_parser("fit", help="Fit on training students; choose regularization on validation")
    predict = sub.add_parser(
        "predict", help="Predict without fitting or updating held-out histories"
    )
    predict.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate = sub.add_parser("evaluate", help="Score model or external LLM success probabilities")
    evaluate.add_argument("predictions")
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.add_argument("--compare", nargs=2)
    contexts = sub.add_parser(
        "export-contexts", help="Export safe ordinary/raw/structured agent inputs"
    )
    contexts.add_argument("--variants", nargs="+", choices=CONTEXT_VARIANTS)
    contexts.add_argument("--mask-fraction", type=float, default=0.0)
    args = parser.parse_args(argv)
    cfg = config(args.config)
    result = None
    if args.command == "fetch-dbe":
        fetch_dbe(cfg)
    elif args.command == "inspect":
        result = inspect_dbe(cfg)
    elif args.command == "import-dbe":
        import_dbe(cfg)
    elif args.command == "prepare":
        result = prepare_confidence(cfg)
    elif args.command == "fit":
        result = fit_models(cfg)
    elif args.command == "predict":
        result = str(predict_models(cfg, args.split))
    elif args.command == "evaluate":
        result = evaluate_predictions(cfg, args.predictions, args.split, args.compare)
    elif args.command == "export-contexts":
        result = export_contexts(cfg, args.variants, args.mask_fraction)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
