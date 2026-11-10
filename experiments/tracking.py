"""Manage the workstation's local MLflow store and UI without a GPU server."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from common import config
    from tracking import client_and_experiment, local_store, log_preparation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/remote.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    commands.add_parser("log-preparation")
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    cfg = config(args.config)
    if args.command == "check":
        client, experiment_id, uri = client_and_experiment(cfg, cfg["experiment"]["name"])
        print(
            json.dumps(
                {
                    "version": importlib.metadata.version("mlflow"),
                    "tracking_uri": uri,
                    "experiment_id": experiment_id,
                    "artifact_location": client.get_experiment(experiment_id).artifact_location,
                },
                indent=2,
            )
        )
    elif args.command == "log-preparation":
        print(json.dumps(log_preparation(cfg), indent=2))
    else:
        if not 1 <= args.port <= 65535:
            parser.error("port must be between 1 and 65535")
        uri, artifacts = local_store(cfg)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlflow",
                "server",
                "--backend-store-uri",
                uri,
                "--default-artifact-root",
                artifacts.as_uri(),
                "--no-serve-artifacts",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--workers",
                "1",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
