"""Produce reviewable working copies; never modify the upstream snapshot."""

import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/exegen"
TARGET = ROOT / "src/author"


def main():
    manifest = json.loads((ROOT / "third_party/manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Upstream snapshot changed: {name}")
    TARGET.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in (
        ("process.py", "process.py"),
        ("experiment/baseline.py", "baseline.py"),
        ("experiment/create_prompt.py", "create_prompt.py"),
        ("experiment/gene_explanation.py", "gene_explanation.py"),
    ):
        original = (SOURCE / source_name).read_text(encoding="utf-8")
        if target_name == "create_prompt.py":
            # Import-level disk I/O is supplied by the worker; keep all function bodies intact.
            tree = ast.parse(original)
            rows = original.splitlines(keepends=True)
            for node in reversed(tree.body):
                if isinstance(node, ast.Assign):
                    del rows[node.lineno - 1 : node.end_lineno]
            changed = "".join(rows)
        elif target_name == "gene_explanation.py":
            tree = ast.parse(original)
            agent = next(
                n
                for n in tree.body
                if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "agent_explanation"
            )
            changed = (
                "from autogen import ConversableAgent\n\n"
                + ast.get_source_segment(original, agent)
                + "\n"
            )
        else:
            changed = original[: original.index("\ntext = {")]
            changed = changed.replace(
                "from experiment.create_prompt import *", "from create_prompt import *"
            )
            changed = changed.replace(
                'pandas.read_csv("subject_data/stuRec_1000_with_tokens.csv")',
                'pandas.DataFrame(job["history"])',
            )
            changed = changed.replace(
                'pandas.read_csv("../subject_data/stuRec_1000_with_tokens.csv")',
                'pandas.DataFrame(job["history"])',
            )
            for prefix in ("subject_data/", "../subject_data/"):
                changed = changed.replace(
                    f'pandas.read_csv("{prefix}examples_with_explanation_with_tokens.csv")',
                    'pandas.read_csv(job["demonstrations_csv"], keep_default_na=False)',
                )
                changed = changed.replace(
                    f'pandas.read_csv("{prefix}concept_relationship_filtered.csv")',
                    'pandas.read_csv(job["relations_csv"], keep_default_na=False)',
                )
            for prefix in ("txtfile/", "../txtfile/"):
                for fmt in ("n_output.txt", "j_output.txt"):
                    changed = changed.replace(
                        f'open("{prefix}{fmt}", "r")', f'open(job["{fmt}"], "r", encoding="utf-8")'
                    )
            # Replace only the configuration assignment, leaving all prompt literals intact.
            tree = ast.parse(changed)
            node = next(
                n
                for n in tree.body
                if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "llm_config"
            )
            rows = changed.splitlines(keepends=True)
            rows[node.lineno - 1 : node.end_lineno] = ['llm_config = job["llm_config"]\n']
            changed = "".join(rows)
            if target_name == "baseline.py":
                # AutoGen 0.3.1 rejects a GroupChat with only one agent. Use that agent directly.
                tree = ast.parse(changed)
                rows = changed.splitlines(keepends=True)
                for node in reversed(tree.body):
                    if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) in (
                        "out_groupchat",
                        "out_groupchat_manager",
                    ):
                        del rows[node.lineno - 1 : node.end_lineno]
                changed = "".join(rows)
        (TARGET / target_name).write_text(changed, encoding="utf-8")
        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                changed.splitlines(keepends=True),
                fromfile="upstream/" + source_name,
                tofile="working/" + target_name,
            )
        )
        (TARGET / (target_name + ".patch")).write_text(diff, encoding="utf-8")
    print("Verified snapshot and generated working copies and diffs")


if __name__ == "__main__":
    main()
