"""Run minimally patched author agents in an isolated process for one student/seed."""

import argparse
import json
import sys
import time
from pathlib import Path

from common import ROOT, read_json, write_json
from telemetry import ResourceMonitor, trace_completions


def author_namespace(job, filename):
    import create_prompt
    import pandas as pd
    from autogen import ConversableAgent

    create_prompt.concept_relation_filtered = pd.read_csv(
        job["relations_csv"], keep_default_na=False
    )
    # Set through the constructor rather than editing every author prompt/agent declaration.
    original = ConversableAgent.__init__

    def initialize(self, *args, **kwargs):
        kwargs.setdefault("human_input_mode", "NEVER")
        kwargs.setdefault("code_execution_config", False)
        return original(self, *args, **kwargs)

    ConversableAgent.__init__ = initialize
    argv = sys.argv
    sys.argv = [
        filename,
        "--type_of_prompt",
        "json_text",
        "--exercise_type",
        "Single_Choice",
        "--output_type",
        "json",
        "--Number_of_Generations",
        "10",
    ]
    namespace = {
        "job": job,
        "__name__": "author_worker",
        "__file__": str(ROOT / "src/author" / filename),
    }
    try:
        exec(
            compile((ROOT / "src/author" / filename).read_text(encoding="utf-8"), filename, "exec"),
            namespace,
        )
    finally:
        sys.argv = argv
        ConversableAgent.__init__ = original
    return namespace


def execute(job, replay=None):
    import pandas as pd
    from autogen import ConversableAgent

    from evaluate import parse_exercises

    sys.path.insert(0, str(ROOT / "src/author"))

    folder = Path(job["run_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_json(folder / "input.json", job)
    write_json(folder / "status.json", {"status": "running", "started_at": started})
    try:
        with (
            ResourceMonitor(folder / "resources.jsonl", job["monitor"]),
            trace_completions(folder / "requests.jsonl", replay),
        ):
            if job["method"] == "explanation":
                namespace = {"llm_config": job["llm_config"]}
                exec(
                    compile(
                        (ROOT / "src/author/gene_explanation.py").read_text(),
                        "gene_explanation.py",
                        "exec",
                    ),
                    namespace,
                )
                reply = namespace["agent_explanation"].generate_reply(
                    messages=[
                        {
                            "content": json.dumps(job["record"], ensure_ascii=False, indent=4),
                            "role": "user",
                        }
                    ]
                )
                explanation = json.loads(reply)["explanation"]
                if not isinstance(explanation, str) or not explanation.strip():
                    raise ValueError("Invalid explanation")
                write_json(folder / "result.json", {"explanation": explanation})
            elif job["method"] == "icl_cot_paper":
                template = (ROOT / "prompts/icl_cot_paper.txt").read_text(encoding="utf-8")
                history = [
                    {
                        k: row[k]
                        for k in ("content", "option", "right_answer", "concept_id", "is_correct")
                    }
                    for row in job["history"]
                ]
                prompt = template.replace(
                    "{student_history}", json.dumps(history, ensure_ascii=False, indent=4)
                )
                agent = ConversableAgent(
                    name="agent_generator",
                    llm_config=job["llm_config"],
                    system_message="",
                    human_input_mode="NEVER",
                    code_execution_config=False,
                )
                # The complete paper prompt already includes the target history.
                reply = agent.generate_reply(messages=[{"role": "user", "content": prompt}])
                write_json(
                    folder / "result.json",
                    {"raw": reply, "exercises": parse_exercises(reply), "author_parsed": None},
                )
            else:
                namespace = author_namespace(
                    job, "process.py" if job["method"] == "exegen_code" else "baseline.py"
                )
                records = pd.DataFrame(job["history"])
                examples = pd.read_csv(
                    job["demonstrations_csv"], keep_default_na=False
                ).reset_index(drop=True)
                selected = [
                    "content",
                    "option",
                    "right_answer",
                    "knowledge_evidence",
                    "is_correct",
                    "explanation",
                ]
                hard, remaining = namespace["get_examples_by_concept"](examples, records.copy())
                # Original helper returns an untyped empty frame when no concept matches.
                hard_records = [] if hard.empty else hard[selected].to_dict(orient="records")
                if not remaining.empty:
                    hard_records += namespace["get_examples_by_similarity"](examples, remaining)[
                        selected
                    ].to_dict(orient="records")
                tasks = records[selected].copy()
                tasks["explanation"] = ""
                message = {
                    "round": "user2",
                    "examples": hard_records,
                    "tasks": tasks.to_dict(orient="records"),
                }
                if job.get("confidence_context") is not None:
                    from confidence_context import attach_confidence

                    message = attach_confidence(message, job["confidence_context"])
                    # AutoGen broadcasts the perceiver's reply, not its initiating user message.
                    # Explicitly supply the same historical evidence to the generator as well.
                    generator = (
                        namespace["ExerciseGenerator"]
                        if job["method"] == "exegen_code"
                        else namespace["agent_generator"]
                    )
                    generator.update_system_message(
                        generator.system_message
                        + "\n\n"
                        + json.dumps(
                            attach_confidence({}, job["confidence_context"]),
                            ensure_ascii=False,
                            indent=4,
                        )
                    )
                write_json(folder / "retrieved_input.json", message)
                prompt = json.dumps(message, ensure_ascii=False, indent=4)
                if job["method"] == "exegen_code":
                    result = namespace["out_groupchat_manager"].initiate_chat(
                        namespace["KnowledgePerceiver"],
                        message=prompt,
                        summary_method="reflection_with_llm",
                        max_turns=20,
                    )
                    history = result.chat_history
                    write_json(folder / "messages.json", history)
                    # Preserve the author's final-message parser; retain complete GroupChat too.
                    write_json(folder / "groupchat.json", namespace["out_groupchat"].messages)
                    reply = history[-1]["content"]
                    summary = result.summary
                    termination = (
                        "stopChat"
                        if "stopChat" in namespace["out_groupchat"].messages[-1]["content"]
                        else "turn_limit_or_return"
                    )
                else:
                    reply = namespace["agent_generator"].generate_reply(
                        messages=[{"role": "user", "content": prompt}]
                    )
                    summary = None
                    termination = "single_reply"
                parsed = namespace["extract_last_json"](reply)
                write_json(
                    folder / "result.json",
                    {
                        "raw": reply,
                        "author_parsed": parsed,
                        "exercises": parse_exercises(reply),
                        "summary": summary,
                        "termination": termination,
                    },
                )
        write_json(
            folder / "status.json",
            {"status": "complete", "started_at": started, "elapsed_seconds": time.time() - started},
        )
    except BaseException as exc:
        write_json(
            folder / "status.json",
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "started_at": started,
                "elapsed_seconds": time.time() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("--replay")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src/author"))
    execute(read_json(args.job), args.replay)
