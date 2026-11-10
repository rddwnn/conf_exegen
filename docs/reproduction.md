# Provenance and reproduction protocol

The published snapshot is `dsz532/exeGen@c5a45afa9f1541a33e7034271b75b64fa8291526`.
Every blob is checked against `third_party/manifest.json`. Upstream has no LICENSE
file at that revision; no new license is asserted for copied material.

`scripts/fetch_upstream.py` recreates the snapshot. `scripts/patch_upstream.py`
verifies it and generates the working copies and unified diffs in `src/author/`.
Never format or edit the snapshot or generated files manually.

## Components

| Component | Treatment |
| --- | --- |
| ExeGen and baseline agents | Copied declarations, unchanged prompt literals |
| Retrieval | Original functions; import-time CSV reads moved to worker setup |
| Explanation agent | Copied declaration and system prompt |
| Speaker selection | Original selector and role descriptions |
| Limits and stop marker | Original `max_round=20`, `max_turns=20`, `stopChat` |
| Summary | Original `reflection_with_llm`, including its token/time cost |
| JSON extraction | Original result retained; separate deterministic evaluation parser |
| ICL+CoT | Appendix B.2 extracted from PDF pages 24–25 with layout normalization |
| Output format files | Reconstructed; upstream `txtfile/` is unpublished |
| Prepared CSVs | Reconstructed; upstream `subject_data/` is unpublished |
| BERT | Original concatenation and sum-pooling; separately downloaded pinned weights |
| Sampling, telemetry, resumability | New explicit outer orchestration |

The working scripts retain their author-defined prefixes. Their hard-coded final
row-slice invocation is replaced by the worker's per-student invocation. Prompts
are never reformatted by the patch generator. Agents use noninteractive input
and disable code execution, which is unused by this experiment.

The published baseline creates a one-agent GroupChat. Its working version invokes
that same generator directly once; this control-flow change is recorded separately.
`baseline_code` is not asserted to be the ICL+CoT used for Table 1.

`icl_cot_paper` sends the complete Appendix B.2 template as one user message with
an empty system prompt and substitutes only the history placeholder. The paper's
example includes a multiple-answer historical question; it is preserved. No extra
retrieval, JSON mode, hidden instructions, or corrective generation is added.
`scripts/extract_paper_prompt.py` reproduces extraction using Poppler; PDF and
template hashes are in `prompts/provenance.json`.

## Released-code gaps

The public scripts reference absent inputs, use fixed slices and inconsistent
relative paths, and do not establish the full process behind Table 1.

Active upstream evidence construction concatenates the concept, the concept
again, and a course name. Prerequisite-chain augmentation is commented out. This
project preserves the active behavior and leaves `knowledge_chain` empty. It does
not claim to reproduce all graph enhancements described in the paper.

The retrieval bank index is reset because the original soft helper mixes label
and positional indexing. Empty hard-match frames are handled without changing
nonempty selection. First-match ordering, similarity, and ties are preserved.

The author LLM score script omits exercises from its actual evaluator message.
It is retained as evidence and not used as a valid quality metric. The current
primary evaluation uses checked statistical metrics and exports human-review
material; a repaired LLM judge is not part of the primary comparison.

## Reconstructed data protocol

- Filter CS courses and CS concept IDs; preserve original Chinese content and IDs.
- Require a single-choice question, content, at least two options, one keyed valid
  answer, and a concept link. Normalize numeric IDs and escaped answer encoding.
- Keep the first source-order CS concept per problem, corresponding to the author
  deduplication fragment. Do not infer missing student-selected answers.
- Retain aggregate repeated-attempt records. Do not invent individual attempts.
- Exclude invalid timestamps/correctness with counts; use disk-backed SQLite.
- Select at most 1,000 eligible students by seeded hash, then shuffle/split
  60/20/20 with seed 42. This is our protocol, not a recovered author split.
- Use the first ten eligible timestamp-ordered records per student. Ties are
  ordered by log ID then source order. No future-answer 70/30 test occurs here.
- Build a bounded, training-only demonstration bank (default 256 records), keeping
  first observed positive and negative examples per concept. Test students are excluded.

The sample and bank bounds are explicit resource adaptations. Preparation records
the input provenance, rejected-record counts, and splits. Embedding fails on inputs
over the BERT positional limit instead of applying undocumented truncation.

## Metrics

Let `G` be unique generated concept IDs, and `E` unique concepts of incorrect
answers in the visible history. Report `HR=|G∩E|/|G|` and `Recall=|G∩E|/|E|`.
Original ECR instead counts repeated generated concept IDs separately.
Unknown IDs stay in the denominator and count as misses; no fuzzy mapping is used.

A valid set has exactly ten structurally valid single-choice exercises from a
completed workflow. Schema validity is not proof of factual correctness or novelty.
Invalid outputs have null quality metrics and separate counts. Recall is undefined
without historical errors. Available-value denominators accompany method means.

The paired comparison uses common valid student/seed pairs. Seed differences are
averaged per student before 2,000 student bootstrap replicates with seed 42. A
single student does not receive a confidence interval. Available-case method means
are additionally reported.

`stopChat` termination is distinguished from turn-limit return. The manager is
not overridden to force expert participation or outcomes. Actual expert calls are
visible in request traces. Historical coverage does not establish learning gains.
