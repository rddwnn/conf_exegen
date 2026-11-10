# Confidence-processing extension

This stage is independent of the MOOCCubeX reproduction running on the server.
Its commands use separate configuration, data, models, and output directories.
Use a separate checkout/worktree with its own `.venv`; do not switch branches or
resync the environment used by an ongoing reproduction process on the server.
The statistical layer runs on CPU with at most three BLAS threads. No new LLM
or GPU allocation is required to prepare, fit, predict, or evaluate it.

The implementation covers historical confidence processing, logistic error risk,
ablations, probability evaluation, and contexts for the author agents. It does
not report real-data results or claim an improvement to the full ExeGen system.

## Data access and verified facts

The primary candidate is [DBE-KT22](https://arxiv.org/html/2208.12651), published
at [ADA, doi:10.26193/6DZWOH](https://dataverse.ada.edu.au/dataset.xhtml?persistentId=doi:10.26193/6DZWOH).
The paper documents sequence confidence codes: 0 = not provided, 1 = low,
2 = medium, 3 = high. These are ordinal features, not calibrated probabilities.

The official archive returned an access-protection HTML page to automated metadata
requests during development on 2026-11-10. Real raw CSV headers and complete field
coverage have **not** been verified. No mirror or guessed schema is silently used.
The downloader reports this condition rather than saving HTML as data. If the
official API becomes available, it can download the documented tables:

```bash
uv sync --frozen --extra confidence
source .venv/bin/activate
python experiments/confidence.py fetch-dbe
```

If access protection persists, obtain the archive through its official page and
place the CSVs in `data/raw/dbe-kt22/`. All data and derived files remain ignored
by Git. Then inspect the real headers:

```bash
python experiments/confidence.py inspect
```

Set the explicit column mappings in `configs/confidence.toml`, including transaction
and question–concept relationship columns. `import.correctness_values` maps the
actual correctness coding to 0/1. Mapping entries are intentionally unset until
the files are inspected. Confirm that the confidence column uses the documented
coding; do not assume it from a similar column name.

```bash
python experiments/confidence.py import-dbe
```

The importer preserves all linked concepts, records source hashes, rejects unknown
codes and missing concept links, and does not invent absent timestamps or responses.
Zero/empty confidence is missing, including numeric `0.0`; it is not low confidence.

## Normalized input contract

Other verified confidence datasets can use the same normalized CSV interface:

```text
attempt_id,student_id,question_id,timestamp,is_correct,confidence,concept_ids
```

- IDs are nonempty strings. `(student_id, attempt_id)` must be unique.
- Timestamps are ISO 8601. Naive timestamps use one shared UTC interpretation;
  timezone-aware timestamps are converted to UTC before sorting.
- `is_correct` is 0 or 1. Confidence is a documented configured code or missing.
- `concept_ids` is a JSON array of unique strings, CSV-quoted as necessary.
- Repeated questions are allowed with distinct attempt IDs.

Question text, options, answer keys, concept descriptions, and prerequisite links
must still be verified before transferring full ExeGen generation to this dataset.
The statistical layer needs the seven columns above; it never manufactures a
question catalog from IDs.

## Leakage-safe preparation

```bash
python experiments/confidence.py prepare
```

Eligible students have at least ten attempts and a nonempty time-separated prefix.
Sorted IDs are shuffled with seed 42 and partitioned 60/20/20. The split and all
excluded students/reasons are saved in the preparation manifest.

Training pairs use each timestamp group as targets and only strictly earlier
groups as history. Answers at equal timestamps never become evidence for each
other. Validation/test students expose the first 70% of attempts as one fixed
history. If the boundary splits a timestamp group, move that entire group into
the target suffix. A resulting empty prefix excludes the student. No target
outcome, confidence, hint, or response time updates that prefix.

Artifacts in `data/confidence/prepared/` include per-split feature JSONL, safe
history/target contexts, splits, confidence coverage, prefix boundaries, and hashes.
Outcome labels live in the feature files for fitting/scoring; exported LLM targets
use an explicit allowlist that excludes them.

## Features and models

The correctness feature vector contains mean concept observation count, mean
concept success rate, fraction of unseen concepts, student observation count,
and student success rate. An unobserved success rate uses a fixed 0.5 prior with
an explicit unseen indicator.

The confidence vector adds mean reported count, mean ordinal confidence, mean
missing-confidence fraction, high-confidence error rate, and low-confidence correct
rate. The two joint rates divide by reported observations, not all observations.
All-missing confidence has a zero mean sentinel and explicit count/missingness
features. A multi-concept question uses an unweighted mean of its concept summaries;
each prior attempt contributes once to each of its linked concepts.

```bash
python experiments/confidence.py fit
python experiments/confidence.py predict --split validation
python experiments/confidence.py predict --split test
```

Variants:

- `correctness_only`: mean per-concept Beta(1,1) posterior success probability.
- `structured_without_confidence`: standardized L2 logistic regression on correctness features.
- `structured_confidence`: the same model with the additional confidence features.

Scalers and coefficients are fitted only on training students. C is chosen by
unmasked validation Brier over a fixed grid; smallest C wins exact ties. Models
are not refitted on validation data and fitting does not read test feature files.
Single-class training or non-convergence fails explicitly. Models predict error
risk internally and export `p_success = 1 - error_risk`.

Models are plain JSON with coefficients, scaler values, feature order, selected C,
validation scores, training/validation IDs, preparation identity, and library
version. There is no pickle loading or hidden training during prediction.

Robustness masks 25/50/75% of *held-out historical* reported confidence using a
stable attempt-ID hash. Masks are nested and consistent between targets. Training
and hyperparameter selection remain unmasked; there is no retraining per mask.
Correctness-only variants remain unchanged. This tests sensitivity of the frozen
layer, rather than claiming to reproduce naturally occurring missingness.

## Probability evaluation

```bash
python experiments/confidence.py evaluate outputs/confidence/test-predictions.csv
```

Outputs include Brier, AUROC, equal-width calibration/ECE, invalid/missing-output
counts, completion rates, calibration CSV, and paired Brier differences with a
student-cluster bootstrap. A negative difference favors the left method. Cluster
resampling preserves all targets per student and the attempt-weighted estimand.
AUROC is null for single-class labels; intervals require two paired students.

External LLM predictions can be evaluated with the same command using columns
`method,mask_fraction,student_id,attempt_id,p_success`. Labels are joined from
the prepared target file, never trusted from predictions. Unknown target keys and
duplicate predictions are rejected. NaN, out-of-range, and nonnumeric probabilities
are counted as invalid, not clipped. Use `--compare METHOD_A METHOD_B` for other
paired comparisons and `--split validation` for validation reports. Metrics files
are named by split; retain separate output directories for distinct prediction sets.

These layer-only baselines are not ordinary/raw/structured **LLM ExeGen** results.
The latter still require the verified English question catalog and matched agent
runs. Test results must not be used to tune settings.

## Contexts for the author agents

```bash
python experiments/confidence.py export-contexts
python experiments/confidence.py export-contexts --mask-fraction 0.5
```

Exports ordinary, raw, structured, and structured-without-confidence contexts to
`outputs/confidence/contexts/`. Ordinary/raw export does not require fitted models:

```bash
python experiments/confidence.py export-contexts --variants ordinary_exegen raw_confidence
```

Each context has the same correctness history and safe target IDs/concepts. Raw
confidence is a separate attempt-linked list. Structured profiles contain concept
error risk, observation counts, success rates, and supporting historical attempt
references. Confidence features are absent from the no-confidence ablation.
Evidence lists have an explicit cap and truncation flag; counts remain complete.

An author worker job can receive the exported `confidence_context` field. The
worker adds it to the perceiver's initial message as `historical_confidence` or
`confidence_profile`. AutoGen does not broadcast that initiating message, so the
same serialized evidence is explicitly appended after the generator's original
system prompt in extension jobs. The author prompt text is retained as a prefix;
role selection and ordinary jobs remain unchanged when the field is absent.
Preparing the matched English author jobs is a separate
data-adaptation step after the raw question/schema audit, not an automatic reuse
of unrelated MOOCCubeX students.

## Verification

```bash
uv run --frozen --extra reproduction --extra prepare --extra analysis --extra confidence pytest -q
```

Tests use synthetic histories, a tiny local BERT, recorded author-agent responses,
and a local HTTP fixture. They cover target leakage, equal timestamps, missing
confidence, multi-concept aggregation, fixed prefixes, disjoint splits, masks,
JSON inference, actual agent-context propagation, and invalid probability outputs.
No real DBE-KT22 scores or LLM extension runs are claimed by these tests.
