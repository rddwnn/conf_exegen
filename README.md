# ExeGen research

Research on personalized exercise generation, knowledge diagnosis, and student confidence, based on [Knowledge Starts with Practice: Knowledge-Aware Exercise Generative Recommendation with Adaptive Multi-Agent Cooperation](https://papers.neurips.cc/paper_files/paper/2025/hash/85dbd2fb8b355e4231b51e454c08ec1c-Abstract-Conference.html) (NeurIPS 2025). The [original implementation](https://github.com/dsz532/exeGen) is a reference for future work.

## Planned research

- **Reproduction:** adapt the Table 1 comparison of ExeGen and ICL+CoT for single-choice exercises on English-language data, using the same open-weight LLM for both methods. Evaluate Error Hit Rate and Error Recall; exact reproduction of published scores is not required.
- **Extension:** test whether structured self-reported confidence improves knowledge diagnosis and prediction of subsequent answers compared with ordinary ExeGen and raw confidence values. The planned processing layer combines concept statistics with regularized logistic regression. Brier Score is the primary metric; secondary metrics include AUROC, calibration, and exercise-generation HR and Recall.

DBE-KT22 and the Eedi confidence dataset are candidates whose availability and suitability remain to be checked. The planned protocol splits students 60/20/20 and uses each evaluation student's first 70% of attempts as history and last 30% as targets.

## Repository structure

```text
src/           Future research code
experiments/   Future experiment entry points
configs/       Future experiment configurations
prompts/       Future prompt templates, versioned in Git
tests/         Future tests
```

These directories currently contain only `.gitkeep` placeholders. No pipeline or experiment logic is implemented.

## Tooling and future setup

This is a research workspace managed with **uv**, not an installable Python package. `pyproject.toml` declares Python >=3.11, core research dependencies, an optional `inference` extra, and a `dev` dependency group. Dependencies have not been resolved or installed; no lockfile or environment has been created. When preparing the working environment later, resolve dependencies and commit `uv.lock`.

**MLflow** is planned for tracking models, configurations, prompt versions, parameters, metrics, artifacts, and costs. Prompt templates will live in `prompts/` and be versioned through Git. Tracking infrastructure is not configured yet.

Future experiments must fit within 20 GB GPU VRAM, 3 CPU cores, and 16 GB RAM at 15 points/hour. Model selection, quantization, inference backend, and CUDA-specific installation settings remain undecided. Caching, context limits, checkpoints, and cost accounting will be designed later.
