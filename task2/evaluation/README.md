# cikm26 evaluation

A small, dockerized `click` CLI that evaluates a cikm26 submission by calling
the [`tira`](https://github.com/tira-io/tira) Python package's
`tira.evaluators.evaluate()` directly with a hardcoded evaluator
configuration, and reports the submission's compliance with the
[event-logging contract](../event-logging-contract/README.md) alongside the
task-specific measure(s).

It is compatible with both kinds of cikm26 spot-check datasets, selected via
`--task`:

- `--task solving` datasets, such as
  [`../datasets/business-trip-spot-check/`](../datasets/business-trip-spot-check/README.md)
  (submission format `predictions.jsonl`, evaluated by `accuracy`), and
- `--task retrieval` datasets, such as
  [`../datasets/retrieval-de-spot-check/`](../datasets/retrieval-de-spot-check/README.md)
  (submission format `run.txt`/`run.txt.gz`, evaluated by `nDCG@10`).

Each `--task`'s evaluator configuration (`TASK_CONFIGS` in
`evaluate_submission.py`) is copied verbatim from the corresponding dataset
README's `tira_configs` block, so this script does not need network access
to TIRA to know which format/measure(s) to use -- unlike `Client.evaluate()`,
which looks this up from the dataset's trusted-evaluator configuration on
TIRA. `--dataset` is only used (and only needed) to download a dataset's
published truths from TIRA when `--truths` is omitted.

## What it adds on top of tira's `tira.evaluators.evaluate()`

In addition to whatever measure(s) `tira.evaluators.evaluate()` reports (e.g.
`accuracy` or `nDCG@10`), this script always adds:

- `model`: the LLM model used for the run, taken from the first
  non-placeholder `model` field found in the submission's
  `run-trace.jsonl.log.gz` event trace, or `"-"` if the trace has none (as is
  expected for model-free/deterministic submissions such as
  `business-trip-always-rejected` or the retrieval baselines).
- `valid_log_lines` / `invalid_log_lines`: how many lines of the run-trace
  log are (respectively, are not) valid events per the
  [event-logging contract](../event-logging-contract/README.md) — i.e. they
  decode as JSON, contain all required fields, and have a non-empty `model`
  and a `status` of `"ok"` or `"error"`. A missing run-trace log is reported
  as `0`/`0`, since logging may legitimately be skipped entirely for
  submissions that use no tools and no model calls (contract requirement 1).

## Usage

Build the image once:

```bash
docker build --tag cikm26-evaluation .
```

### Task 2 "Solving" example (business-trip)

```bash
docker run --rm \
  --volume "$PWD/predictions:/predictions:ro" \
  --volume "$PWD/../datasets/business-trip-spot-check:/truths:ro" \
  cikm26-evaluation \
  --predictions /predictions \
  --task solving \
  --truths /truths
```

### Task 1 "Retrieval" example

```bash
docker run --rm \
  --volume "$PWD/predictions:/predictions:ro" \
  --volume "$PWD/../datasets/retrieval-de-spot-check:/truths:ro" \
  cikm26-evaluation \
  --predictions /predictions \
  --task retrieval \
  --truths /truths
```

Both print a single JSON object, e.g.:

```json
{
  "accuracy": 0.83,
  "model": "gpt-oss-20b",
  "valid_log_lines": 214,
  "invalid_log_lines": 0
}
```

### Options

| Option | Required | Description |
| --- | --- | --- |
| `--predictions` | yes | Directory with the submission's output (`predictions.jsonl` or `run.txt`/`run.txt.gz`), and optionally its `run-trace.jsonl.log.gz` event trace. |
| `--task` | yes | Which task's evaluator configuration to use: `solving` (e.g. business-trip-spot-check, evaluated by `accuracy`) or `retrieval` (e.g. any of the retrieval-*-spot-check datasets, evaluated by `nDCG@10`). See `TASK_CONFIGS` in `evaluate_submission.py`. |
| `--truths` | no* | Local truths directory (e.g. the dataset folder under `../datasets/`, which already contains `ground-truth.jsonl`/`decision-trail/` or `qrels.txt`). |
| `--dataset` | no* | The TIRA dataset ID to download published truths from, e.g. `business-trip-spot-check-20260907-training` or `retrieval-de-spot-check-20260816-training`, used only when `--truths` is omitted. Dataset IDs are date-versioned; check the baselines' READMEs under [`../baselines/`](../baselines/) for the current ID. |
| `--run-trace` | no | Override the run-trace log path. Defaults to `<predictions>/run-trace.jsonl.log.gz`. |
| `--output` | no | Directory to additionally write an `evaluation.prototext` to (TIRA's expected evaluator output format). Only needed when this image is invoked as a dataset's `tira_configs.evaluator` (see below); created if it doesn't exist. |

\* Exactly one of `--truths` or `--dataset` must be given.

## Using this image as a dataset's TIRA evaluator

Every cikm26 spot-check dataset's `tira_configs.evaluator` (see the dataset
READMEs under [`../datasets/`](../datasets/)) points at this image,
published as `ghcr.io/uniagent-webis-de/uniagent-cikm-evaluator:0.0.1`
(kept in sync with the version tag in the `Dockerfile`'s header comment),
with a task-specific command such as:

```yaml
evaluator:
  measures: ["accuracy"]
  image: "ghcr.io/uniagent-webis-de/uniagent-cikm-evaluator:0.0.1"
  command: "/evaluate_submission.py --predictions $inputRun --truths $inputDataset --task solving --output $outputDir"
```

`tira`'s command normalization maps `$inputRun`/`$inputDataset`/`$outputDir`
to the submission's predictions, the dataset's truths, and the directory
the evaluator must write its `evaluation.prototext` to, respectively (see
`__normalize_command()` in tira's `local_execution_integration.py`); this is
exactly what `--output` (together with `--predictions`/`--truths`) supports.

## Requirements and network access

With `--truths` given, this script needs no network access to TIRA at all,
since `--task`'s evaluator configuration is hardcoded (see `TASK_CONFIGS` in
`evaluate_submission.py`); with `--dataset` instead, network access to TIRA
is needed to download the dataset's published truths. The underlying
evaluators need no further network access at runtime:

- The business-trip datasets' `accuracy` measure uses HuggingFace's
  `evaluate` package, which would otherwise download its metric script from
  the HuggingFace Hub on first use; the Dockerfile pre-fetches it at build
  time (`evaluate.load("accuracy")`) and sets `OFFLINE=1`, so the image
  never needs to reach the Hub at runtime.
- The retrieval datasets' `nDCG@10` measure uses `trectools` (installed via
  the `tira[ir]` extra), which needs no extra network access beyond loading
  the (`qrels.txt`/`run.txt`) files themselves.

## Tests

The unit tests run automatically as part of the image build (see the
Dockerfile); the build fails if any test fails:

```bash
docker build --tag cikm26-evaluation .
```

The tests mock out `tira.evaluators.evaluate()` itself (network access to
TIRA is not assumed to be available), and instead focus on the log-analysis
logic (`analyze_run_trace`, `_is_valid_event`) and the CLI wiring.
