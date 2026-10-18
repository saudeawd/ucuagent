#!/usr/bin/env python3
"""Dockerized evaluation for UniAgent'26 cikm26 submissions.

Uses the `tira` Python package's `tira.evaluators.evaluate()` (the
unsandboxed evaluator that `Client.evaluate()` itself delegates to once it
has resolved an evaluator configuration, see `tira_client.py`) to compute the
task-specific measure(s) for a submission's predictions, compatible with
both kinds of spot-check datasets in this repository:

- "solving" datasets such as `../datasets/business-trip-spot-check/`
  (`predictions.jsonl`, evaluated by `accuracy`, see that dataset's README's
  `tira_configs`), and
- "retrieval" datasets such as `../datasets/retrieval-de-spot-check/`
  (`run.txt`/`run.txt.gz`, evaluated by `nDCG@10`, see that dataset's
  README's `tira_configs`).

Rather than asking TIRA over the network which measure(s)/format(s) to use
for a given `--dataset` (which requires the dataset to have a trusted
evaluator configured on TIRA, and requires network access even when it
does), this script hardcodes the evaluator configuration for each task
directly from the corresponding dataset README's `tira_configs` (see
`TASK_CONFIGS` below), and the caller picks the right one with `--task`.
This mirrors exactly what the dataset READMEs document; adjust
`TASK_CONFIGS` if a README's `tira_configs` ever changes.

In addition to the measure(s) `tira.evaluators.evaluate()` reports, this
script always reports the LLM model used for the run (per the required
`model` field in `../event-logging-contract/README.md`, or `-` for
model-free submissions such as `business-trip-always-rejected` or the
retrieval baselines) and how many lines of the submission's
`run-trace.jsonl.log.gz` are contract-valid versus invalid, so a
submission's compliance with the event-logging contract is visible
alongside its task measure(s).
"""
import gzip
import json
from pathlib import Path
from typing import Any, Optional

import click
from tira.evaluators import evaluate as tira_evaluate
from tira.io_utils import to_prototext
from tira.rest_api_client import Client as RestClient

RUN_TRACE_FILENAME = "run-trace.jsonl.log.gz"

# Evaluator configurations for `tira.evaluators.evaluate()`, copied verbatim
# from the `tira_configs` of the corresponding dataset READMEs (merging
# `tira_configs.evaluator` with the run/truth `format`/`config` blocks
# `tira.evaluators.load_evaluator_config()` expects). `format_configuration`
# (not `run_format_configuration`) is required here because
# `HuggingFaceEvaluator.throw_if_conf_invalid()` only reads `re_map` from
# `config["format_configuration"]`.
TASK_CONFIGS: dict[str, dict[str, Any]] = {
    # ../datasets/business-trip-spot-check/README.md `tira_configs`.
    "solving": {
        "measures": ["accuracy"],
        "run_format": "*.jsonl",
        "format_configuration": {
            "id_field": "antrag",
            "value_field": "result",
            "required_fields": ["antrag", "result"],
            "minimum_lines": 5,
            "re_map": {"abgelehnt": 0, "angenommen": 1},
        },
        "truth_format": "*.jsonl",
        "truth_format_configuration": {
            "id_field": "antrag",
            "value_field": "result",
            "required_fields": ["antrag", "result"],
            "minimum_lines": 5,
        },
    },
    # ../datasets/retrieval-{de,en,hessian-law-de}-spot-check/README.md
    # `tira_configs` (identical across all three retrieval spot-check
    # datasets).
    "retrieval": {
        "measures": ["nDCG@10"],
        "run_format": ["run.txt"],
        "truth_format": "qrels.txt",
    },
}

# Every event must have exactly these fields per the event-logging contract
# (../event-logging-contract/README.md, requirement 3); values may be null,
# but the keys themselves must be present.
REQUIRED_EVENT_FIELDS = (
    "case_id",
    "event_id",
    "parent_event_id",
    "timestamp",
    "event_type",
    "model",
    "tool",
    "input",
    "output",
    "status",
    "error",
)


def _is_valid_event(entry: Any) -> bool:
    """Check one decoded JSON line against the event-logging contract.

    Requires an object with all `REQUIRED_EVENT_FIELDS` present, a non-empty
    `model` (required on every event, never null/omitted per the contract),
    and `status` being one of the two values the contract allows.
    """
    if not isinstance(entry, dict):
        return False
    if not all(field in entry for field in REQUIRED_EVENT_FIELDS):
        return False
    if not isinstance(entry.get("model"), str) or not entry["model"].strip():
        return False
    if entry.get("status") not in {"ok", "error"}:
        return False
    return True


def analyze_run_trace(path: Path) -> tuple[str, int, int]:
    """Return `(model, valid_line_count, invalid_line_count)` for a run-trace log.

    `model` is the first non-placeholder (i.e. not `"-"`) model identifier
    found across all valid events, or `"-"` if the log has none (e.g. a
    purely deterministic submission). A missing file is treated as zero
    valid and zero invalid lines rather than an error, since logging can
    legitimately be skipped entirely for submissions with no tools and no
    model calls (contract requirement 1).
    """
    if not path.is_file():
        return "-", 0, 0

    model = "-"
    valid = 0
    invalid = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if not _is_valid_event(entry):
                invalid += 1
                continue
            valid += 1
            if model == "-":
                candidate = entry.get("model")
                if candidate and candidate != "-":
                    model = candidate
    return model, valid, invalid


def run_tira_evaluate(predictions: Path, truths: Path, task: str) -> dict[str, Any]:
    """Compute `task`'s measure(s) for `predictions` against `truths`.

    Calls `tira.evaluators.evaluate()` directly with the hardcoded
    `TASK_CONFIGS[task]` evaluator configuration (copied from the relevant
    dataset README's `tira_configs`), instead of asking TIRA over the
    network which format/measure(s) to use for a `--dataset` id. This needs
    no network access and works for any dataset of the given `task`,
    matching what that task's baseline(s) are evaluated with on TIRA.
    """
    try:
        return tira_evaluate(predictions, truths, TASK_CONFIGS[task])
    except Exception as error:
        raise click.ClickException(f"tira evaluation failed: {error}") from error


def download_truths(dataset: str) -> Path:
    """Download `dataset`'s published truths from TIRA (used only when `--truths` is omitted)."""
    client = RestClient()
    try:
        dataset_handle = client.get_dataset(dataset)
        task_id = dataset_handle.get("task_id") or dataset_handle.get("default_task")
        if not task_id:
            raise ValueError("Task configuration is invalid: no task_id/default_task for dataset " + dataset)
        return Path(client.download_dataset(task_id, dataset_handle["dataset_id"], truth_dataset=True))
    except Exception as error:
        raise click.ClickException(f"could not download truths for dataset '{dataset}' from TIRA: {error}") from error


@click.command()
@click.option(
    "--predictions",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory with the submission's output (predictions.jsonl or run.txt.gz), "
    f"and optionally its {RUN_TRACE_FILENAME} event trace.",
)
@click.option(
    "--task",
    required=True,
    type=click.Choice(sorted(TASK_CONFIGS)),
    help="Which task's evaluator configuration to use: 'solving' (e.g. business-trip-spot-check, "
    "evaluated by accuracy) or 'retrieval' (e.g. retrieval-de-spot-check, evaluated by nDCG@10).",
)
@click.option(
    "--dataset",
    default=None,
    help="The TIRA dataset ID to download published truths from, e.g. "
    "business-trip-spot-check-20260907-training or retrieval-de-spot-check-20260816-training. "
    "Only used (and then required) when --truths is omitted.",
)
@click.option(
    "--truths",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Local truths directory. If omitted, --dataset's published truths are downloaded from TIRA.",
)
@click.option(
    "--run-trace",
    "run_trace_path",
    default=None,
    type=click.Path(path_type=Path),
    help=f"Override the run-trace log path (default: <predictions>/{RUN_TRACE_FILENAME}).",
)
@click.option(
    "--output",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Optional directory to additionally write an evaluation.prototext to (TIRA's expected "
    "evaluator output format, e.g. when this image is wired up as a dataset's tira_configs "
    "evaluator and TIRA runs it with $outputDir). Created if it doesn't exist yet.",
)
def main(
    predictions: Path,
    task: str,
    dataset: Optional[str],
    truths: Optional[Path],
    run_trace_path: Optional[Path],
    output_dir: Optional[Path],
) -> None:
    """Evaluate a cikm26 submission's `task` measure(s), plus event-log stats.

    Works for both "solving" datasets (e.g. business-trip-spot-check,
    evaluated by accuracy) and "retrieval" datasets (e.g.
    retrieval-de-spot-check, evaluated by nDCG@10) -- see
    ../datasets/business-trip-spot-check/README.md and
    ../datasets/retrieval-de-spot-check/README.md.
    """
    if truths is None:
        if not dataset:
            raise click.ClickException("--dataset is required when --truths is omitted.")
        truths = download_truths(dataset)

    measures = run_tira_evaluate(predictions, truths, task)

    trace_path = run_trace_path if run_trace_path is not None else predictions / RUN_TRACE_FILENAME
    model, valid_lines, invalid_lines = analyze_run_trace(trace_path)

    report = dict(measures)
    report["model"] = model
    report["valid_log_lines"] = valid_lines
    report["invalid_log_lines"] = invalid_lines

    click.echo(json.dumps(report, ensure_ascii=False, indent=2))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "evaluation.prototext").write_text(to_prototext([report]))


if __name__ == "__main__":
    main()
