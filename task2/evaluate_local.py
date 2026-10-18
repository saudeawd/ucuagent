#!/usr/bin/env python3
"""Score local runs against the spot-check ground truth.

Usage: ./evaluate_local.py RUN_DIR [RUN_DIR ...]

Each RUN_DIR holds predictions.jsonl and run-trace.jsonl.log.gz. Ground truth
is read from every datasets/*-truths/ground-truth.jsonl. Prints accuracy per
case type, the misclassified cases with the gold reasoning (for error
analysis), and the number of event-log lines that violate the logging
contract.
"""
import collections
import gzip
import json
import sys
from pathlib import Path

REQUIRED_FIELDS = (
    "case_id", "event_id", "parent_event_id", "timestamp", "event_type",
    "model", "tool", "input", "output", "status", "error",
)


def load_truth(datasets_root: Path) -> dict[str, dict]:
    truth = {}
    for path in datasets_root.glob("*-truths/ground-truth.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                truth[entry["antrag"]] = entry
    return truth


def check_log(path: Path) -> tuple[int, int, set[str]]:
    valid, invalid, decided = 0, 0, set()
    if not path.exists():
        return 0, 0, decided
    try:
        with gzip.open(path, "rt", encoding="utf-8") as log:
            lines = list(log)
    except EOFError:
        # A run killed mid-way leaves a truncated gzip; report what survived.
        lines = []
        print(f"warning: {path} is truncated (run was interrupted)")
    for line in lines:
        try:
            event = json.loads(line)
            ok = (
                all(field in event for field in REQUIRED_FIELDS)
                and event["model"]
                and event["status"] in {"ok", "error"}
            )
        except json.JSONDecodeError:
            ok = False
        valid += bool(ok)
        invalid += not ok
        if ok and event["event_type"] == "decision":
            decided.add(event["case_id"])
    return valid, invalid, decided


def main(run_dirs: list[Path]) -> None:
    truth = load_truth(Path(__file__).parent / "datasets")
    per_type = collections.defaultdict(lambda: [0, 0])
    errors = []
    for run_dir in run_dirs:
        predictions = [
            json.loads(line)
            for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for prediction in predictions:
            gold = truth[prediction["antrag"]]
            case_type = prediction["antrag"].rsplit("-", 1)[0]
            correct = prediction["result"] == gold["result"]
            per_type[case_type][0] += correct
            per_type[case_type][1] += 1
            if not correct:
                errors.append((prediction, gold))
        valid, invalid, decided = check_log(run_dir / "run-trace.jsonl.log.gz")
        missing = {p["antrag"] for p in predictions} - decided
        print(f"{run_dir}: {valid} valid / {invalid} invalid log lines; "
              f"cases without decision event: {sorted(missing) or 'none'}")

    print()
    total_correct = sum(c for c, _ in per_type.values())
    total = sum(n for _, n in per_type.values())
    for case_type, (correct, count) in sorted(per_type.items()):
        print(f"{case_type:22s} {correct}/{count} = {correct / count:.2f}")
    print(f"{'overall':22s} {total_correct}/{total} = {total_correct / max(total, 1):.2f}")

    for prediction, gold in errors:
        print(f"\n✗ {prediction['antrag']}: predicted {prediction['result']}, gold {gold['result']}")
        print(f"  agent: {prediction.get('begruendung', '')[:400]}")
        print(f"  gold:  {gold.get('begruendung', '')[:400]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main([Path(arg) for arg in sys.argv[1:]])
