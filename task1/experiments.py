#!/usr/bin/env python3

import collections
import csv
import itertools
import math
import sys
from pathlib import Path

from tira.third_party_integrations import ir_datasets

from baseline import detect_query_language
from ucuagent import QUERY_FIELDS, create_index, retrieve

DATASETS = ["retrieval-de-spot-check", "retrieval-en-spot-check", "retrieval-hessian-law-de-spot-check"]


def load_qrels(path: Path) -> dict:
    qrels = collections.defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            qid, _, docno, relevance = line.split()
            qrels[qid][docno] = int(relevance)
    return qrels


def mean_ndcg_at_10(run, qrels: dict) -> float:
    ranked = {
        str(qid): [str(d) for d in group.sort_values("rank")["docno"]]
        for qid, group in run.groupby("qid")
    }
    scores = []
    for qid, relevant in qrels.items():
        docs = ranked.get(qid, [])
        dcg = sum(relevant.get(d, 0) / math.log2(i + 2) for i, d in enumerate(docs[:10]))
        ideal = sorted(relevant.values(), reverse=True)[:10]
        idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
        scores.append(dcg / idcg if idcg else 0.0)
    return sum(scores) / len(scores)


def main(datasets_dir: Path, output_csv: Path, wmodel: str = "DPH") -> None:
    rows = []
    for name in DATASETS:
        dataset_dir = datasets_dir / name
        dataset = ir_datasets.load(str(dataset_dir))
        language = detect_query_language(dataset)
        qrels = load_qrels(dataset_dir / "qrels.txt")
        for add_title, drop_expired in itertools.product([False, True], [False, True]):
            index, doc_hashes, stats = create_index(dataset, language, add_title, drop_expired)
            if drop_expired and stats["expired"] == 0:
                print(f"{name}: no expired documents, skipping drop_expired=True")
                continue
            for query_fields, dedup in itertools.product(QUERY_FIELDS, [False, True]):
                run = retrieve(dataset, index, language, wmodel, query_fields, doc_hashes, dedup)
                score = mean_ndcg_at_10(run, qrels)
                row = {
                    "dataset": name,
                    "add_title": add_title,
                    "drop_expired": drop_expired,
                    "query_fields": query_fields,
                    "dedup": dedup,
                    "ndcg@10": round(score, 3),
                }
                rows.append(row)
                print(row)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 100)
    print(f"{'de':>7} {'en':>7} {'law':>7} {'mean':>7}   configuration")
    configs = collections.defaultdict(dict)
    for row in rows:
        key = (row["add_title"], row["drop_expired"], row["query_fields"], row["dedup"])
        configs[key][row["dataset"]] = row["ndcg@10"]
    summary = []
    for (add_title, drop_expired, query_fields, dedup), scores in configs.items():
        if drop_expired:
            base = configs.get((add_title, False, query_fields, dedup), {})
            scores = {**base, **scores}
        if len(scores) < len(DATASETS):
            continue
        mean = sum(scores.values()) / len(scores)
        summary.append((mean, scores, f"title={add_title} expired={drop_expired} fields={query_fields} dedup={dedup}"))
    for mean, scores, label in sorted(summary, key=lambda item: item[0], reverse=True):
        print(f"{scores[DATASETS[0]]:7.3f} {scores[DATASETS[1]]:7.3f} {scores[DATASETS[2]]:7.3f} {mean:7.3f}   {label}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else Path("experiments.csv"))
