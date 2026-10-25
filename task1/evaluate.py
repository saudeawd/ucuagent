#!/usr/bin/env python3
"""Compute nDCG@10 for one run file and print the rank of the relevant document per query.

Usage:
    ./evaluate.py <qrels.txt> <run.txt.gz>
"""
import collections
import gzip
import math
import sys

qrels_path, run_path = sys.argv[1], sys.argv[2]

qrels = collections.defaultdict(dict)
for line in open(qrels_path, encoding="utf-8"):
    if line.strip():
        qid, _, doc, rel = line.split()
        qrels[qid][doc] = int(rel)

run = collections.defaultdict(list)
opener = gzip.open if run_path.endswith(".gz") else open
for line in opener(run_path, "rt"):
    qid, _, doc, rank, *_ = line.split()
    run[qid].append((int(rank), doc))

ndcgs = []
for qid, rel in sorted(qrels.items(), key=lambda item: int(item[0])):
    ranked = [doc for _, doc in sorted(run.get(qid, []))]
    dcg = sum(rel.get(doc, 0) / math.log2(i + 2) for i, doc in enumerate(ranked[:10]))
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(sorted(rel.values(), reverse=True)[:10]))
    ndcg = dcg / idcg if idcg else 0.0
    pos = next((i + 1 for i, doc in enumerate(ranked) if rel.get(doc, 0) > 0), None)
    ndcgs.append(ndcg)
    print(f"query {qid:>3}: nDCG@10 = {ndcg:.3f}   relevant doc at rank {pos}")

print(f"\nMEAN nDCG@10 = {sum(ndcgs) / len(ndcgs):.3f}  (queries: {len(ndcgs)})")
