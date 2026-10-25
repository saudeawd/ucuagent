#!/usr/bin/env python3
"""Print every query whose relevant document is not ranked first, with its
description, narrative, the relevant document and the top-3 retrieved documents.

Usage:
    ./inspect_failures.py <dataset_dir> <run.txt.gz>
"""
import gzip, json, sys, collections

dataset_dir, run_path = sys.argv[1], sys.argv[2]
top_k = 3

queries = {}
for line in open(f"{dataset_dir}/queries.jsonl", encoding="utf-8"):
    q = json.loads(line)
    queries[str(q["qid"])] = q

qrels = collections.defaultdict(set)
for line in open(f"{dataset_dir}/qrels.txt"):
    qid, _, doc, rel = line.split()
    if int(rel) > 0:
        qrels[qid].add(doc)

run = collections.defaultdict(list)
for line in gzip.open(run_path, "rt"):
    qid, _, doc, rank, *_ = line.split()
    run[qid].append((int(rank), doc))
for qid in run:
    run[qid] = [d for _, d in sorted(run[qid])]

needed = set()
failing = []
for qid, relevant in qrels.items():
    ranked = run.get(qid, [])
    pos = next((i + 1 for i, d in enumerate(ranked) if d in relevant), None)
    if pos is None or pos > 1:
        failing.append((qid, pos))
        needed |= relevant
        needed |= set(ranked[:top_k])

docs = {}
for line in gzip.open(f"{dataset_dir}/documents.jsonl.gz", "rt", encoding="utf-8"):
    d = json.loads(line)
    if d["doc_id"] in needed:
        docs[d["doc_id"]] = d

def short(doc_id):
    d = docs.get(doc_id, {})
    return f"{d.get('title', '?')[:90]}  |  {d.get('url', '')}"

for qid, pos in sorted(failing, key=lambda x: int(x[0])):
    q = queries[qid]
    oq = q["original_query"]
    print("=" * 100)
    print(f"QUERY {qid}  (relevant doc at rank {pos})")
    print(f"  query:       {q['query']}")
    print(f"  description: {oq.get('description')}")
    print(f"  narrative:   {oq.get('narrative')}")
    for doc_id in qrels[qid]:
        print(f"  RELEVANT:    {short(doc_id)}")
        print(f"               {docs.get(doc_id, {}).get('content', '')[:300].replace(chr(10), ' ')}")
    for i, doc_id in enumerate(run[qid][:top_k], 1):
        print(f"  top-{i}:       {short(doc_id)}")
print("=" * 100)
print(f"{len(failing)} of {len(qrels)} queries do not have the relevant doc at rank 1")
