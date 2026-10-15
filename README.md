# Retrieval Baseline

This is the initial TIRA-compatible baseline scaffold for the UniAgent
retrieval task. It currently:

1. Loads the retrieval dataset via `ir_datasets` directory.
2. Reads the language (either german or english) from the query.
3. Creates a language-specific PyTerrier index.
4. Retrieves with a PyTerrier retrieval model.


## Example Usage

```
./baseline.py --dataset ../../datasets/retrieval-de-spot-check/ --wmodel BM25 --output runs/retrieval-de-spot-check/bm25
```

## Event-logging contract

This baseline is a classic statistical retrieval pipeline (PyTerrier
indexing + weighting model) — it uses no agentic tools and calls no LLM.
Per requirement 1 of
[`../../event-logging-contract/README.md`](../../event-logging-contract/README.md),
event-trace logging (`run-trace.jsonl.log.gz`) is therefore skipped entirely;
there are no tool calls or model calls worth tracing. Resource-consumption
metadata is already tracked separately via `tirex_tracker` into
`index-ir-metadata.yml`/`retrieval-ir-metadata.yml`.

## Submit to TIRA

The following command can be used to submit the solution to TIRA (Please remove the `--dry-run` flag to submit to TIRA):

```bash
tira-cli code-submission \
  --path . \
  --task uniagent-2026 \
  --dataset retrieval-de-spot-check-20260816-training \
  --command '/app/baseline.py --dataset $inputDataset --wmodel BM25 --output $outputDir' \
  --dry-run
```


## Tests

```bash
PYTHONPATH=. pytest .
```

