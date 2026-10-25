# UCU Agent — Task 1 (Retrieval)

Retrieval system for UniAgent'26 Task 1: for every query in `$inputDataset`
it ranks the documents of the collection (University of Kassel website and
intranet, or Hessian laws and court decisions) and writes

- `$outputDir/run.txt.gz` — a TREC run file (`qid Q0 docno rank score tag`,
  up to 1,000 documents per query), evaluated with **nDCG@10**.

It is based on the organizers' `retrieval-baseline-pyterrier` baseline
(`baseline.py`, unchanged):

1. Detect the query language (`de` / `en`).
2. Index the documents with PyTerrier: language-specific tokeniser,
   stop-word list (`german-stopwords.txt` or Terrier's English list) and
   stemmer (German Terrier stemmer or Porter).
3. Rank with the **DPH** weighting model, a parameter-free model from the
   divergence-from-randomness family.

## Our changes (`ucuagent.py`)

Four components, each motivated by an error analysis of the baseline on the
spot-check data. Each is a command-line flag; without flags `ucuagent.py`
behaves exactly like the DPH baseline.

| # | Flag | What it does | Why |
|---|---|---|---|
| 1 | `--query-fields query+description+narrative` | query text = `query` + `description` + `narrative` from `original_query`; missing fields are skipped | the keyword `query` has only 2.7–3.8 words; the other fields contain the vocabulary of the relevant document |
| 2 | `--add-title` | the document title is prepended to its text before indexing | law titles carry the paragraph and act, e.g. `§ 5 HRKG` |
| 3 | `--drop-expired` | documents whose title contains `gültig bis: DD.MM.YYYY` with a date in the past are not indexed | 743 of 2,829 legal documents are expired versions of a norm; expired versions outranked the version in force |
| 4 | `--dedup` | after ranking, only the highest-ranked document per MD5 hash of the normalised text is kept | 22.3 % of German and 15.3 % of English documents are exact duplicates under different URLs |

The system makes no LLM or tool calls, so no `run-trace.jsonl.log.gz` is
written (allowed by the organizers' event-logging contract for such systems).

## Results (spot-check datasets, nDCG@10)

| System | German | English | Law | Mean |
|---|---|---|---|---|
| Baseline (DPH, `query` only) | 0.738 | 0.860 | 0.395 | 0.664 |
| `ucuagent-full` (all four flags) | **0.930** | **0.933** | **0.702** | **0.855** |
| all flags, but `query` only | 0.762 | 0.860 | 0.475 | 0.699 |

The last row matters if the test topics have no `description` / `narrative`:
the system then falls back to the keyword query automatically. Results of
all 24 combinations come from `experiments.py`; the TIRA runs reproduce the
local numbers exactly.

## Files

| File | Purpose |
|---|---|
| `ucuagent.py` | our retrieval system (entry point) |
| `baseline.py` | organizers' baseline, unchanged; `ucuagent.py` imports its language configuration |
| `experiments.py` | runs all 24 combinations of the four components on the three spot-check sets and prints a table; saves a CSV |
| `evaluate.py` | nDCG@10 of one run file plus the rank of the relevant document per query |
| `eda/inspect_failures.py` | error analysis: prints every query whose relevant document is not ranked first, with its description, narrative and the top-3 retrieved documents |
| `eda/eda-de.txt`, `eda/eda-en.txt`, `eda/eda-law.txt` | output of the error analysis of the DPH baseline on the three spot-check sets; the four components above follow from it |
| `test_baseline.py` | organizers' unit tests, run during the Docker build |
| `german-stopwords.txt`, `german-stopwords.NOTICE` | German stop-word list and its licence |
| `requirements.txt` | Python packages |
| `Dockerfile` | image for TIRA (organizers' PyTerrier base image + `baseline.py` + `ucuagent.py`) |

## Data

The spot-check datasets are in the organizers' repository
([`uniagent-code/cikm26/datasets`](https://github.com/uniagent-webis-de/uniagent-code/tree/main/cikm26/datasets)):
`retrieval-de-spot-check`, `retrieval-en-spot-check`,
`retrieval-hessian-law-de-spot-check`. Each contains `queries.jsonl`,
`documents.jsonl.gz` and `qrels.txt`.

```bash
git clone https://github.com/uniagent-webis-de/uniagent-code.git
```

## Run locally

Requires Python ≥ 3.10 and Java ≥ 11.

```bash
cd task1
pip install -r requirements.txt

D=../../uniagent-code/cikm26/datasets   # adjust to where you cloned it
./ucuagent.py --dataset $D/retrieval-hessian-law-de-spot-check --output runs/law \
  --wmodel DPH --query-fields query+description+narrative --add-title --drop-expired --dedup
./evaluate.py $D/retrieval-hessian-law-de-spot-check/qrels.txt runs/law/run.txt.gz
# MEAN nDCG@10 = 0.702

# error analysis: queries where the relevant document is not at rank 1
./eda/inspect_failures.py $D/retrieval-hessian-law-de-spot-check runs/law/run.txt.gz

# all 24 configurations on the three spot-check sets:
python3 experiments.py $D experiments.csv
```

## Tests

```bash
cd task1
PYTHONPATH=. pytest .
```

## Submit to TIRA

`tira-cli code-submission` needs a clean git repository (everything
committed). Run from the repository root; dry run first:

```bash
tira-cli code-submission \
  --path task1 \
  --task uniagent-2026 \
  --dataset retrieval-hessian-law-de-spot-check-20260902-training \
  --command '/app/ucuagent.py --dataset $inputDataset --output $outputDir --wmodel DPH --query-fields query+description+narrative --add-title --drop-expired --dedup' \
  --dry-run
```

Then drop `--dry-run` to upload and start the software in the TIRA web UI
(resources: Large) on the six Task 1 datasets.

Submitted software (team `ucuagent`):

| TIRA software | Command |
|---|---|
| `dph-baseline` | `/app/baseline.py --dataset $inputDataset --wmodel DPH --output $outputDir` |
| `ucuagent-full` | `/app/ucuagent.py --dataset $inputDataset --output $outputDir --wmodel DPH --query-fields query+description+narrative --add-title --drop-expired --dedup` |

On Apple Silicon, set `DOCKER_DEFAULT_PLATFORM=linux/amd64` and
`BUILDX_NO_DEFAULT_ATTESTATIONS=1` before submitting so that TIRA accepts the
image.

## Limitations

- The biggest gain comes from `narrative`; on the synthetic spot-check topics
  it sometimes nearly names the answer, so the gain may be smaller on real
  queries.
- The configuration was chosen on the same 65 spot-check queries it is
  reported on.
- Only exact duplicates are removed; German compounds and the length bias
  towards long court decisions are not addressed.
- `--drop-expired` compares with the date of the run, so a later run may
  remove more documents.
