# UCU Agent — Task 2 (Solving)

Agent for UniAgent'26 Task 2: for every case directory under `$inputDataset`
(`dienstreiseantrag-*`, `beschaffungsantrag-*`, `auslagenerstattung-*`) it
decides `angenommen` / `abgelehnt` and writes

- `$outputDir/predictions.jsonl` — one line per case:
  `{"antrag": "...", "result": "angenommen|abgelehnt", "begruendung": "..."}`
- `$outputDir/run-trace.jsonl.log.gz` — event trace per the
  [event-logging contract](https://github.com/uniagent-webis-de/uniagent-code/blob/main/cikm26/event-logging-contract/README.md).

It is based on the organizers' `solving-smolagents-with-retrieval` baseline:

1. Read all case PDFs (`pdftotext -layout`), extract amounts, attach the
   case-type policy checklist.
2. Build one BM25 index per corpus found under `retrieval-corpora/` (the
   intranet corpus of the test set is picked up automatically).
3. Up to 3 retrieval rounds in which the LLM names the deciding aspects with
   `(corpus, doc_id)` citations.
4. Final LLM decision as strict JSON.

Changes over the baseline, for running safely on TIRA:

- **Time limits.** Each case gets at most `CASE_BUDGET_SECONDS` (270 s,
  TIRA allows 5 min), shrunk further if the 4 h run budget
  (`RUN_BUDGET_SECONDS`, default 3.75 h) would otherwise be exceeded. Each
  LLM request gets the remaining case time as its timeout; aspect analysis
  keeps 90 s in reserve for the decision.
- **Per-case pipeline.** Cases are solved one after another and each
  prediction is flushed immediately; any failure (endpoint down, timeout,
  invalid JSON, broken PDF) yields a fallback `abgelehnt` with a `decision`
  event, so `predictions.jsonl` is always complete.
- **Environment.** `OPENAI_BASE_URL` is required (`OPENAI_API_BASE` accepted);
  `OPENAI_API_KEY` defaults to `EMPTY`; if `OPENAI_MODEL` is unset the first
  model listed by `/models` is used.
- **Context window.** Case documents are capped (15k chars each, 45k per case)
  to fit Qwen2.5-7B's 32k context.
- **Lenient parsing** of the model's label (`genehmigt`, `rejected`, …) and of
  a wrong/placeholder `antrag` id.

## Versions

All versions live in the same Docker image; the TIRA `--command` picks one.

| Version | Command | What it does |
|---|---|---|
| v1 strict | `/app/predict.py --input $inputDataset --output $outputDir --prompt-variant strict` | organizers' decision prompt (reject when a point cannot be verified) |
| v1 material | `/app/predict.py --input $inputDataset --output $outputDir --prompt-variant material` | reject only for a documented, material violation |
| **v2** | `/app/predict_v2.py --input $inputDataset --output $outputDir` | v1 material + deterministic `rule_checks` (`rule_checks.py`) |

`rule_checks.py` computes facts the small models get wrong and adds them to
the decision evidence: position sums vs. stated net totals, award-note
amounts vs. the offer documents, whether the cheapest offer won, number of
offers vs. the 1,000 € net threshold, framework-contract checkbox, invoices
vs. the chosen offer, official reference on expense receipts, the same order
number on several invoices. Each finding is `verstoss` (material violation
unless a document resolves it), `hinweis` (model judges), or `ok`. The checks
use document conventions only (no case ids or amounts) and report nothing
when their inputs are missing; they are logged as `tool_call` +
`observation` events.

## Files

| File | Purpose |
|---|---|
| `predict.py` | v1 entry point / agent pipeline |
| `predict_v2.py` | v2 entry point (v1 + rule checks) |
| `rule_checks.py` | deterministic checks used by v2 |
| `case_tools.py` | PDF tools, policies per case type, fact checks |
| `retrieval_tools.py` | PyTerrier BM25 tool per corpus |
| `event_logging.py` | contract-compliant event trace |
| `test_baseline.py`, `test_rule_checks.py` | unit tests (real PDFs and corpora, fake LLM) |
| `evaluate_local.py` | accuracy + error list + log validity against `datasets/*-truths` |
| `run_local.sh` | run the image on all three spot-check sets and evaluate |
| `evaluation/` | organizers' official evaluator (copy) |

## Data

Unzip the TIRA downloads into `datasets/` (git-ignored):

```
datasets/business-trip-spot-check-inputs/         datasets/business-trip-spot-check-truths/
datasets/task-2-procurement-spot-check-inputs/    datasets/task-2-procurement-spot-check-truths/
datasets/task-2-expense-reimbursement-spot-check-inputs/  ..._truths/
```

## Run locally with Ollama

```bash
brew install ollama && OLLAMA_CONTEXT_LENGTH=32768 ollama serve &
ollama pull qwen2.5:7b
docker build -t ucuagent-task2 .
OPENAI_BASE_URL=http://host.docker.internal:11434/v1 OPENAI_MODEL=qwen2.5:7b ./run_local.sh
# version 2:
SCRIPT=predict_v2.py OPENAI_BASE_URL=http://host.docker.internal:11434/v1 OPENAI_MODEL=qwen2.5:7b ./run_local.sh
```

Ollama's default context is only 4k tokens, so `OLLAMA_CONTEXT_LENGTH` must be
raised or prompts get truncated silently.

## Tests

```bash
docker run --rm -v "$PWD:/app" -v "$PWD/datasets:/app/datasets:ro" --entrypoint python3 \
  ucuagent-task2 -m unittest discover -s /app -p 'test_*.py'
```

## Submit to TIRA

`tira-cli code-submission` needs `--path` inside a **clean git repository**
(everything committed locally; pushing to GitHub is not required). Dry run
first:

```bash
tira-cli code-submission \
  --path . \
  --task uniagent-2026 \
  --dataset task-2-procurement-spot-check-20260928-training \
  --forward-environment-variable OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
  --command '/app/predict.py --input $inputDataset --output $outputDir' \
  --dry-run
```

Then drop `--dry-run` to upload. Each version is a separate submission: repeat
the command with the `--command` of the version (see *Versions*); the image
is the same. Afterwards, in the TIRA web UI, run the
software on the test datasets (`task-2-business-travel-20260928-test`,
`task-2-procurement-20260928-test`, `task-2-expense-reimbursement-20260928-test`)
with the hosted model of choice (gpt-oss-20b or Qwen2.5-7B-Instruct).

## Submit to TIRA via GitHub Actions

The workflow `../.github/workflows/upload-task2-to-tira.yml` (repository root)
does the same as the command above on a GitHub runner:

1. Push the repository (with `task2/` and `.github/`) to GitHub; `datasets/`
   and `runs/` are git-ignored.
2. Repository *Settings → Secrets and variables → Actions*: add
   `TIRA_CLIENT_TOKEN` (shown on the TIRA submit page). Optionally add
   `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` of an endpoint reachable
   from GitHub; without them the pre-upload test run uses dummy values, every
   model call fails fast, and the agent still writes valid fallback output.
3. *Actions → Upload Task 2 to TIRA → Run workflow*, choose `v2`,
   `v1-strict` or `v1-material`. Run it once per version.
4. The uploaded software appears on the TIRA submit page; start it there on
   the test datasets and enter the hosted model's `OPENAI_*` values.
