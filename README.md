# ucuagent — UniAgent at CIKM 2026

Team **ucuagent** (Ukrainian Catholic University) in the
[UniAgent shared task](https://uniagent.webis.de/cikm26/uniagent26-web/agentic-university-administration.html)
on agentic AI in university administration (CIKM 2026 AnalytiCup).
The task uses real administrative processes of the University of Kassel:
business travel, reimbursement of privately advanced costs, and procurement.
Both systems run as Docker images on [TIRA](https://www.tira.io/task-overview/uniagent-2026/)
in an offline sandbox.

| Folder | Subtask | What it does | Metric | Author |
|---|---|---|---|---|
| [`task1/`](task1/README.md) | Task 1 — Retrieval | ranks university web/intranet pages and Hessian laws for a query (PyTerrier + DPH, query fields, titles, expired-law filter, duplicate removal) | nDCG@10 | Anastasia Liubenchuk |
| [`task2/`](task2/README.md) | Task 2 — Solving | agent that reads the case PDFs, retrieves rules and decides accept / reject with an LLM and deterministic rule checks | accuracy | Anastasiia Yablunovska |

`.github/workflows/upload-task2-to-tira.yml` uploads the Task 2 agent to
TIRA via GitHub Actions.

Each folder has its own README with the method, results, how to run it
locally and how to submit it to TIRA.
