# docker build -t ghcr.io/uniagent-webis-de/uniagent-cikm-evaluator:0.0.1 .
FROM python:3.13-slim

# click: this script's CLI.
# tira[ir]: provides `tira-cli` plus trectools (for the retrieval datasets'
#   nDCG@10 measure).
# evaluate + scikit-learn: needed by tira's HuggingFaceEvaluator for the
#   business-trip datasets' `accuracy` measure (evaluate.load("accuracy")
#   uses scikit-learn under the hood).
RUN pip install --no-cache-dir click 'tira[ir]' evaluate scikit-learn

# Pre-fetch the HuggingFace `evaluate` "accuracy" metric script at build
# time, so the business-trip datasets' accuracy measure works fully offline
# at runtime (evaluate.load("accuracy") would otherwise hit the HuggingFace
# Hub on first use and fail without network access, per OFFLINE=1 below).
RUN python3 -c "import evaluate; evaluate.load('accuracy')"
ENV OFFLINE=1

COPY evaluate_submission.py test_evaluate_submission.py /
RUN chmod +x /evaluate_submission.py

# Run the unit tests as part of the build; fail the build if they don't pass.
# They mock out tira.evaluators.evaluate() and RestClient, so no network
# access to TIRA is needed here.
RUN cd / && python3 -m unittest discover -s / -p 'test_*.py' \
    && rm /test_evaluate_submission.py

ENTRYPOINT ["/evaluate_submission.py"]
