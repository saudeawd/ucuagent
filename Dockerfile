FROM ghcr.io/uniagent-webis-de/uniagent-pyterrier-baseline:dev-0.0.1

WORKDIR /app

ADD german-stopwords.txt test_baseline.py baseline.py /app/

RUN PYTHONPATH=. pytest .
