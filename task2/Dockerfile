FROM ghcr.io/uniagent-webis-de/uniagent-pyterrier-baseline:dev-0.0.1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --requirement /app/requirements.txt

COPY german-stopwords.txt event_logging.py case_tools.py retrieval_tools.py rule_checks.py predict.py predict_v2.py /app/
RUN chmod +x /app/predict.py /app/predict_v2.py

ENTRYPOINT ["/app/predict.py"]
