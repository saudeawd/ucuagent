#!/usr/bin/env python3

import datetime
import hashlib
import re
import tempfile
from pathlib import Path

import click
import pandas as pd
import pyterrier as pt
from tqdm import tqdm
from tira.third_party_integrations import ir_datasets
from tirex_tracker import tracking

from baseline import RETRIEVAL_CONFIGURATION, detect_query_language

QUERY_FIELDS = {
    "query": ("query",),
    "query+description": ("query", "description"),
    "query+description+narrative": ("query", "description", "narrative"),
}

EXPIRED_PATTERN = re.compile(r"gültig bis:\s*(\d{1,2})\.(\d{1,2})\.(\d{4})")


def document_field(document, name: str) -> str | None:
    value = getattr(document, name, None)
    return value if isinstance(value, str) else None


def is_expired(title: str | None, today: datetime.date) -> bool:
    if not title:
        return False
    match = EXPIRED_PATTERN.search(title)
    if not match:
        return False
    day, month, year = (int(part) for part in match.groups())
    try:
        valid_until = datetime.date(year, month, day)
    except ValueError:
        return False
    return valid_until < today


def content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def query_text(query, fields: tuple[str, ...]) -> str:
    original_query = getattr(query, "original_query", None)
    if not isinstance(original_query, dict):
        original_query = {}
    parts = []
    for field in fields:
        value = query.default_text() if field == "query" else original_query.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts)


def iter_documents(dataset, add_title: bool, drop_expired: bool, doc_hashes: dict, stats: dict):
    today = datetime.date.today()
    for document in tqdm(dataset.docs_iter(), "Index documents"):
        title = document_field(document, "title")
        if drop_expired and is_expired(title, today):
            stats["expired"] += 1
            continue
        text = document.default_text()
        doc_hashes[str(document.doc_id)] = content_hash(text)
        if add_title and title:
            text = f"{title}\n{text}"
        stats["indexed"] += 1
        yield {"docno": str(document.doc_id), "text": text}


def create_index(dataset, language: str, add_title: bool = False, drop_expired: bool = False, output_dir: Path | None = None):
    if language not in RETRIEVAL_CONFIGURATION:
        raise ValueError(f"Unsupported language: {language}")

    index_directory = Path(tempfile.mkdtemp(prefix="ucuagent-index_"))
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="ucuagent-index-metadata_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    indexer = pt.IterDictIndexer(
        str(index_directory.resolve()),
        overwrite=True,
        meta={"docno": 100},
        verbose=True,
        **RETRIEVAL_CONFIGURATION[language],
    )
    doc_hashes: dict[str, str] = {}
    stats = {"indexed": 0, "expired": 0}
    with tracking(export_file_path=output_dir / "index-ir-metadata.yml"):
        index_ref = indexer.index(iter_documents(dataset, add_title, drop_expired, doc_hashes, stats))
    return pt.IndexFactory.of(index_ref), doc_hashes, stats


def deduplicate(run: pd.DataFrame, doc_hashes: dict) -> pd.DataFrame:
    run = run.sort_values(["qid", "rank"], kind="mergesort").copy()
    run["content_hash"] = [doc_hashes.get(str(docno), str(docno)) for docno in run["docno"]]
    run = run.drop_duplicates(subset=["qid", "content_hash"], keep="first").drop(columns="content_hash")
    run["rank"] = run.groupby("qid").cumcount()
    return run.reset_index(drop=True)


def retrieve(
    dataset,
    index,
    language: str,
    wmodel: str = "DPH",
    query_fields: str = "query",
    doc_hashes: dict | None = None,
    dedup: bool = False,
) -> pd.DataFrame:
    if language not in RETRIEVAL_CONFIGURATION:
        raise ValueError(f"Unsupported language: {language}")

    tokeniser = pt.TerrierTokeniser.java_tokeniser(RETRIEVAL_CONFIGURATION[language]["tokeniser"])
    topics = pd.DataFrame(
        [
            {
                "qid": str(query.query_id),
                "query": " ".join(tokeniser.getTokens(query_text(query, QUERY_FIELDS[query_fields]))),
            }
            for query in dataset.queries_iter()
        ],
        columns=["qid", "query"],
    )
    if topics.empty:
        raise ValueError("Cannot retrieve because the dataset has no queries.")

    run = pt.terrier.Retriever(index, wmodel=wmodel, verbose=False).transform(topics)
    if dedup and doc_hashes:
        run = deduplicate(run, doc_hashes)
    return run


@click.command()
@click.option("--dataset", required=True, help="The TIRA dataset ID or local dataset directory.")
@click.option("--output", required=True, type=click.Path(path_type=Path, file_okay=False), help="The output directory.")
@click.option("--wmodel", default="DPH", show_default=True, help="The PyTerrier weighting model.")
@click.option("--query-fields", default="query", show_default=True, type=click.Choice(list(QUERY_FIELDS)))
@click.option("--add-title/--no-add-title", default=False, show_default=True)
@click.option("--drop-expired/--no-drop-expired", default=False, show_default=True)
@click.option("--dedup/--no-dedup", default=False, show_default=True)
def main(dataset: str, output: Path, wmodel: str, query_fields: str, add_title: bool, drop_expired: bool, dedup: bool) -> None:
    ir_dataset = ir_datasets.load(dataset)
    try:
        language = detect_query_language(ir_dataset)
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    output.mkdir(parents=True, exist_ok=True)
    (output / "language.txt").write_text(f"{language}\n", encoding="utf-8")
    click.echo(f"Detected query language: {language}")

    index, doc_hashes, stats = create_index(ir_dataset, language, add_title, drop_expired, output)
    click.echo(f"Indexed documents: {stats['indexed']}, dropped expired: {stats['expired']}")

    with tracking(export_file_path=output / "retrieval-ir-metadata.yml"):
        run = retrieve(ir_dataset, index, language, wmodel, query_fields, doc_hashes, dedup)
    pt.io.write_results(run, output / "run.txt.gz")


if __name__ == "__main__":
    main()
