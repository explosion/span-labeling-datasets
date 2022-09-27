import os
import csv

import spacy
import typer

from spacy.tokens import Doc, DocBin
from collections import defaultdict
from typing import Sequence

from _util import info


def _per_label_fields():
    return {"count": 0, "length": 0, "doc_freq": 0}


def _per_label(docs: DocBin):
    classes = defaultdict(_per_label_fields)
    for i, doc in enumerate(docs, start=1):
        seen = set()
        for span in doc.spans['sc']:
            label = span.label_
            classes[label]['count'] += 1
            classes[label]["length"] += len(span)
            if label not in seen:
                classes[label]["doc_freq"] += 1
                seen.add(label)
    for key in classes:
        classes[key]["length"] /= classes[key]["count"]
    return classes


def _per_span(docs: Sequence[Doc]):
    spans = []
    for doc in docs:
        for span in doc.spans["sc"]:
            row = [len(span), len(doc), len(doc.spans["sc"])]
            spans.append(row)
    return spans


def analyze(
    dataset: str,
    model: str,
    split: str,
    *,
    data_dir: str = "corpus",
    output_dir: str = "analytics"
):
    datainfo = info("spancat", home=data_dir)
    path = datainfo[dataset][split].path
    nlp = spacy.blank(datainfo[dataset].lang)
    docs = list(
        DocBin().from_disk(path).get_docs(nlp.vocab)
    )
    label_stats = _per_label(docs)
    span_stats = _per_span(docs)
    label_stats_path = os.path.join(
        output_dir, f"{dataset}-{split}-labels.csv"
    )
    span_stats_path = os.path.join(
        output_dir, f"{dataset}-{split}-spans.csv"
    )
    with open(label_stats_path, "w") as csvfile:
        fieldnames = ["label", "count", "length", "doc_freq"]
        writer = csv.DictWriter(
            csvfile, fieldnames=fieldnames
        )
        for label, stats in label_stats.items():
            row = {"label": label}
            row.update(stats)
            writer.writerow(row)
    with open(span_stats_path, "w") as csvfile:
        fieldnames = ["length", "doc_length", "neighbours"]
        writer = csv.DictWriter(
            csvfile, fieldnames=fieldnames
        )
        for record in span_stats:
            row = {
                "length": record[0],
                "doc_length": record[1],
                "neighbours": record[2]
            }
            writer.writerow(row)


if __name__ == "__main__":
    typer.run(analyze)
