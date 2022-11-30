import os
import csv
import tqdm

import spacy
import typer
import pandas as pd

from spacy.tokens import Doc, DocBin
from typing import Sequence, List, Union, Dict
from _util import SplitInfo

Number = Union[int, float]


def per_ent_stats(docs: Sequence[Doc]) -> List[Dict]:
    spans = []
    for i, doc in tqdm.tqdm(enumerate(docs), total=len(docs)):
        # Special "null span" row.
        if not doc.ents:
            row = {
                "doc_id": i,
                "text": None,
                "label": None,
                "length": None,
                "doc_length": len(doc),
                "num_ents": 0
            }
            spans.append(row)
        for span in doc.ents:
            row = {
                "doc_id": i,
                "text": span.text,
                "label": span.label_,
                "length": len(span),
                "doc_length": len(doc),
                "num_ents": len(doc.ents)
            }
            spans.append(row)
    return spans


def datastats(data: Sequence[Dict]):
    df = pd.DataFrame.from_dict(data)
    doc_group = df.groupby(["doc_id"])
    print(f"Num docs {len(doc_group)}")
    print(f"Number of classes {df['label'].nunique()}")
    print(f"Average doc-length: {doc_group['doc_length'].mean().mean()}")
    print(f"Average number of entities: {doc_group['num_ents'].mean().mean()}")
    print(f"Average document length: {doc_group['doc_length'].mean().mean()}")
    print(f"Average entity length: {df['length'].mean()}")
    print(f"Total number of entities: {len(df[df['label'].notnull()])}")


def analyze(
    docbin_path: str,
    model: str,
    *,
    data_dir: str = "corpus",
    output_dir: str = "analyses"
):
    """
    Write two .csv files one with label statistics
    and another with properties of each entity in
    the data set.
    """
    nlp = spacy.load(model)
    vocab = nlp.vocab
    docs = list(
        DocBin().from_disk(docbin_path).get_docs(vocab)
    )
    splitinfo = SplitInfo(docbin_path)
    span_stats = per_ent_stats(docs)
    datastats(span_stats)
    vocabulary = set()
    norms = set()
    prefixes = set()
    suffixes = set()
    shapes = set()
    for doc in docs:
        for token in doc:
            vocabulary.add(token.text)
            norms.add(token.norm_)
            prefixes.add(token.prefix_)
            suffixes.add(token.suffix_)
            shapes.add(token.shape_)
    vec_vocabulary = {nlp.vocab.strings[k] for k in nlp.vocab.vectors.keys()}
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Unknown words: {len(vocabulary - vec_vocabulary)}")
    print(f"Number of norms: {len(norms)}")
    print(f"Number of prefixes: {len(prefixes)}")
    print(f"Number of suffixes: {len(suffixes)}")
    print(f"Number of shapes: {len(shapes)}")
    prefix = (f"{splitinfo.dataset}-{splitinfo.split}"
              f"-{splitinfo.seen}")
    span_stats_path = os.path.join(output_dir, f"{prefix}.csv")
    vocabulary_path = os.path.join(output_dir, f"{prefix}.vocab")
    with open(span_stats_path, "w", encoding="utf-8") as csvfile:
        fieldnames = [
            "doc_id", "text", "label", "length", "doc_length", "num_ents"
        ]
        writer = csv.DictWriter(
            csvfile, fieldnames=fieldnames, delimiter="\t"
        )
        writer.writeheader()
        for record in span_stats:
            writer.writerow(record)
    with open(vocabulary_path, "w", encoding="utf-8") as vocabfile:
        vocab_str = "\n".join(vocabulary)
        vocabfile.write(vocab_str)


if __name__ == "__main__":
    typer.run(analyze)
