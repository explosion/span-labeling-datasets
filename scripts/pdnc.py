import ast
import csv
import random
from pathlib import Path
from typing import List, Optional
from xml.sax.xmlreader import AttributesImpl

import typer
import spacy
from spacy.tokens import Doc, DocBin, SpanGroup
from wasabi import msg


Arg = typer.Argument
Opt = typer.Option

ID = "pndc"
TRAIN_SIZE = 0.8
DEV_SIZE = 0.1
TEST_SIZE = 0.1


def read_reference_csv(filepath: Path) -> List[str]:
    novels: List[str] = []
    with filepath.open() as f:
        csv_reader = csv.DictReader(f, delimiter=",")
        for row in csv_reader:
            novels.append(row["folder"])
    return novels


def create_doc_from_novel(
    input_dir: Path,
    name: str,
    single_label: bool = False,
    use_ents: bool = False,
    spans_key: str = "sc",
) -> Doc:
    nlp = spacy.blank("en")

    # Process text
    text_path = input_dir / name / "text.txt"
    msg.text(f"Processing {name} ({text_path})...")
    with text_path.open("r") as f:
        text = f.read()
    doc = nlp(text)

    # Process annotations
    annotations_path = input_dir / name / "quotations.csv"
    spans = []
    skips = 0
    with annotations_path.open() as f:
        csv_reader = csv.DictReader(f, delimiter=",")
        for row in csv_reader:
            label = "Quotation" if single_label else row["qType"]
            # There are cases when qType is empty. I'll just skip over them
            if label:
                for span_indices in ast.literal_eval(row["qSpan"]):
                    start, end = span_indices
                    span = doc.char_span(start, end, label, alignment_mode="expand")
                    spans.append(span)
            else:
                skips += 1

    if skips > 0:
        msg.warn(f"Skipped {skips} annotation(s) because they're empty!")

    # Attach spans to doc as either Doc.spans or Doc.ents
    if use_ents:
        try:
            doc.set_ents(spans)
        except ValueError:
            # FIXME: There are some spans that overlap, but mostly due to annotation
            # errors. I'll find a way to include them (or remove them altogether)
            pass
    else:
        group = SpanGroup(doc, name=spans_key, spans=spans)
        doc.spans[spans_key] = group

    return doc


def main(
    # fmt: off
    input_path: Path = Arg(..., help="Path to the pdnc directory", exists=True), 
    output_dir: Path = Arg(..., help="Directory to save the train/dev/test files"), 
    reference_filepath: Path = Arg(..., help="Path to the novels list reference text file", exists=True), 
    use_ents: bool = Opt(False, "--use-ents", "-e", help="Use Doc.ents, don't transfer to Doc.spans"),
    single_label: bool = Opt(False, "--single", "-s", help="Use single label to turn it into a quotation detection task"),
    spans_key: str = Opt("sc", "--spans-key", help="Spans key to use when storing entities"),
    shuffle: bool = Opt(False, "--shuffle", "-sf", help="Shuffle the dataset before splitting"),
    seed: Optional[int] = Opt(None, "--seed", "-sd", help="Random seed for shuffling the data")
    # fmt: on
):
    novels = read_reference_csv(reference_filepath)
    docs: List[Doc] = []
    for novel in novels:
        doc = create_doc_from_novel(
            input_path,
            novel,
            spans_key=spans_key,
            use_ents=use_ents,
            single_label=single_label,
        )
        docs.append(doc)

    msg.info(f"Processed {len(docs)} docs")

    # Separate training and test
    train_dev_size = int(len(docs) * (TRAIN_SIZE + DEV_SIZE))
    train_dev_docs = docs[:train_dev_size]
    test_docs = docs[train_dev_size:]

    # Get dev set from training
    train_size = int(len(docs) * TRAIN_SIZE)
    train_docs = train_dev_docs[:train_size]
    dev_docs = train_dev_docs[train_size:]

    msg.info(
        f"Split datasets into train ({len(train_docs)}), "
        f"dev ({len(dev_docs)}), and test ({len(test_docs)})."
    )

    datasets = [("train", train_docs), ("dev", dev_docs), ("test", test_docs)]

    for name, _docs in datasets:
        doc_bin = DocBin(docs=_docs)
        output_file = output_dir / f"{ID}-{name}.spacy"
        doc_bin.to_disk(output_file)
        msg.good(f"Saved {name} dataset to {output_file}")


if __name__ == "__main__":
    typer.run(main)
