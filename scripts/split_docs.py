"""Split a spaCy-formatted file into train, dev, and test partitions"""

from pathlib import Path
from typing import Tuple, Optional

import typer
import spacy
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option


def split_docs(
    # fmt: off
    input_path: Path, 
    output_dir: Path, 
    split_size: Tuple[float, float, float] = Arg((0.8, 0.1, 0.1), help="Split sizes for train/dev/test respectively"),
    shuffle: bool = Opt(False, "--shuffle", "-sf", help="Shuffle the dataset before splitting"),
    seed: Optional[int] = Opt(None, "--seed", "-sd", help="Random seed for shuffling the data")
    # fmt: on
):
    if sum(split_size) != 1.0:
        msg.fail(
            "Split sizes for train, dev, and test should sum up to 1.0 "
            f"({' + '.join(map(str, split_size))} != 1.0)",
            exits=1,
        )

    nlp = spacy.blank("xx")
    db = DocBin().from_disk(input_path)
    docs = list(db.get_docs(nlp.vocab))
    msg.info(f"Found {len(docs)} docs in {input_path}")

    train_size, dev_size, test_size = split_size
    msg.info(f"Splitting docs using sizes: {split_size}")

    train, test = train_test_split(
        docs, test_size=1 - train_size, random_state=seed, shuffle=shuffle
    )
    dev, test = train_test_split(
        test,
        test_size=test_size / (test_size + dev_size),
        random_state=seed,
        shuffle=shuffle,
    )
    datasets = {"train": train, "dev": dev, "test": test}

    msg.text(
        f"Done splitting the train ({len(train)}), dev ({len(dev)}), "
        f" and test ({len(test)}) datasets!"
    )

    for dataset, docs in datasets.items():
        output_path = output_dir / f"{input_path.stem}-{dataset}.spacy"
        db_new = DocBin(docs=docs)
        db_new.to_disk(output_path)
        msg.good(f"Saved {dataset} ({len(docs)}) dataset to {output_path}")


if __name__ == "__main__":
    typer.run(split_docs)
