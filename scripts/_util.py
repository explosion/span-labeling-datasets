import os
from pathlib import Path
from typing import Union
from collections import defaultdict
from spacy.util import ensure_path
from dataclasses import dataclass


@dataclass
class SplitInfo:
    """
    Provides convenient wrapper to parse
    the data file names, but its also useful
    to validate that the file names are in
    stardardized format.
    """
    path: Union[Path, str]

    def __post_init__(self):
        self.path = ensure_path(self.path)
        self.name = self.path.name
        tokens = self.name.split("-")
        if not 1 < len(tokens) <= 3:
            raise ValueError(
                f"Incorrect file name {self.name}"
            )
        self.source = tokens[0]
        self.split = tokens[-1].split(".")[0]
        if self.split not in {"train", "dev", "test"}:
            raise ValueError(
                "Splits has to be either 'train', 'dev' or 'test', "
                f"but found {self.split}"
            )
        if len(tokens) == 3:
            self.lang = tokens[1]
        else:
            self.lang = ""


@dataclass
class DatasetInfo:
    train: SplitInfo
    dev: SplitInfo
    test: SplitInfo

    def __getitem__(self, key):
        return self.__dict__[key]


def info(model: str, *, home: str = "corpus"):
    """
    Provides convenient wrapper to avoid keep
    parsing the filenames. It's also useful to
    validate that all splits are there and the
    filenames are in the standardized format.
    """
    assert model in ["ner", "spancat"]
    home = os.path.join(home, model)
    filenames = os.listdir(home)
    splits = []
    for name in filenames:
        path = os.path.join(home, name)
        splits.append(SplitInfo(path))
    datasets = defaultdict(dict)
    for split in splits:
        datasets[split.source][split.split] = split
    out = {}
    for source in datasets:
        if len(datasets[source]) < 3:
            raise ValueError(
                "Each dataset has to have 3 splits"
            )
        datainfo = DatasetInfo(
            datasets[source]["train"],
            datasets[source]["dev"],
            datasets[source]["test"]
        )
        out[source] = datainfo
    return out
