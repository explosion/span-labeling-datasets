import os

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Tuple
from collections import defaultdict

from spacy.tokens import DocBin
from spacy.util import ensure_path


@dataclass
class SplitInfo:
    """
    Provides convenient wrapper to parse
    the data file names, but its also useful
    to validate that the file names are in
    stardardized format.

    It checks that all files have the format
    "source-split.spacy" or "lang-source-split.spacy"
    and stores the full path, "source", "split" and "lang"
    fields. Additionally if one data set comes in multiple
    languages like "es-conll-train.spacy" and "nl-conll-train.spacy"
    it stores "es-conll" or "nl-conll" as .source, but
    "conll" as .dataset for both.
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
        self.split = tokens[-1].split(".")[0]
        if self.split not in {"train", "dev", "test"}:
            raise ValueError(
                "Splits has to be either 'train', 'dev' or 'test', "
                f"but found {self.split}"
            )
        if len(tokens) == 3:
            source = tokens[1]
            self.lang = tokens[0]
            self.source = f"{self.lang}-{source}"
        elif tokens[0] in ["anem", "wnut17", "archaeo", "finer"]:
            source = tokens[0]
            self.source = tokens[0]
            self.lang = "en"
        else:
            source = tokens[0]
            self.source = tokens[0]
            self.lang = "xx"
        self.dataset = source


@dataclass
class DatasetInfo:
    source: str
    train: SplitInfo
    dev: SplitInfo
    test: SplitInfo

    def __post_init__(self):
        langs = [self.train.lang, self.dev.lang, self.test.lang]
        if not len(set(langs)) == 1:
            raise ValueError(
                "All splits of the same dataset should have the "
                f"same langauge, but found {langs}."
            )
        else:
            self.lang = self.train.lang

    def __getitem__(self, key):
        return self.__dict__[key]

    def load(self) -> Tuple[DocBin, DocBin, DocBin]:
        train = DocBin().from_disk(self.train.path)
        dev = DocBin().from_disk(self.dev.path)
        test = DocBin().from_disk(self.test.path)
        return train, dev, test


def info(model: str, *, home: str = "corpus"):
    """
    Provides convenient wrapper to avoid keep
    parsing the filenames. It's also useful to
    validate that all splits are there and the
    filenames are in the standardized format.
    """
    if model not in ["ner", "spancat"]:
        raise ValueError(
            "'model' has to be 'ner' or 'spancat', "
            f"but found {model}"
        )
    home = os.path.join(home, model)
    filenames = os.listdir(home)
    splits = []
    for name in filenames:
        path = os.path.join(home, name)
        split = SplitInfo(path)
        splits.append(split)
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
            source,
            datasets[source]["train"],
            datasets[source]["dev"],
            datasets[source]["test"]
        )
        out[source] = datainfo
    return out