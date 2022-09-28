import typer
import spacy
import os

from typing import Set, Sequence, Optional

from tqdm import tqdm
from wasabi import msg
from spacy.tokens import DocBin, Doc
from _util import info


def _only_unseen(
    docs: Sequence[Doc],
    seen: Set[str],
    *,
    total: Optional[int] = None
) -> DocBin:
    new_docbin = DocBin()
    for doc in tqdm(docs, total=total):
        if len(doc.ents) != 0:
            # mark entities that appear in the training set as missing
            missing = []
            for ent in doc.ents:
                if ent.text in seen:
                    missing.append(ent)
            doc.set_ents([], missing=missing, default="unmodified")
        new_docbin.add(doc)
    return new_docbin


def make_unseen():
    datasets = info("ner")
    for _, dataset in datasets.items():
        trainbin, devbin, testbin = dataset.load()
        msg.good(f"Loaded data set {dataset.source}.")
        nlp = spacy.blank(dataset.lang)
        train_entities = set()
        all_ents = 0
        for doc in tqdm(trainbin.get_docs(nlp.vocab), total=len(trainbin)):
            all_ents += len(doc.ents)
            entities = {span.text for span in doc.ents}
            train_entities.update(entities)
        msg.good(
            f"Collected {len(train_entities)} unique "
            f"entities from a total of {all_ents}."
        )
        new_dev = _only_unseen(
            devbin.get_docs(nlp.vocab), train_entities, total=len(devbin)
        )
        new_test = _only_unseen(
            testbin.get_docs(nlp.vocab), train_entities, total=len(testbin)
        )
        dev_path = os.path.join(
            "unseen", f"{dataset.source}-dev-unseen.spacy"
        )
        test_path = os.path.join(
            "unseen", f"{dataset.source}-test-unseen.spacy"
        )
        new_dev.to_disk(dev_path)
        new_test.to_disk(test_path)


if __name__ == "__main__":
    typer.run(make_unseen)
