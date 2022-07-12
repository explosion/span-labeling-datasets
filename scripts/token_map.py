import spacy
import srsly
import typer

from collections import Counter
from pathlib import Path
from typing import Dict, Optional
from typing import Callable, List

from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.attrs import intify_attr
from spacy.tokens import DocBin

from multiembed import MultiEmbed


app = typer.Typer()


@spacy.registry.callbacks("set_attr")
def create_callback(
    path: Path,
    component: str,
    attr: str,
    layer: Optional[str],
) -> Callable[[Language], Language]:
    """
    Should be set as a callback of [initialize.before_init].
    You need to set the right ref in your model when you create it.
    This is useful when you have some layer that requires a data
    file from disk. The value will only be loaded during the '
    initialize' step before training.
    After training the attribute value will be serialized into the model,
    and then during deserialization it's loaded
    back in with the model data.
    path (Path): The path to load the attribute from.
    """
    attr_value = srsly.read_jsonl(path)

    def set_attr(nlp: Language) -> Language:
        if not nlp.has_pipe(component):
            raise ValueError(
                "Trying to set attribute for non-existing component"
            )
        pipe: TrainablePipe = nlp.get_pipe(component)
        model = pipe.model.get_ref(layer) if layer is not None else pipe.model
        model.attrs[attr] = attr_value
        return nlp
    return set_attr


@app.command()
def make_mapper(
        path: Path,
        out_path: Path,
        attrs: List[str],
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        unk: int = 0,
        limit: Optional[int] = 0,
        min_freqs: Optional[List[int]] = [],
        max_symbols: Optional[List[int]] = [],
) -> None:
    error_msg = "One of 'model' or 'langauge' has to be provided"
    if model is None and language is None:
        raise ValueError(error_msg)
    elif model is not None and language is not None:
        raise ValueError(error_msg)
    else:
        if model:
            nlp = spacy.load(model)
        else:
            nlp = spacy.blank(language)
    attrs_counts = {}
    docbin = DocBin().from_disk(path)
    for attr in attrs:
        attr_id = intify_attr(attr)
        counts = Counter()
        for doc in docbin.get_docs(nlp.vocab):
            counts.update(doc.count_by(attr_id))
        attrs_counts[attr] = counts
    # Create mappers
    mappers: Dict[str, Dict[int, int]] = {}
    for attr in attrs:
        sorted_counts = attrs_counts[attr].most_common()
        mappers[attr] = {}
        new_id = 0
        for i, (symbol, count) in enumerate(sorted_counts):
            if i == limit and limit != 0:
                break
            if min_freqs:
                if count < min_freqs[i]:
                    break
            elif max_symbols:
                if len(mappers[attr]) > max_symbols[i]:
                    break
            else:
                # Leave the id for the unknown symbol out of the mapper.
                if new_id == unk:
                    new_id += 1
                mappers[attr][symbol] = new_id
                new_id += 1
    srsly.write_msgpack(out_path, mappers)


if __name__ == "__main__":
    texts = [
        "Hello, I am Robert Face.",
        "I like hanging out.",
        "Like everyone else, you know.",
        "Just hanging out",
        "So yeah ...",
    ]
    nlp = spacy.blank('en')
    docbin = DocBin()
    docs = list(nlp.pipe(texts))
    for doc in docs:
        docbin.add(doc)
    docbin.to_disk("test.spacy")
    make_mapper(
        "test.spacy",
        "mappers.msg",
        ["LOWER", "ORTH", "SHAPE"],
        language="en"
    )
    mappers = srsly.read_msgpack("mappers.msg")
    embedder = MultiEmbed(100, 0)
    embedder.attrs["tables"] = mappers
    embedder.initialize(docs)
    Y, _ = embedder(docs, False)
    print(len(Y))
    for y in Y:
        print(y.shape)
