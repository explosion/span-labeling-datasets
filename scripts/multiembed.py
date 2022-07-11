import spacy
import thinc

from functools import partial

from spacy.tokens import Doc
from thinc.api import Model, uniform_init, chain, with_array
from with_column import remap_ids
from thinc.api import list2ragged, ragged2list, concatenate, noop
from thinc.types import Floats2d
from typing import Optional, Callable, List, Dict


Remap = remap_ids
Embed = thinc.registry.layers.get("Embed.v1")
Maxout = thinc.registry.layers.get("Maxout.v1")
Dropout = thinc.registry.layers("Dropout.v1")
Vectors = thinc.registry.layers("spacy.StaticVectors.v2")
Extract = thinc.registry.layers.get("spacy.FeatureExtractor.v1")


@spacy.registry.layers("MultiEmbed.v1")
def MultiEmbed(
    nO: int,
    unk: int,
    *,
    tables: Optional[Dict[str, Dict[int, int]]] = None,
    initializer: Callable = uniform_init,
    include_static_vectors: Optional[bool] = False,
    dropout: Optional[float] = None
) -> Model[List[Doc], Floats2d]:
    attrs = {
        "tables": tables,
        "unk": unk,
        "include_static_vectors": include_static_vectors
    }
    if dropout is not None:
        attrs["dropout"] = dropout
    else:
        attrs["dropout"] = 0.0
    # Two layers: embedding and output projection.
    layers = [noop, noop]
    model: Model = Model(
        "embed",
        forward,
        init=partial(init, initializer),
        attrs=attrs,
        dims={"nO": nO},
        layers=layers,
        params={},
    )
    return model


def forward(
        model: Model[List[Doc], List[Floats2d]],
        X: List[Doc],
        is_train=False
) -> Floats2d:
    embedding_layer = model.layers[0]
    output_layer = model.layers[1]
    embedded, bp_embed = embedding_layer(X, is_train)
    Y, bp_output = output_layer(X, is_train)

    def backprop(dY: Floats2d):
        dO = bp_output(dY)
        dX = bp_embed(dO)
        return dX

    return Y, backprop


def _make_embed(
    attr: str,
    unk: int,
    width: int,
    column: int,
    table: Dict[int, int],
    initializer: Callable,
    dropout: float
) -> Model[List[Doc], Floats2d]:
    """
    Helper function to create an embedding layer.
    """
    rows = len(table) + 1
    embedder = chain(
        Remap(table, default=unk, column=column),
        Embed(nO=width, nV=rows, column=column, dropout=dropout)
    )
    return embedder


def init(
    initializer: Callable,
    model: Model[List[Doc], Floats2d],
    X: Optional[List[Doc]] = None,
    Y: Optional = None,
) -> None:
    """
    Build and initialize the embedding and output
    all layers of MultiEmbed and initialize.
    """
    tables = model.attrs["tables"]
    unk = model.attrs["unk"]
    width = model.get_dim("nO")
    include_static_vectors = model.attrs["include_static_vectors"]
    dropout = model.attrs["dropout"]
    embeddings = []
    old_embeddings = model.layers[0]
    old_output = model.layers[1]
    attrs = []
    for i, (attr, mapper) in enumerate(tables.items()):
        attrs.append(attr)
        embedding = _make_embed(
            attr, unk, width, i, mapper, initializer, dropout
        )
        embeddings.append(embedding)
    full_width = (len(embeddings) + include_static_vectors) * width
    max_out = chain(
        with_array(
            Maxout(
                width,
                full_width,
                nP=3,
                dropout=dropout,
                normalize=True
            )
        ),
        ragged2list()
    )
    embedding_layer = chain(
        Extract(attrs),
        list2ragged(),
        with_array(concatenate(*embeddings)),
    )
    if include_static_vectors:
        embedding_layer = chain(
            concatenate(
                embedding_layer, Vectors()
            ),
            Dropout(rate=dropout)
        )
    embedding_layer.initialize(X)
    embedded = embedding_layer(X)
    max_out.initialize(embedded)
    model.replace_node(old_embeddings, embeddings)
    model.replace_node(old_output, max_out)
