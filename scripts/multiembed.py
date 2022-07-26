import spacy
import thinc
import srsly
from pathlib import Path

from spacy.tokens import Doc
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from thinc.api import Model, uniform_init, chain, with_array
from thinc.api import list2ragged, ragged2list, concatenate, noop
from thinc.types import Floats2d, DTypes, Ints2d, Array2d
from typing import Optional, Callable, List, Dict, Any
from typing import Union, Sequence, Tuple

Embed = thinc.registry.layers.get("Embed.v1")
Maxout = thinc.registry.layers.get("Maxout.v1")
Dropout = thinc.registry.layers("Dropout.v1")
Vectors = thinc.registry.layers("spacy.StaticVectors.v2")
Extract = thinc.registry.layers.get("spacy.FeatureExtractor.v1")


InT = Union[Sequence[Any], Array2d]
OutT = Ints2d


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
    """
    attr_value = srsly.read_msgpack(path)

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

@thinc.registry.layers("remap_ids.v2")
def remap_ids(
    table: Dict[Any, int] = {},
    default: int = 0,
    dtype: DTypes = "i",
    column: Optional[int] = None
) -> Model[InT, OutT]:
    """Remap string or integer inputs using a mapping table, usually as a
    preprocess before embeddings. The mapping table can be passed in on input,
    or updated after the layer has been created. The mapping table is stored in
    the "mapping_table" attribute.
    """
    return Model(
        "remap_ids",
        remap_forward,
        attrs={
            "table": table,
            "dtype": dtype,
            "default": default,
            "column": column
        },
    )


def remap_forward(
    model: Model[InT, OutT], inputs: InT, is_train: bool
) -> Tuple[OutT, Callable]:
    table = model.attrs["table"]
    default = model.attrs["default"]
    dtype = model.attrs["dtype"]
    column = model.attrs["column"]
    if column is not None:
        inputs = inputs[:, column]
    # We wrap int around x, because in cupy each integer
    # in the arrays is a cuda array with shape ()
    values = [table.get(int(x), default) for x in inputs]
    arr = model.ops.asarray2i(values, dtype=dtype)
    output = model.ops.reshape2i(arr, -1, 1)

    def backprop(dY: OutT) -> InT:
        return model.ops.asarray([])

    return output, backprop


@spacy.registry.architectures("spacy.MultiEmbed.v1")
def MultiEmbed(
    width: int,
    unk: int,
    *,
    tables: Optional[Dict[str, Dict[int, int]]] = None,
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
    layers = [noop(), noop()]
    model: Model = Model(
        "embed",
        forward,
        init=init,
        attrs=attrs,
        dims={"width": width},
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
    Y, bp_output = output_layer(embedded, is_train)

    def backprop(dY: List[Floats2d]):
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
    dropout: float
) -> Model[List[Doc], Floats2d]:
    """
    Helper function to create an embedding layer.
    """
    rows = len(table) + 1
    embedder = chain(
        remap_ids(table, default=unk, column=column),
        Embed(nO=width, nV=rows, column=0, dropout=dropout)
    )
    return embedder


def init(
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
    width = model.get_dim("width")
    include_static_vectors = model.attrs["include_static_vectors"]
    dropout = model.attrs["dropout"]
    embeddings = []
    old_embeddings = model.layers[0]
    old_output = model.layers[1]
    attrs = []
    for i, (attr, mapper) in enumerate(tables.items()):
        attrs.append(attr)
        embedding = _make_embed(
            attr, unk, width, i, mapper, dropout
        )
        embeddings.append(embedding)
    full_width = (len(embeddings) + include_static_vectors) * width
    max_out = chain(
        with_array(
            Maxout(
                width, full_width, nP=3, dropout=dropout, normalize=True
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
    embedded, _ = embedding_layer(X, is_train=False)
    max_out.initialize(embedded)
    model.replace_node(old_embeddings, embedding_layer)
    model.replace_node(old_output, max_out)
