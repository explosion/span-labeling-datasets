import spacy
import thinc

from typing import List, TypeVar, Optional, Callable
from pathlib import Path
from spacy.language import Language

from thinc.api import Model, concatenate, chain, Dropout, noop
from thinc.api import get_width, with_array, with_padded, Linear
from thinc.types import Floats2d
from spacy.tokens import Doc


InT = TypeVar("InT")
MidT = TypeVar("MidT")
OutT = TypeVar("OutT")


def freeze_tok2vec(layer: Model[InT, OutT]) -> Model[InT, OutT]:
    return Model(
        f"frozen({layer.name})",
        freeze_forward,
        init=freeze_init,
        layers=[layer]
    )


def freeze_forward(model: Model, X, is_train: bool):
    tok2vec = model.layers[0]
    Y, _ = tok2vec(X, is_train)

    def backprop(dY):
        return []

    return Y, backprop


def freeze_init(
    model: Model[InT, OutT],
    X: Optional[InT] = None,
    Y: Optional[OutT] = None,
) -> None:
    layer = model.layers[0]
    layer.initialize(X=X, Y=Y)


@spacy.registry.callbacks("set_lstms")
def create_load_tok2vecs_callback(
    charlstm_path: Path,
    tokenlstm_path: Path,
) -> Callable[[Language], Language]:

    def set_lstms(nlp: Language) -> Language:
        tok2vec = nlp.get_pipe("tok2vec")
        metalstm = tok2vec.model.get_ref("embed")
        if metalstm.attrs["has_lstm"]:
            return nlp
        nlp_char = spacy.load(charlstm_path)
        nlp_tok = spacy.load(tokenlstm_path)
        strings1 = nlp_char.vocab.strings
        strings2 = nlp_tok.vocab.strings
        # XXX I dunno the right way of checking whether two vocabs are the same
        if set(strings1) != set(strings2):
            raise ValueError("Vocabs don't seem to match")
        for i in strings1:
            if strings1.as_int(i) != strings2.as_int(i):
                raise ValueError("Vocabs don't seem to match")
        charlstm = nlp_char.get_pipe("tok2vec").model
        tokenlstm = nlp_tok.get_pipe("tok2vec").model
        layer = concatenate(
            freeze_tok2vec(charlstm),
            freeze_tok2vec(tokenlstm)
        )
        tok2null = metalstm.layers[0]
        metalstm.replace_node(tok2null, layer)
        metalstm.attrs["has_lstm"] = True
        return nlp
    return set_lstms


@spacy.registry.architectures("spacy.MetaLSTM.v1")
def MetaLSTM(
    width: int,
    depth: int,
    charlstm: Optional[Model[List[Doc], Floats2d]] = None,
    tokenlstm: Optional[Model[List[Doc], Floats2d]] = None,
    dropout: Optional[float] = None
) -> Model[List[Doc], Floats2d]:
    attrs = {"depth": depth}
    if dropout is not None:
        attrs["dropout"] = dropout
    else:
        attrs["dropout"] = 0.0
    if not bool(charlstm) == bool(tokenlstm):
        raise ValueError(
            "Either both charlstm and tokenlstm have to be "
            "provided or neither."
        )
    if charlstm is not None:
        layer = concatenate(charlstm, tokenlstm)
        attrs["has_lstm"] = True
    else:
        layer = noop()
        attrs["has_lstm"] = False
    model: Model = Model(
        "metalstm",
        forward_metalstm,
        init=init_metalstm,
        attrs=attrs,
        dims={"width": width},
        layers=[layer],
        params={},
    )
    return model


def init_metalstm(
    model: Model[List[Doc], Floats2d],
    X: Optional[List[Doc]] = None,
    Y: Optional = None
) -> None:
    if not model.attrs["has_lstm"]:
        raise ValueError(
            "LSTMs have to be provided on initialization"
        )
    dropout = model.attrs["dropout"]
    output_width = model.get_dim("width")
    depth = model.attrs["depth"]
    embed = model.layers[0]
    norm_ = thinc.registry.layers.get("LayerNorm.v1")
    dropout = Dropout(rate=dropout)
    metalstm_ = spacy.registry.architectures.get("spacy.TorchBiLSTMEncoder.v1")
    embedded, _ = embed(X, is_train=False)
    lstm_width = get_width(embedded)
    metalstm = chain(
        with_array(norm_()),
        with_padded(metalstm_(lstm_width, depth, dropout=0.0)),
        with_array(Linear(output_width))
    )
    metalstm.initialize(embedded)
    layer = chain(embed, dropout, metalstm)
    model.replace_node(embed, layer)


def forward_metalstm(
        model: Model[List[Doc], List[Floats2d]],
        X: List[Doc],
        is_train=False
) -> Floats2d:
    layer = model.layers[0]
    Y, bp_embed = layer(X, is_train)

    def backprop(dY: List[Floats2d]):
        return bp_embed(dY)

    return Y, backprop
