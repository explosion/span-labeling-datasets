import spacy
import thinc
import numpy as np

from spacy.tokens import Doc
from typing import Tuple, List
from thinc.types import Ints1d, Floats2d
from thinc.api import HashEmbed, Model, chain
from thinc.api import with_getitem, with_array, with_padded
from thinc.api import get_array_module


def docs2unicode() -> Model[List[Doc], Tuple[List[Ints1d], List[Ints1d], List[Ints1d]]]:
    """
    Convert a list of docs into variable length unicode arrays.
    The layer also returns the start and end indices for each token.
    """
    return Model("doc2bytes", forward_docs2unicode)


def gather_tokens() -> Model:
    """
    Takes the Character-level LSTM features and concatenates
    the start and end features for each word. For backprop it
    returns 0 vectors everywhere except where the features were
    gathered from i.e.: it backprops to 2 * n_tokens hidden states.
    """
    return Model("gather_tokens", forward_gather_tokens)


def to_unicode(doc: Doc, xp) -> Tuple[Ints1d, Ints1d, Ints1d]:
    """
    Convert Doc into a list of unicode codepoints
    and also return the word-boundaries.
    Adds white space " " between each token to signal
    word-boundaries.
    """
    token_starts = xp.zeros(len(doc), dtype="int")
    token_ends = xp.zeros(len(doc), dtype="int")
    byte_list = []
    for i, token in enumerate(doc):
        token_codepoint = [ord(char) for char in token.orth_ + " "]
        byte_list += token_codepoint
        if i == 0:
            token_ends[0] = len(token_codepoint) - 1
        else:
            token_starts[i] = token_ends[i - 1] + 1
            token_ends[i] = token_starts[i] + len(token_codepoint) - 1
    return xp.array(byte_list), token_starts, token_ends


def forward_docs2unicode(
    model: Model, docs: List[Doc], is_train: bool
) -> Tuple[List[Ints1d], List[Ints1d], List[Ints1d]]:
    unicodes_list, start_list, end_list = [], [], []
    xp = model.ops.xp
    for doc in docs:
        unicodes, starts, ends = to_unicode(doc, xp)
        unicodes_list.append(unicodes)
        start_list.append(starts)
        end_list.append(ends)

    def backprop(dY):
        return []

    return (unicodes_list, start_list, end_list), backprop


def forward_gather_tokens(
    model: Model,
    features_and_bounds: Tuple[List[Floats2d], List[Ints1d], List[Ints1d]],
    is_train: bool,
) -> List[Floats2d]:
    bilstm_features, start_inds, end_inds = features_and_bounds
    assert len(bilstm_features) == len(start_inds)
    assert len(start_inds) == len(end_inds)

    def backprop(dY: List[Floats2d]) -> Tuple[List[Floats2d], List]:
        l_dX = []
        for i, dy in enumerate(dY):
            dX = xp.zeros_like(bilstm_features[i])
            d_start, d_end = xp.split(dy, 2, axis=1)
            dX[start_inds[i]] = d_start
            dX[end_inds[i]] = d_end
            l_dX.append(dX)
        return l_dX, []

    tokens_features = []
    for char_feats, tok_starts, tok_ends in zip(bilstm_features, start_inds, end_inds):
        xp = get_array_module(char_feats)
        token_features = xp.hstack((char_feats[tok_starts], char_feats[tok_ends]))
        tokens_features.append(token_features)

    return tokens_features, backprop


@spacy.registry.architectures("spacy.UnicodeLSTMEmbed.v1")
def UnicodeLSTMEmbed(
    *,
    width: int = 256,
    rows: int = 1000,
    depth: int = 2,
    seed: int = 100101,
    dropout: float = 0.2
):
    """
    Character-level LSTM running on codepoints embedded
    with the hashing-trick -- similar to CANINE.
    https://arxiv.org/pdf/2103.06874.pdf.

    Each character in the documents is mapped to a unicode codepoint.
    The codepoints are then fed to HashEmbed layer
    to produce embeddings for each character.

    The embedded codepoints are then forwarded through a BiLSTM.
    The representation of each token is the concatenation of the
    forward and backward LSTM states at the start and the end of the token.

    """
    embedder = HashEmbed(width, rows, seed=seed)
    bilstm_ = spacy.registry.architectures.get("spacy.TorchBiLSTMEncoder.v1")
    norm = thinc.registry.layers.get("LayerNorm.v1")
    relu = thinc.registry.layers.get("Relu.v1")
    bilstm = bilstm_(width, depth, dropout)
    mlp_out = chain(relu(width, normalize=True), relu(width, normalize=True))
    encode = chain(
        docs2unicode(),
        with_getitem(
            0, chain(with_array(embedder), with_padded(bilstm), with_array(mlp_out))
        ),
        gather_tokens(),
    )
    return encode


@spacy.registry.architectures("spacy.IdentityEncode.v1")
def IdentityEncode():
    """
    Give this to the Tok2Vec as encoder when only trying
    to pre-train contextual-embedders.
    """
    identity = thinc.registry.layers.get("noop.v1")
    return identity()


if __name__ == "__main__":
    emb = UnicodeLSTMEmbed()
    emb.initialize()
    nlp = spacy.blank("en")
    result, _ = emb([nlp("This is nice."), nlp("This is cool")], False)
    print(result[0].shape)