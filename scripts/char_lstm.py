import numpy as np
import spacy

from spacy.tokens import Doc
from typing import Tuple, List
from thinc.types import Ints1d, Floats2d
from thinc.api import HashEmbed, Model, chain
from thinc.api import with_getitem, with_array, with_padded
from thinc.api import get_array_module


def docs2unicode() -> Model[List[Doc], Tuple[
        List[Ints1d], List[Ints1d], List[Ints1d]
]]:
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


def to_unicode(
    doc: Doc,
) -> Tuple[Ints1d, Ints1d, Ints1d]:
    """
    Convert Doc into a list of unicode codepoints
    and also return the word-boundaries.
    Adds white space " " between each token to signal
    word-boundaries.
    """
    token_starts = np.zeros(len(doc), dtype='int')
    token_ends = np.zeros(len(doc), dtype='int')
    byte_list = []
    for i, token in enumerate(doc):
        token_codepoint = [ord(char) for char in token.orth_ + " "]
        byte_list += token_codepoint
        if i == 0:
            token_ends[0] = len(token_codepoint) - 2
        else:
            token_starts[i] = token_ends[-1] + 2
            token_ends[i] = token_starts[i] + len(token_codepoint) - 2
    return np.array(byte_list), token_starts, token_ends


def forward_docs2unicode(
    model: Model,
    docs: List[Doc],
    is_train: bool
) -> Tuple[List[Ints1d], List[Ints1d], List[Ints1d]]:
    byte_list, start_list, end_list = [], [], []
    for doc in docs:
        unicode_data = to_unicode(doc)
        byte_list.append(unicode_data[0])
        start_list.append(unicode_data[1])
        end_list.append(unicode_data[2])

    def backprop(dY):
        return []

    return (byte_list, start_list, end_list), backprop


def forward_gather_tokens(
    model: Model,
    features_and_bounds: Tuple[List[Floats2d], List[Ints1d], List[Ints1d]],
    is_train: bool
) -> List[Floats2d]:
    bilstm_features, start_inds, end_inds = features_and_bounds
    assert len(bilstm_features) == len(start_inds)
    assert len(start_inds) == len(end_inds)
    tokens_features = []
    for char_feats, tok_starts, tok_ends in zip(
            bilstm_features, start_inds, end_inds
    ):
        xp = get_array_module(char_feats)
        token_features = xp.hstack(
            (char_feats[tok_starts], char_feats[tok_ends])
        )
        tokens_features.append(token_features)

        def backprop(dY: Floats2d):
            dY = xp.zeros_like(bilstm_features)
            d_start, d_end = xp.split(dY, 2, axis=1)
            dY[start_inds] = d_start
            dY[end_inds] = d_end
            return dY

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
    Character-level LSTM running on codepoints similar to CANINE.
    https://arxiv.org/pdf/2103.06874.pdf.

    Each character in the documents is mapped to a unicode codepoint.
    The codepoints are then fed to HashEmbed layer
    to produce embeddings for each character.

    The embedded codepoints are then forwarded through a BiLSTM.
    The representation of each token is the concatenation of the
    forward and backward LSTM states at the start and the end of the token.

    """
    embedder = HashEmbed(
        width, rows, seed=seed
    )
    bilstm_ = spacy.registry.architectures.get("spacy.TorchBiLSTMEncoder.v1")
    bilstm = bilstm_(width, depth, dropout)
    encode = chain(
        docs2unicode(),
        with_getitem(
            0,
            chain(
                with_array(embedder),
                with_padded(bilstm)
            )
        ),
        gather_tokens()
    )
    return encode


if __name__ == "__main__":
    emb = UnicodeLSTMEmbed()
    emb.initialize()
    nlp = spacy.blank("en")
    result, _ = emb([nlp("This is nice."), nlp("This is cool")], False)
    print(result[0].shape)
