"""Softmax scorer for exclusive classes in spancat"""

from spacy import registry
from thinc.api import Softmax, Model, glorot_uniform_init
from thinc.types import Floats2d


@registry.layers("spacy.Softmax.v1")
def build_softmax(nO=None, nI=None) -> Model[Floats2d, Floats2d]:
    """An output layer for span classification. Uses a softmax layer for exclusive classes"""
    return Softmax(nO=nO, nI=nI, init_W=glorot_uniform_init)
