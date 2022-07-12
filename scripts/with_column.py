from typing import Tuple, Callable, Sequence
from typing import Optional, Dict, Any, Union

from thinc.api import Model
from thinc.api import registry
from thinc.types import Ints2d, DTypes, Array2d


InT = Union[Sequence[Any], Array2d]
OutT = Ints2d


@registry.layers("remap_ids.v2")
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
        forward,
        attrs={
            "table": table,
            "dtype": dtype,
            "default": default,
            "column": column
        },
    )


def forward(
    model: Model[InT, OutT], inputs: InT, is_train: bool
) -> Tuple[OutT, Callable]:
    table = model.attrs["table"]
    default = model.attrs["default"]
    dtype = model.attrs["dtype"]
    column = model.attrs["column"]
    if column is not None:
        inputs = inputs[:, column]
    values = [table.get(x, default) for x in inputs]
    arr = model.ops.asarray2i(values, dtype=dtype)
    output = model.ops.reshape2i(arr, -1, 1)

    def backprop(dY: OutT) -> InT:
        return []
    return output, backprop
