"""Split a spaCy-formatted file into train, dev, and test partitions"""

from pathlib import Path, Tuple

import typer
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option


def split_docs(
    # fmt: off
    input_path: Path, 
    output_dir: Path, 
    split_size: Tuple[float, float, float] = Arg((0.8, 0.1, 0.1), help="Split sizes for train/dev/test respectively"),
    # fmt: on
):
    pass
