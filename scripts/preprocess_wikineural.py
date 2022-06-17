from pathlib import Path
from string import digits

import typer
from wasabi import msg
import pandas as pd

Arg = typer.Argument
Opt = typer.Option


def preprocess_wikineural(input_path: Path):
    """Helper function to remove the indices for the WikiNeural dataset"""
    with input_path.open() as f:
        lines = f.readlines()

    doc_delimiter = "-DOCSTART-\tO\n"

    with input_path.open("w") as f:
        for line in lines:
            if doc_delimiter in line:
                pass
            else:
                new_line = line if line == "\n" else line.lstrip(digits)[1:]
                f.write(new_line)


if __name__ == "__main__":
    typer.run(preprocess_wikineural)
