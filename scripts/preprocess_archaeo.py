import pandas as pd
from pathlib import Path

import typer


def preprocess_archaeo(input_path: Path, force: bool = False):
    df = pd.read_csv(input_path, delimiter=" ").drop("Parts")
    dfs = [group for _, group in df.groupby("SentenceId")]
    breakpoint()
    pass


if __name__ == "__main__":
    typer.run(preprocess_archaeo)
