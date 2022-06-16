import pandas as pd
from pathlib import Path
from wasabi import msg

import typer


def preprocess_archaeo(
    input_path: Path,
    output_path: Path,
    force: bool = False,
):
    df = pd.read_csv(input_path, delimiter=" ")
    file_not_yet_processed = "Parts" in df.columns  # naive check

    if file_not_yet_processed or force:
        df = df.drop("Parts", axis=1)
        dfs = [group for _, group in df.groupby("SentenceId")]

        if output_path.exists():
            msg.warn(
                f"Output path '{output_path}' exists, will delete the file"
                " and replace it with a new version"
            )
            output_path.unlink(missing_ok=True)

        with open(output_path, "a") as f:
            for gdf in dfs:
                gdf.drop("SentenceId", axis=1).to_csv(
                    f, index=False, header=False, sep="\t"
                )
                f.write("\n")
            msg.good(f"Saved output to {output_path}")

    else:
        msg.info(
            "It looks like the file has already been processed."
            " You can force this command by passing the --force parameter."
        )


if __name__ == "__main__":
    typer.run(preprocess_archaeo)
