import typer
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal
from wasabi import msg

FINAL_COLUMNS = ["dataset", "label", "p", "r", "f", "speed"]


def _convert_to_df(
    name: str, data: Dict, component: Literal["ents", "spans_sc"] = "ents"
):
    # Prepare main dataframe
    df = pd.DataFrame.from_dict(data, orient="index").transpose()
    metrics = [f"{component}_{m}" for m in ("p", "r", "f")]
    meta = ["speed", f"{component}_per_type"]
    df = df[metrics + meta]

    # Get ents per type results
    res_per_type = df[f"{component}_per_type"].values.tolist()[0]
    tmp = []
    for label, results in res_per_type.items():
        results["label"] = label
        tmp.append(results)
    res_per_type_df = pd.DataFrame.from_dict(tmp)

    # Combine the two dataframes together
    df = df.drop(f"{component}_per_type", axis=1).rename(
        columns={m: m[-1] for m in metrics}
    )
    df["dataset"] = name
    final_df = pd.concat([df, res_per_type_df])
    final_df = final_df[FINAL_COLUMNS]
    return final_df


def main(metrics_dir: Path, output_dir: Path):
    """Collate the results from different experiments and put them into one CSV

    It works better if the directory structure of 'metrics_dir' looks like this:
        metrics
        |---- ${dataset}
            |--- ner
            |--- spancat

    If you have different configurations, it might be better if they're in a
    separate metrics directory.
    """
    ner_score_paths = metrics_dir.glob("**/ner/*")
    spancat_score_paths = metrics_dir.glob("**/spancat/*")

    ner_results = []
    spancat_results = []
    for ner_path, spancat_path in zip(ner_score_paths, spancat_score_paths):
        with ner_path.open() as f:
            data = json.load(f)
        ner = _convert_to_df(ner_path.parts[1], data, component="ents")
        ner_results.append(ner)

        with spancat_path.open() as f:
            data = json.load(f)
        spancat = _convert_to_df(spancat_path.parts[1], data, component="spans_sc")
        spancat_results.append(spancat)

    # Save results to disk
    results = {"ner": ner_results, "spancat": spancat_results}
    for model, result in results.items():
        result_df = pd.concat(result)
        timestamp = datetime.now().replace(microsecond=0).isoformat()
        output_path = output_dir / f"{timestamp}-{model}.csv"
        result_df.to_csv(output_path, index=False)
        msg.good(f"Processed {len(results)} ({model}) datasets to {output_path}")


if __name__ == "__main__":
    typer.run(main)
