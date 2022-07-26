from pathlib import Path
from typing import Optional, Dict

import typer
import yaml
from spacy import util
from spacy.cli._util import import_code, parse_config_overrides, setup_gpu
from spacy.cli._util import show_validation_error
from spacy.training.initialize import init_nlp
from spacy.training.loop import train
from thinc.api import Config
from wasabi import msg
from yaml.loader import SafeLoader

import wandb

app = typer.Typer()


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def main(
    # fmt: off
    ctx: typer.Context,  # This is only used to read additional arguments
    default_config: Path = typer.Argument(..., help="Path for the default spaCy config to override."),
    wandb_config: Path = typer.Argument(..., help="Path for the WandB YAML configuration."),
    code_path: Optional[Path] = typer.Option(None, "--code", "-c", help="Path to Python file with additional code (registered functions) to be imported."),
    project_id: str = typer.Option("spancat-paper", help="WandB project ID."),
    num_trials: int = typer.Option(2, help="Number of trials to run for each hyperparam combination"),
    use_gpu: int = typer.Option(0, help="GPU id to use. Pass -1 to use the CPU."),
    autofill_tok2vec_rows: Optional[int] = typer.Option(None, help="Update the tok2vec hyperparameter rows based on the chosen attr")
    # fmt: on
):

    overrides = parse_config_overrides(ctx.args)
    import_code(code_path)
    setup_gpu(use_gpu)

    def train_spacy():
        with show_validation_error(default_config):
            loaded_local_config = util.load_config(
                default_config, overrides=overrides, interpolate=False
            )
        with wandb.init() as run:
            sweeps_config = Config(util.dot_to_dict(run.config))
            merged_config = Config(loaded_local_config).merge(sweeps_config)

            # FIXME: WandB overrides
            if autofill_tok2vec_rows:
                merged_config = _autofill_tok2vec_rows(merged_config)

            with show_validation_error(merged_config, hint_fill=False):
                nlp = init_nlp(merged_config, use_gpu=use_gpu)
            output_path.mkdir(parents=True, exist_ok=True)
            train(nlp, output_path=None, use_gpu=use_gpu)

    with open(wandb_config) as f:
        sweep_config = yaml.load(f, Loader=SafeLoader)

    sweep_id = wandb.sweep(sweep_config, project=project_id)
    wandb.agent(sweep_id, train_spacy, count=num_trials)


def _autofill_tok2vec_rows(merged_config: Dict, row_size: 5000) -> Dict:
    """WandB overload that fills the components.tok2vec.model.embed.rows based on the attrs

    The problem with WandB is that it cannot do conditional hyperparameter setup. So we have
    to manually override the components.tok2vec.model.embed.rows with a value based on the
    chosen components.tok2vec.model.embed.attrs.
    """
    rows = merged_config["components"]["tok2vec"]["model"]["embed"]["rows"]
    attrs = merged_config["components"]["tok2vec"]["model"]["embed"]["attrs"]
    if len(rows) != len(attrs) or not all(i == row_size for i in rows):
        nrows = [row_size] * len(attrs)
        msg.text(f"Setting components.tok2vec.model.embed.rows to {nrows}")
        merged_config["components"]["tok2vec"]["model"]["embed"]["rows"] = nrows

    return merged_config


if __name__ == "__main__":
    app()