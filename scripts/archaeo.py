from scripts.convert_to_spans import convert
from pathlib import Path
from typing import Optional, Union

import typer
from spacy.cli.convert import CONVERTERS, _get_converter
from spacy.cli.convert import verify_cli_args
from wasabi import Printer

FILE_TYPE = "spacy"
Arg = typer.Argument
Opt = typer.Option


def convert_cli(
    # fmt: off
    input_path: str = Arg(..., help="Input file or directory", exists=True),
    output_dir: Path = Arg("-", help="Output directory. '-' for stdout.", allow_dash=True, exists=True),
    spans_key: str = Opt("sc", "--spans-key", "-sc", help="Spans key to use when storing entities"),
    n_sents: int = Opt(1, "--n-sents", "-n", help="Number of sentences per doc (0 to disable)"),
    seg_sents: bool = Opt(False, "--seg-sents", "-s", help="Segment sentences (for -c ner)"),
    model: Optional[str] = Opt(None, "--model", "--base", "-b", help="Trained spaCy pipeline for sentence segmentation to use as base (for --seg-sents)"),
    morphology: bool = Opt(False, "--morphology", "-m", help="Enable appending morphology to tags"),
    merge_subtokens: bool = Opt(False, "--merge-subtokens", "-T", help="Merge CoNLL-U subtokens"),
    converter: str = Opt("auto", "--converter", "-c", help=f"Converter: {tuple(CONVERTERS.keys())}"),
    ner_map: Optional[Path] = Opt(None, "--ner-map", "-nm", help="NER tag mapping (as JSON-encoded dict of entity types)", exists=True),
    lang: Optional[str] = Opt(None, "--lang", "-l", help="Language (if tokenizer required)"),
    use_ents: bool = Opt(False, "--use-ents", "-e", help="Use Doc.ents, don't transfer to Doc.spans"),
    train_size: Optional[float] = Opt(None, "--train-size", "-sz", help="Size of the training dataset for splitting"),
    shuffle: bool = Opt(False, "--shuffle", "-sf", help="Shuffle the dataset before splitting"),
    seed: Optional[int] = Opt(None, "--seed", "-sd", help="Random seed for shuffling the data")
    # fmt: on
):
    """
    Convert files into json or DocBin format for training. The resulting .spacy
    file can be used with the train command and other experiment management
    functions.

    If no output_dir is specified and the output format is JSON, the data
    is written to stdout, so you can pipe them forward to a JSON file:
    $ spacy convert some_file.conllu --file-type json > some_file.json

    NOTE: This is a monkeypatched version of the original `convert` command.
    Here, we added an additional step that transfers the entitites into
    the Doc.spans attribute for Span Categorization.

    DOCS: https://spacy.io/api/cli#convert
    """
    input_path = Path(input_path)
    output_dir: Union[str, Path] = "-" if output_dir == Path("-") else output_dir
    silent = output_dir == "-"
    msg = Printer(no_print=silent)
    verify_cli_args(msg, input_path, output_dir, FILE_TYPE, converter, ner_map)
    converter = _get_converter(msg, converter, input_path)
    convert(
        input_path,
        output_dir,
        n_sents=n_sents,
        seg_sents=seg_sents,
        model=model,
        morphology=morphology,
        merge_subtokens=merge_subtokens,
        converter=converter,
        ner_map=ner_map,
        lang=lang,
        silent=silent,
        msg=msg,
        spans_key=spans_key,
        use_ents=use_ents,
        train_size=train_size,
        shuffle=shuffle,
        seed=seed,
    )
