"""Monkey-patched version of the convert command that transfers entities to Doc.spans"""

import itertools
from pathlib import Path
from typing import Iterable, Optional, Union

import srsly
import typer
from spacy.cli.convert import CONVERTERS, _print_docs_to_stdout
from spacy.cli.convert import _write_docs_to_file, walk_directory
from spacy.cli.convert import verify_cli_args, _get_converter
from spacy.tokens import Doc, DocBin, SpanGroup
from spacy.training.gold_io import docs_to_json
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
    concatenate: bool = Opt(None, "--concatenate", "-C", help="Concatenate output to a single file"),
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
        concatenate=concatenate,
        silent=silent,
        msg=msg,
        spans_key=spans_key,
    )


def transfer_ents_to_spans(docs: Iterable[Doc], spans_key: str) -> Iterable[Doc]:
    _docs = []
    for doc in docs:
        if spans_key not in doc.spans:
            doc.spans[spans_key] = SpanGroup(doc)
        doc.spans[spans_key].extend(list(doc.ents))
        doc.set_ents([])
        _docs.append(doc)
    return _docs


def convert(
    input_path: Path,
    output_dir: Union[str, Path],
    *,
    n_sents: int = 1,
    seg_sents: bool = False,
    model: Optional[str] = None,
    morphology: bool = False,
    merge_subtokens: bool = False,
    converter: str = "auto",
    ner_map: Optional[Path] = None,
    lang: Optional[str] = None,
    concatenate: bool = False,
    silent: bool = True,
    msg: Optional[Printer] = None,
    spans_key: str = "sc",
) -> None:
    input_path = Path(input_path)
    if not msg:
        msg = Printer(no_print=silent)
    ner_map = srsly.read_json(ner_map) if ner_map is not None else None
    doc_files = []
    for input_loc in walk_directory(input_path, converter):
        with input_loc.open("r", encoding="utf-8") as infile:
            input_data = infile.read()
        # Use converter function to convert data
        func = CONVERTERS[converter]
        docs = func(
            input_data,
            n_sents=n_sents,
            seg_sents=seg_sents,
            append_morphology=morphology,
            merge_subtokens=merge_subtokens,
            lang=lang,
            model=model,
            no_print=silent,
            ner_map=ner_map,
        )
        # Monkeypatched version converting docs to spans
        docs = transfer_ents_to_spans(docs, spans_key)
        doc_files.append((input_loc, docs))
    if concatenate:
        all_docs = itertools.chain.from_iterable([docs for _, docs in doc_files])
        doc_files = [(input_path, all_docs)]
    for input_loc, docs in doc_files:
        db = DocBin(docs=docs, store_user_data=True)
        len_docs = len(db)
        data = db.to_bytes()  # type: ignore[assignment]
        if output_dir == "-":
            _print_docs_to_stdout(data, FILE_TYPE)
        else:
            if input_loc != input_path:
                subpath = input_loc.relative_to(input_path)
                output_file = Path(output_dir) / subpath.with_suffix(f".{FILE_TYPE}")
            else:
                output_file = Path(output_dir) / input_loc.parts[-1]
                output_file = output_file.with_suffix(f".{FILE_TYPE}")
            _write_docs_to_file(data, output_file, FILE_TYPE)
            msg.good(f"Generated output file ({len_docs} documents): {output_file}")


if __name__ == "__main__":
    typer.run(convert_cli)
