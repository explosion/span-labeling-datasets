"""Monkey-patched version of the convert command that transfers entities to Doc.spans"""

import itertools
from pathlib import Path
from typing import Iterable, Optional, Union

import srsly
import typer
from spacy.cli.convert import CONVERTERS, _print_docs_to_stdout
from spacy.cli.convert import _write_docs_to_file, walk_directory
from spacy.tokens import Doc, DocBin, SpanGroup
from spacy.training.gold_io import docs_to_json
from wasabi import Printer

FILE_TYPE = "spacy"


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
    typer.run(main)
