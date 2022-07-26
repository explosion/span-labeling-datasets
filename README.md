<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Spancat datasets

This project compiles various spancat datasets and their converters into the
[spaCy format](https://spacy.io/api/data-formats).


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `install` | Install dependencies |
| `init-fasttext` | Initialize the FastText vectors |
| `convert-wnut17` | Convert WNUT17 dataset into the spaCy format |
| `train-wnut17-ner` | Train an ner model for WNUT17 |
| `train-wnut17-spancat` | Train a spancat model for WNUT17 |
| `evaluate-wnut17-ner` | Evaluate the NER results for the WNUT17 dataset |
| `evaluate-wnut17-spancat` | Evaluate the spancat results for the WNUT17 dataset |
| `clean-wikineural` | Remove unnecessary indices from wikineural data |
| `convert-wikineural-spans` | Convert WikiNeural dataset (de, en, es, nl) into the spaCy format |
| `convert-wikineural-ents` | Convert WikiNeural dataset (de, en, es, nl) into the spaCy format |
| `make-wikineural-tables` | Pre-compute token-to-id tables from the Wikineural training sets. |
| `train-wikineural-spancat` | Train a spancat model for Wikineural datasets |
| `train-wikineural-ner` | Train an ner model for Wikineural datasets |
| `evaluate-wikineural-ner` | Evaluate the ner results for the Wikineural datasets |
| `evaluate-wikineural-spancat` | Evaluate the spancat results for the Wikineural datasets |
| `clean-conll` | Remove unnecessary indices from ConLL data |
| `convert-conll-spans` | Convert CoNLL dataset (de, en, es, nl) into the spaCy format |
| `convert-conll-ents` | Convert CoNLL dataset (de, en, es, nl) into the spaCy format |
| `make-conll-tables` | Pre-compute token-to-id tables from the Wikineural training sets. |
| `train-conll-spancat` | Train a spancat model for Wikineural datasets |
| `train-conll-ner` | Train an ner model for Wikineural datasets |
| `evaluate-conll-ner` | Evaluate the ner results for the CoNLL datasets |
| `evaluate-conll-spancat` | Evaluate the spancat results for the Wikineural datasets |
| `clean` | Remove intermediary files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `wnut17-ner` | `convert-wnut17` &rarr; `train-wnut17-ner` &rarr; `evaluate-wnut17-ner` |
| `wnut17-spancat` | `convert-wnut17` &rarr; `train-wnut17-spancat` &rarr; `evaluate-wnut17-spancat` |
| `wikineural-ner` | `clean-wikineural` &rarr; `convert-wikineural-ents` &rarr; `train-wikineural-ner` &rarr; `evaluate-wikineural-ner` |
| `wikineural-spancat` | `clean-wikineural` &rarr; `convert-wikineural-spans` &rarr; `train-wikineural-spancat` &rarr; `evaluate-wikineural-spancat` |
| `conll-ner` | `clean-conll` &rarr; `convert-conll-ents` &rarr; `train-conll-ner` &rarr; `evaluate-conll-ner` |
| `conll-spancat` | `clean-conll` &rarr; `convert-conll-spans` &rarr; `train-conll-spancat` &rarr; `evaluate-conll-spancat` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/wnut17-train.iob` | Git | WNUT17 training dataset for Emerging and Rare Entities Task from Derczynski et al., 2017 |
| `assets/wnut17-dev.iob` | Git | WNUT17 dev dataset for Emerging and Rare Entities Task from Derczynski et al., 2017 |
| `assets/wnut17-test.iob` | Git | WNUT17 test dataset for Emerging and Rare Entities Task from Derczynski et al., 2017 |
| `assets/raw-en-wikineural-train.iob` | Git | WikiNeural (en) training dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-en-wikineural-dev.iob` | Git | WikiNeural (en) dev dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-en-wikineural-test.iob` | Git | WikiNeural (en) test dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-de-wikineural-train.iob` | Git | WikiNeural (de) training dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-de-wikineural-dev.iob` | Git | WikiNeural (de) dev dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-de-wikineural-test.iob` | Git | WikiNeural (de) test dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-es-wikineural-train.iob` | Git | WikiNeural (es) training dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-es-wikineural-dev.iob` | Git | WikiNeural (es) dev dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-es-wikineural-test.iob` | Git | WikiNeural (es) test dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-nl-wikineural-train.iob` | Git | WikiNeural (nl) training dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-nl-wikineural-dev.iob` | Git | WikiNeural (nl) dev dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-nl-wikineural-test.iob` | Git | WikiNeural (nl) test dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/raw-en-conll-train.iob` | Git | CoNLL 2003 (en) training dataset |
| `assets/raw-en-conll-dev.iob` | Git | CoNLL 2003 (en) dev dataset |
| `assets/raw-en-conll-test.iob` | Git | CoNLL 2003 (en) test dataset |
| `assets/raw-de-conll-train.iob` | Git | CoNLL 2003 (de) training dataset |
| `assets/raw-de-conll-dev.iob` | Git | CoNLL 2003 (de) dev dataset |
| `assets/raw-de-conll-test.iob` | Git | CoNLL 2003 (de) test dataset |
| `assets/raw-es-conll-train.iob` | Git | CoNLL 2002 (es) training dataset |
| `assets/raw-es-conll-dev.iob` | Git | CoNLL 2002 (es) dev dataset |
| `assets/raw-es-conll-test.iob` | Git | CoNLL (es) test dataset |
| `assets/raw-nl-conll-train.iob` | Git | CoNLL 2002 (nl) training dataset |
| `assets/raw-nl-conll-dev.iob` | Git | CoNLL 2002 (nl) dev dataset |
| `assets/raw-nl-conll-test.iob` | Git | CoNLL 202 (nl) test dataset |
| `assets/fasttext.en.gz` | URL |  |
| `assets/fasttext.de.gz` | URL | German fastText vectors. |
| `assets/fasttext.es.gz` | URL | Spanish fastText vectors. |
| `assets/fasttext.nl.gz` | URL | Dutch fastText vectors. |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->