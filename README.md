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
| `convert-toxic-spans` | Convert Toxic Spans dataset into the spaCy format |
| `train-toxic-spans` | Train ner and spancat models for the Toxic Spans dataset |
| `evaluate-toxic-spans` | Evaluate models for the Toxic Spans dataset |
| `convert-sec_filings` | Convert SEC filings dataset into the spaCy format |
| `train-sec_filings` | Train an NER and Spancat model for SEC Filings dataset |
| `evaluate-sec_filings` | Evaluate the results for the SEC Filings dataset |
| `convert-wnut17` | Convert WNUT17 dataset into the spaCy format |
| `train-wnut17` | Train an ner and spancat model for WNUT17 |
| `evaluate-wnut17` | Evaluate the results for the WNUT17 dataset |
| `convert-btc` | Convert BTC dataset into the spaCy format |
| `convert-anem` | Convert AneM dataset into the spaCy format |
| `train-anem` | Train an NER and Spancat model for AnEM |
| `evaluate-anem` | Evaluate AnEM dataset |
| `convert-wikigold` | Convert the Wikigold dataset into the spaCy format |
| `clean-wikineural` | Remove unnecessary indices from wikineural data |
| `convert-wikineural-spans` | Convert WikiNeural dataset (de, en, es, nl) into the spaCy format |
| `convert-wikineural-ents` | Convert WikiNeural dataset (de, en, es, nl) into the spaCy format |
| `train-wikineural` | Train an ner and spancat model for Wikineural datasets |
| `evaluate-wikineural` | Evaluate the results for the Wikineural datasets |
| `clean` | Remove intermediary files |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `anem` | `convert-anem` &rarr; `train-anem` &rarr; `evaluate-anem` |
| `sec-filings` | `convert-sec_filings` &rarr; `train-sec_filings` &rarr; `evaluate-sec_filings` |
| `wnut17` | `convert-wnut17` &rarr; `train-wnut17` &rarr; `evaluate-wnut17` |
| `wikineural` | `clean-wikineural` &rarr; `convert-wikineural-ents` &rarr; `convert-wikineural-spans` &rarr; `train-wikineural` &rarr; `evaluate-wikineural` |
| `toxic-spans` | `convert-toxic-spans` &rarr; `train-toxic-spans` &rarr; `evaluate-toxic-spans` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/sec_filings-train.iob` | URL | SEC filings training dataset from Alvarado et al. (ALTA 2015) |
| `assets/sec_filings-test.iob` | URL | SEC filings test dataset from Alvarado et al. (ALTA 2015) |
| `assets/wnut17-train.iob` | URL | WNUT17 training dataset for Emerging and Rare Entities Task from Derczynski et al., 2017 |
| `assets/wnut17-dev.iob` | URL | WNUT17 dev dataset for Emerging and Rare Entities Task from Derczynski et al., 2017 |
| `assets/wnut17-test.iob` | URL | WNUT17 test dataset for Emerging and Rare Entities Task from Derczynski et al., 2017 |
| `assets/btc-general.iob` | URL | Broad Twitter Corpus (BTC) containing UK general tweets from Derczynski et al., 2016 |
| `assets/anem-train.iob` | URL | Anatomical Entity Mention (AnEM) training corpus containing abstracts and full-text biomedical papers from Ohta et al. (ACL 2012) |
| `assets/anem-test.iob` | URL | Anatomical Entity Mention (AnEM) test corpus containing abstracts and full-text biomedical papers from Ohta et al. (ACL 2012) |
| `assets/wikigold.iob` | URL | Wikigold dataset containing a manually annotated collection of Wikipedia text by Balasuriya et al. (ACL 2009). |
| `assets/en-wikineural-train.iob` | URL | WikiNeural (en) training dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/en-wikineural-dev.iob` | URL | WikiNeural (en) dev dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/en-wikineural-test.iob` | URL | WikiNeural (en) test dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/de-wikineural-train.iob` | URL | WikiNeural (de) training dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/de-wikineural-dev.iob` | URL | WikiNeural (de) dev dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/de-wikineural-test.iob` | URL | WikiNeural (de) test dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/es-wikineural-train.iob` | URL | WikiNeural (es) training dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/es-wikineural-dev.iob` | URL | WikiNeural (es) dev dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/es-wikineural-test.iob` | URL | WikiNeural (es) test dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/nl-wikineural-train.iob` | URL | WikiNeural (nl) training dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/nl-wikineural-dev.iob` | URL | WikiNeural (nl) dev dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/nl-wikineural-test.iob` | URL | WikiNeural (nl) test dataset from Tedeschi et al. (EMNLP 2021) |
| `assets/toxic_spans.csv` | URL | Toxic Spans dataset from Pavlopoulos et al. (ACL 2022) |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->