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
| `convert-sec_filings` | Convert SEC filings dataset into the spaCy format |
| `convert-wnut17` | Convert WNUT17 dataset into the spaCy format |
| `convert-btc` | Convert BTC dataset into the spaCy format |
| `convert-anem` | Convert AneM dataset into the spaCy format |
| `clean` | Remove intermediary files |

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

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->