
## Overview


## Repository Structure

```text
.
├── README.md
├── config.json
├── data.py
├── evaluate_.py
├── generate.py
├── main.py
├── prep_elastic.py
└── retriever.py
```

### File Description

- `main.py`: main entry for GCDT inference
- `retriever.py`: retrieval utilities
- `prep_elastic.py`: build the Elasticsearch index for Wikipedia passages
- `generate.py`: generation-related utilities
- `data.py`: dataset loading and preprocessing
- `evaluate_.py`: evaluation script
- `config.json`: example runtime configuration

---

## Method Summary



---

## Installation

We recommend using **Python 3.9**.

```bash
conda create -n etc python=3.9
conda activate etc
pip install torch==2.1.1 transformers==4.30.2(if llama3,transformers==4.44.0) beir==1.0.1
python -m spacy download en_core_web_sm
```

---

## Prepare the Retriever

This repository uses a Wikipedia passage collection together with Elasticsearch to build the retriever.

### 1. Download Wikipedia Passages

```bash
mkdir -p data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

### 2. Install and Start Elasticsearch

```bash
cd data
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz
tar zxvf elasticsearch-7.17.9.tar.gz
rm elasticsearch-7.17.9.tar.gz
cd elasticsearch-7.17.9
nohup bin/elasticsearch &
```

### 3. Build the Wikipedia Index

```bash
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki
```

---

## Datasets

ETC is evaluated on the following QA benchmarks:

- **2WikiMultihopQA**
- **HotpotQA**
- **StrategyQA**
- **IIRC**
- **BioASQ**
- **PubMedQA**

### Download Instructions

#### 2WikiMultihopQA

Download the dataset manually from its official repository, then unzip it and move the folder to:

```text
data/2wikimultihopqa
```

Reference download link:

```text
https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?e=1
```

#### StrategyQA

```bash
wget -O data/strategyqa_dataset.zip https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip
mkdir -p data/strategyqa
unzip data/strategyqa_dataset.zip -d data/strategyqa
rm data/strategyqa_dataset.zip
```

#### HotpotQA

```bash
mkdir -p data/hotpotqa
wget -O data/hotpotqa/hotpotqa-dev.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

#### IIRC

```bash
wget -O data/iirc.tgz https://iirc-dataset.s3.us-west-2.amazonaws.com/iirc_train_dev.tgz
tar -xzvf data/iirc.tgz
mv iirc_train_dev/ data/iirc
rm data/iirc.tgz
```

#### BioASQ

```bash
mkdir -p data/bioasq_7b_yesno
wget -O data/bioasq_7b_yesno/Task7B_yesno_train.json \
  https://huggingface.co/datasets/nanyy1025/bioasq_7b_yesno/resolve/main/Task7B_yesno_train.json
wget -O data/bioasq_7b_yesno/Task7B_yesno_validation.json \
  https://huggingface.co/datasets/nanyy1025/bioasq_7b_yesno/resolve/main/Task7B_yesno_validation.json
wget -O data/bioasq_7b_yesno/Task7B_yesno_test.json \
  https://huggingface.co/datasets/nanyy1025/bioasq_7b_yesno/resolve/main/Task7B_yesno_test.json
```

#### PubMedQA

```bash
mkdir -p data/pubmedQA
wget -O data/pubmedQA/pqal_train_set.json \
  https://huggingface.co/datasets/tan9/pubmedQA/resolve/main/pqal_train_set.json
wget -O data/pubmedQA/test_set.json \
  https://huggingface.co/datasets/tan9/pubmedQA/resolve/main/test_set.json
```

---

## Configuration

Main runtime options are specified in `config.json`.

### Important Arguments

| Argument | Description | Example |
|---|---|---|
| `model_name_or_path` | Hugging Face model path | `meta-llama/Llama-2-13b-chat` |
| `dataset` | Dataset name | `2wikimultihopqa`, `hotpotqa`, `iirc`, `strategyqa` |
| `data_path` | Dataset directory | `../data/2wikimultihopqa` |
| `fewshot` | Number of few-shot examples | `6` |
| `sample` | Number of sampled questions. `-1` means all data | `1000` |
| `shuffle` | Whether to shuffle the dataset | `true`, `false` |
| `generate_max_length` | Maximum generated query length | `64` |
| `query_formulation` | Retrieval query generation strategy | `direct`, `real_words`, `current`, `last_sentence` |
| `retrieve_keep_top_k` | Number of reserved tokens for query construction | `35` |
| `output_dir` | Output directory for results | `../result/2wikimultihopqa_llama2_13b` |
| `retriever` | Retriever type | `BM25`, `SGPT` |
| `es_index_name` | Elasticsearch index name | `wiki` |

---

## Quick Start

After preparing the retriever and datasets:

```bash
python main.py -c config.json
```

You can also run with another config file:

```bash
python main.py -c path_to_config_file
```

---

## Evaluation

Run the evaluation script with:

```bash
python evaluate_.py
```

---

## Citation

