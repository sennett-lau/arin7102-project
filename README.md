# ARIN7102 Project

## Table of Contents

- [Description](#description)
- [Installation](#installation)

## Description

Product analysis for pharmacy store.

## Installation

Setup a virtual environment with [Python 3.10](https://www.python.org/downloads/):
```bash
# sample command with conda virtual environment creation
conda create -n arin7102-gp python=3.10
conda activate arin7102-gp
```

Install packages with commands:
```bash
pip install notebook jupyterlab pandas matplotlib seaborn scikit-learn pandas numpy torch transformers scikit-learn tqdm wordcloud nltk tf-keras torch sentence-transformers gensim sentence_transformers
```

## Datasets

- [WebMD Drug Reviews Dataset](https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset)
- [Social media + drug review classification](https://drive.google.com/file/d/1bLmyuNYcBjDcEMkcWiFXQQ_3Vj_AZ9tO/view?usp=sharing)

### Downloading the datasets and weights

1. Download the datasets from given links from Kaggle.
2. Decompress the file and place in the `src/data` folder.
