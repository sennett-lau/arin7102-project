# ARIN7102 Project

## Table of Contents

- [Description](#description)
- [Installation](#installation)

## Description

Product analysis for pharmacy store, with a Vite based frontend so serve with chatbot and dashboard feature, Flask based backend as LLM proxy while process data locally with python.

## Installation

### Application

Copy the `flask-api/.env.example` as `flask-api/.env`, please your openai api key to `OPENAI_API_KEY`.

Currently the frontend is using MOCK mode, control it with the `VITE_IS_MOCK` at `/vite-site/.env`, set it to `false` to begin the chat with the api.

Make sure you have installed Docker and Docker Compose, the simply run:

```
docker-compose up -d --build
```

Project will then start for
- Vite on http://localhost:7100
- Flask on http://localhost:7101

### For local data analysis

Setup a virtual environment with [Python 3.10](https://www.python.org/downloads/):
```bash
# sample command with conda virtual environment creation
conda create -n arin7102-gp python=3.10
conda activate arin7102-gp
```

Install packages with commands:
```bash
pip install notebook jupyterlab pandas matplotlib seaborn scikit-learn pandas numpy torch transformers scikit-learn tqdm wordcloud nltk tf-keras torch sentence-transformers gensim sentence_transformers textblob
```

## Datasets

- [WebMD Drug Reviews Dataset](https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset)
- [Social media + drug review classification](https://drive.google.com/file/d/1bLmyuNYcBjDcEMkcWiFXQQ_3Vj_AZ9tO/view?usp=sharing)

### Downloading the datasets and weights

1. Download the datasets from given links from Kaggle.
2. Decompress the file and place in the `src/data` folder.
