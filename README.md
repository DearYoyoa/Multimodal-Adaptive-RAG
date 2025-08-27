# Multimodal-Adaptive-RAG

This repository provides the implementation of **Multimodal Adaptive Retrieval-Augmented Generation through Internal Representation Learning**. The project investigates how internal hidden representations can be leveraged to decide whether retrieval is necessary in multimodal question answering (VQA) tasks.

## ⚙️ Setup

### Environment
We recommend using **Python 3.10+** with Conda or virtualenv:

```
conda create -n mmarag python=3.10
conda activate mmarag
pip install -r requirements.txt
```

## 📊 Datasets
Each dataset contains images and annotations, download and prepare three [datasets](https://drive.google.com/file/d/1nY8O6yKrOwVwHgy3cooBiZ5-oWdGMe4-/view?usp=drive_link). 

Obtain the google search screenshot corresponding to the image:
```
python rir_api.py 
```
3. Use the extract.py file to extract visual features and text hidden states to generate responses simultaneously and determine whether the answers are true or false. 
4. Run counter.py to set labels for the samples, including four scenarios.
(true, true), (true, false), (false, true), (false, false)

## 🚀 Train
### 1. The classifier predicts whether retrieval should be applied for a given query.
python classifier_token_probe_okvqa.py

## 🚀 Evaluate
Test using run_okvqa_i2_cls2_test.py
