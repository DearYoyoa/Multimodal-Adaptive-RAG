# Multimodal-Adaptive-RAG

This repository provides the implementation of **Multimodal Adaptive Retrieval-Augmented Generation through Internal Representation Learning**. The project investigates how **internal hidden representations** can be leveraged to decide whether retrieval is necessary in multimodal question answering (VQA) tasks.

---

## ⚙️ Setup

### 1. Environmen
We recommend using **Python 3.10+** with Conda or virtualenv:

```bash
conda create -n mmarag python=3.10
conda activate mmarag
pip install -r requirements.txt

### 2. Datasets
Download and prepare datasets:

OK-VQA: https://okvqa.allenai.org/

InfoSeek: https://github.com/MMRAG/infoseek

E-VQA: https://evqa.org/
