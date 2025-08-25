# Multimodal-Adaptive-RAG
Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning

# Multimodal Adaptive Retrieval-Augmented Generation (MM-ARAG)

This repository provides the implementation of **Multimodal Adaptive Retrieval-Augmented Generation through Internal Representation Learning**.  
The project investigates how **internal hidden representations** can be leveraged to decide whether retrieval is necessary in multimodal question answering (VQA) tasks.

We conduct experiments with **three different backbone models** across **three benchmark datasets**:
- **OK-VQA**
- **InfoSeek**
- **E-VQA**

---

## 📂 Repository Structure
├── classifier/ # Classifier training code (hidden state + image features → prediction)
├── okvqa/ # OK-VQA dataset preprocessing & evaluation scripts
├── infoseek/ # InfoSeek dataset preprocessing & evaluation scripts
├── evqa/ # E-VQA dataset preprocessing & evaluation scripts
├── scripts/ # Bash scripts to reproduce experiments
├── requirements.txt # Python dependencies
└── README.md # Project documentation
