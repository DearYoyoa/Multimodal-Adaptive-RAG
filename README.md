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
Extract visual features and text hidden states using screenshot and without screenshot to generate responses simultaneously and determine whether the answer is true or false. We can adjust the pooling method of image feature patches and the specific positions of the hidden states by themselves.
```
python run_okvqa_i2_text_cls2_extract.py
```
Run counter.py to set labels (rir, wo rir) for the samples, including four scenarios.
```
(true, true), (true, false), (false, true), (false, false)
```

## 🚀 Train
Train the classifier to predict whether retrieval should be applied to a given query and image. The number of visual feature layers used for training and the method of feature fusion can be specified.
```
python classifier_token_probe_okvqa.py
```
Two four-classifiers can be trained. The RIR-Optimistic Strategy calls RIR for all cases except (false, true), while the RIR-Pessimistic Strategy does not call RIR for all cases except (true, false). Or, the binary classifier can be directly trained.
## 🚀 Evaluate
Test using the trained classifier. Finally, make the judgment using exact matching or by using the Qwen2.5VL-72b model.
```
bash cl2_token_okvqa.sh
```
