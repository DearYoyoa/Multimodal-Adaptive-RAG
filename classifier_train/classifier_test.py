import json
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib  # 用于加载模型
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_classifier(model_path):
    """
    加载逻辑回归分类器模型。
    """
    classifier = joblib.load(model_path)
    return classifier

def query_with_image(classifier, image_hidden_state, original_hidden_state, last_hidden):
    """
    使用逻辑回归分类器进行预测。
    """
    # 只使用第一个图像的表征
    last_hidden = last_hidden[-1]
    original_hidden_state = original_hidden_state[0]
    # print("last_hidden.shape", last_hidden.shape)
    # print("original_hidden_state.shape", original_hidden_state.shape)
    x = torch.cat((torch.tensor(last_hidden), torch.tensor(original_hidden_state)), dim=0).to(DEVICE)
    x = x.reshape(1, -1).cpu().numpy()  # 将特征展平为二维数组
    prediction = classifier.predict(x)[0]
    prediction_score = classifier.predict_proba(x)[0][1]  # 获取正类的概率分数
    return prediction, prediction_score

def main(args):
    # 加载逻辑回归模型
    classifier = load_classifier(args.classifier_path)

    # 创建输出目录
    os.makedirs(f'{args.output_root}/', exist_ok=True)

    # 加载测试数据
    with open('data_okvqa/okvqa_input.json', 'r') as f:
        data = json.load(f)

    samples = [sample for sample in data]
    total_samples = len(samples)
    correct = 0
    logs = []
    true_labels = []
    prediction_scores = []
    predictions = []

    for idx, sample in tqdm(enumerate(samples[args.idx_offset:]), total=len(samples[args.idx_offset:])):
        idx = args.idx_offset + idx

        image_path = f"data_okvqa/okvqa_image/{sample['image_id']}.jpg"
        screenshot_path = f"data_okvqa/screenshot/{sample['image_id']}-search_result.png"
        # last_hidden = np.load(f"hidden_state_i2_test/hidden_okvqa/last_hidden_state/{sample['image_id']}.npy")
        last_hidden = np.load(f"hidden_state_i2_test/hidden_okvqa/exact_answer_features_mlp/exact_answer_last_token/{sample['image_id']}.npy")
        image_hidden = np.load(f"hidden_state_i2_test/hidden_okvqa/hidden_state/{sample['image_id']}.npy")
        original_hidden = np.load(f"hidden_state_i2_test/hidden_okvqa/original_hidden_state/{sample['image_id']}.npy")

        # 使用逻辑回归分类器进行预测
        prediction, prediction_score = query_with_image(classifier, image_hidden, original_hidden, last_hidden)

        # 获取真实标签
        answer_in_pred = None
        with open('data_okvqa/okvqa/okvqa_i2_test/Qwen25_72B_Instruct_awq_vllm_l20_judged_logs_okvqa_i2_test.json', 'r') as f:
            test_data = json.load(f)
            for d in test_data:
                if d["image_id"] == sample["image_id"]:
                    answer_in_pred = d["answer_in_pred"]
                    break

        true_labels.append(1 if answer_in_pred else 0)
        prediction_scores.append(prediction_score)
        predictions.append(prediction)

        if prediction == 1 and answer_in_pred is True:
            correct += 1
        elif prediction == 0 and answer_in_pred is False:
            correct += 1

        log_entry = {
            'idx': idx,
            'data_id': sample['data_id'],
            'image_id': sample['image_id'],
            'question': sample['question'],
            'answer': sample['answer'],
            'answer_eval': sample['answer_eval'],
            'answer_in_pred': answer_in_pred,
            'classifier_prediction': prediction,
        }
        logs.append(log_entry)

    # 计算AUC
    auc_score = roc_auc_score(true_labels, prediction_scores)
    # 计算Precision, Recall, F1
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    correctness = correct / total_samples

    with open(f'{args.output_root}/{args.log_name}_accuracy.json', 'w') as f:
        json.dump({
            'probe_accuracy': correctness,
            'auc_score': auc_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }, f, indent=4)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # basics
    argparser.add_argument('--output_root', type=str, required=True)
    argparser.add_argument('--log_name', type=str, required=True)
    argparser.add_argument('--idx_offset', type=int, required=True)
    argparser.add_argument('--classifier_path', type=str, required=True, help='Path to the trained logistic regression model')

    args = argparser.parse_args()
    main(args)