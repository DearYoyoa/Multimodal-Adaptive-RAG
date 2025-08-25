import json
import argparse
from tqdm import tqdm

def calculate_accuracy(file_path, metric):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    correct_count = 0
    total_count = 0
    
    for entry in tqdm(data, desc=f"计算{metric}准确率"):
        if metric in entry:
            total_count += 1
            if entry[metric]:
                correct_count += 1
    
    if total_count == 0:
        print(f"警告：没有找到有效的 {metric} 字段")
        return 0
    
    accuracy = correct_count / total_count
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="计算回答准确率")
    parser.add_argument('--input_file', type=str, required=True, help="包含 answer_in_pred, binomial_recall 或 genus_recall 字段的JSON文件路径")
    parser.add_argument('--metric', type=str, required=True, choices=['answer_in_pred', 'binomial_recall', 'genus_recall'], help="选择计算的指标：answer_in_pred, binomial_recall 或 genus_recall")
    args = parser.parse_args()

    accuracy = calculate_accuracy(args.input_file, args.metric)
    print(f"{args.metric} 准确率：{accuracy:.2%}")

if __name__ == "__main__":
    main()