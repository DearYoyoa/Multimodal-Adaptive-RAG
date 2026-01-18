import json
import argparse
from tqdm import tqdm

def calculate_accuracy(file_path, metric):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    correct_count = 0
    total_count = 0
    
    for entry in tqdm(data, desc=f"{metric}accuracy"):
        if metric in entry:
            total_count += 1
            if entry[metric]:
                correct_count += 1
    
    if total_count == 0:
        print(f"warning：no invalid {metric}")
        return 0
    
    accuracy = correct_count / total_count
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="response accuracy")
    parser.add_argument('--input_file', type=str, required=True, help="include answer_in_pred, binomial_recall or genus_recall")
    parser.add_argument('--metric', type=str, required=True, choices=['answer_in_pred', 'binomial_recall', 'genus_recall'], help="select metric：answer_in_pred, binomial_recall 或 genus_recall")
    args = parser.parse_args()

    accuracy = calculate_accuracy(args.input_file, args.metric)
    print(f"{args.metric} ：{accuracy:.2%}")

if __name__ == "__main__":
    main()
