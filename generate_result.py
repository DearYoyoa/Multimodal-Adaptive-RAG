import os
import json
import pandas as pd


def extract_accuracy_values(file_path):
    """提取accuracy和answer_in_pred_accuracy的值并四舍五入到小数点后三位"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    accuracy = data.get("accuracy", None)
    answer_in_pred_accuracy = data.get("answer_in_pred_accuracy", None)
    
    # 四舍五入保留小数点后三位
    if accuracy is not None:
        accuracy = round(accuracy, 3)
    if answer_in_pred_accuracy is not None:
        answer_in_pred_accuracy = round(answer_in_pred_accuracy, 3)
    
    return accuracy, answer_in_pred_accuracy

def extract_accuracy_values_image(file_path):
    """提取accuracy和answer_in_pred_accuracy的值并四舍五入到小数点后三位"""
    with open(file_path, 'r') as f:
        data = json.load(f)
# probe_accuracy auc_score
    probe_accuracy = data.get("probe_accuracy", None)
    if probe_accuracy is not None:
        probe_accuracy = round(probe_accuracy, 4)
    return probe_accuracy

def process_folders(base_folder):
    """遍历指定文件夹中的所有文件夹并提取需要的数据"""
    results = {}  # 用于存储数据，模型名称为键，每个轮数对应的值为子字典

    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        
        if os.path.isdir(folder_path):  # 确保是文件夹
            # 提取轮数（epoch）
            epoch_number = folder_name.split("epoch_")[-1]  # 提取epoch后的轮数
            
            # 提取文件夹中的两个accuracy结尾的json文件
            accuracy_files = [f for f in os.listdir(folder_path) if f.endswith("accuracy.json")]
            if len(accuracy_files) == 2:  # 确保有两个accuracy文件
                for idx, accuracy_file in enumerate(accuracy_files, start=1):
                    file_path = os.path.join(folder_path, accuracy_file)
                    accuracy, answer_in_pred_accuracy = extract_accuracy_values(file_path)
                    result = f"{accuracy}/{answer_in_pred_accuracy}" if accuracy is not None and answer_in_pred_accuracy is not None else ""
                    
                    # 提取文件名中的后缀 1 或 2
                    middle_number = accuracy_file.split("_epoch_")[-2].split("_")[-1]
                    
                    # 文件名前缀作为纵标，文件夹名去掉epoch后加上1或2
                    model_name = f"{folder_name.split('epoch_')[0]}_{middle_number}"
                    
                    
                    # 如果模型名不存在，初始化子字典
                    if model_name not in results:
                        results[model_name] = {}
                    
                    # 将当前轮数和结果添加到模型对应的子字典中
                    results[model_name][epoch_number] = result

    return results

def process_folders_image(base_folder):
    """遍历指定文件夹中的所有文件夹并提取需要的数据"""
    results = {}  # 用于存储数据，模型名称为键，每个轮数对应的值为子字典

    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        
        if os.path.isdir(folder_path):  # 确保是文件夹
            # 提取轮数（epoch）
            epoch_number = folder_name.split("epoch_")[-1]  # 提取epoch后的轮数
            
            # 提取文件夹中的两个accuracy结尾的json文件
            accuracy_files = [f for f in os.listdir(folder_path) if f.endswith("accuracy.json")]
            if len(accuracy_files) == 1:  # 确保有两个accuracy文件
                for idx, accuracy_file in enumerate(accuracy_files, start=1):
                    file_path = os.path.join(folder_path, accuracy_file)
                    probe_accuracy = extract_accuracy_values(file_path)
                    result = f"{probe_accuracy}" if probe_accuracy is not None else ""
                    
                    # 文件名前缀作为纵标，文件夹名去掉epoch后加上1或2
                    model_name = f"{folder_name.split('_epoch_')[0]}"
                    
                    # 如果模型名不存在，初始化子字典
                    if model_name not in results:
                        results[model_name] = {}
                    
                    # 将当前轮数和结果添加到模型对应的子字典中
                    results[model_name][epoch_number] = result

    return results

def save_to_excel(data, output_path):
    """将提取的数据保存到Excel文件中"""
    # 将数据转换为DataFrame
    all_epochs = sorted({epoch for model_data in data.values() for epoch in model_data.keys()})
    df = pd.DataFrame(columns=["Model"] + all_epochs)

    for model_name, epochs_data in data.items():
        row = {"Model": model_name}
        for epoch in all_epochs:
            row[epoch] = epochs_data.get(epoch, "")  # 如果某轮无数据，填空
        df = df._append(row, ignore_index=True)
    
    # 保存为Excel文件
    # df.to_excel(output_path, index=False)
    df.to_csv(output_path, index=False, encoding='utf-8')

def main():
    base_folder = "data_1017/evqa_qwen_test_2"  # 修改为您的文件夹路径
    output_path = "results_evqa_qwen.csv"  # 输出的Excel文件路径

    # 提取数据
    data = process_folders(base_folder)
    
    # 保存到Excel
    save_to_excel(data, output_path)
    print(f"Results have been saved to {output_path}")

if __name__ == "__main__":
    main()
