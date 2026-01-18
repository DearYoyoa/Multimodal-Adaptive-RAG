import os
import json
import pandas as pd


def extract_accuracy_values(file_path):

    with open(file_path, 'r') as f:
        data = json.load(f)
    accuracy = data.get("accuracy", None)
    answer_in_pred_accuracy = data.get("answer_in_pred_accuracy", None)

    if accuracy is not None:
        accuracy = round(accuracy, 3)
    if answer_in_pred_accuracy is not None:
        answer_in_pred_accuracy = round(answer_in_pred_accuracy, 3)
    
    return accuracy, answer_in_pred_accuracy

def extract_accuracy_values_image(file_path):

    with open(file_path, 'r') as f:
        data = json.load(f)
# probe_accuracy auc_score
    probe_accuracy = data.get("probe_accuracy", None)
    if probe_accuracy is not None:
        probe_accuracy = round(probe_accuracy, 4)
    return probe_accuracy

def process_folders(base_folder):

    results = {}  

    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        
        if os.path.isdir(folder_path):  

            epoch_number = folder_name.split("epoch_")[-1] 

            accuracy_files = [f for f in os.listdir(folder_path) if f.endswith("accuracy.json")]
            if len(accuracy_files) == 2:
                for idx, accuracy_file in enumerate(accuracy_files, start=1):
                    file_path = os.path.join(folder_path, accuracy_file)
                    accuracy, answer_in_pred_accuracy = extract_accuracy_values(file_path)
                    result = f"{accuracy}/{answer_in_pred_accuracy}" if accuracy is not None and answer_in_pred_accuracy is not None else ""

                    middle_number = accuracy_file.split("_epoch_")[-2].split("_")[-1]

                    model_name = f"{folder_name.split('epoch_')[0]}_{middle_number}"

                    if model_name not in results:
                        results[model_name] = {}
                    results[model_name][epoch_number] = result

    return results

def process_folders_image(base_folder):

    results = {}  

    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        
        if os.path.isdir(folder_path): 

            epoch_number = folder_name.split("epoch_")[-1]  
            
            accuracy_files = [f for f in os.listdir(folder_path) if f.endswith("accuracy.json")]
            if len(accuracy_files) == 1:  
                for idx, accuracy_file in enumerate(accuracy_files, start=1):
                    file_path = os.path.join(folder_path, accuracy_file)
                    probe_accuracy = extract_accuracy_values(file_path)
                    result = f"{probe_accuracy}" if probe_accuracy is not None else ""

                    model_name = f"{folder_name.split('_epoch_')[0]}"

                    if model_name not in results:
                        results[model_name] = {}
                    results[model_name][epoch_number] = result

    return results

def save_to_excel(data, output_path):

    all_epochs = sorted({epoch for model_data in data.values() for epoch in model_data.keys()})
    df = pd.DataFrame(columns=["Model"] + all_epochs)

    for model_name, epochs_data in data.items():
        row = {"Model": model_name}
        for epoch in all_epochs:
            row[epoch] = epochs_data.get(epoch, "") 
        df = df._append(row, ignore_index=True)
    

    # df.to_excel(output_path, index=False)
    df.to_csv(output_path, index=False, encoding='utf-8')

def main():
    base_folder = "data_1017/evqa_qwen_test_2"
    output_path = "results_evqa_qwen.csv"
    data = process_folders(base_folder)
    save_to_excel(data, output_path)
    print(f"Results have been saved to {output_path}")

if __name__ == "__main__":
    main()
