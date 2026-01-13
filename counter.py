import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def compare_and_label_files(file1, file2):
    data1 = load_json(file1)
    data2 = load_json(file2)

    groups = {
        (False, False): [],
        (False, True): [],
        (True, False): [],
        (True, True): []
    }

    for item1, item2 in zip(data1, data2):
        # key = (item1['judge_correct'], item2['judge_correct'])
        key = (item1['judge_correct'], item2['judge_correct'])
        group_label = f"{int(key[0])}_{int(key[1])}"
        item1['group'] = group_label
        item2['group'] = group_label
        groups[key].append(item1['image_id'])

    return data1, data2, groups

def print_group_counts(groups):
    for key, value in groups.items():
        print(f"group {key}: {len(value)} samples")

def print_group_ids(groups, group_keys):
    for key in group_keys:
        print(f"\n group {key}  ID:")
        for id in groups[key]:
            print(id)

file1 = '/ossfs/workspace/aml2/aml_ri/fengyi/adaptive-MMRAG/evqa_train_data/evqa_i3_prompt/evqa_train_i3_rir/Qwen25_72B_Instruct_awq_vllm_l20_judged_logs_evqa_train_i3_rir.json'
file2 = '/ossfs/workspace/aml2/aml_ri/fengyi/adaptive-MMRAG/evqa_train_data/evqa_i3_prompt/evqa_train_i3/Qwen25_72B_Instruct_awq_vllm_l20_judged_logs_evqa_train_i3.json'
# file1 = 'data_0921/infoseek_i3/infoseek_train_i3_rir/Qwen25_72B_Instruct_awq_vllm_l20_judged_logs_infoseek_train_i3_rir.json'
# file2 = 'data_0921/infoseek_i3/infoseek_train_i3/Qwen25_72B_Instruct_awq_vllm_l20_judged_logs_infoseek_train_i3.json'

labeled_data1, labeled_data2, groups = compare_and_label_files(file1, file2)


print("numbers of every group:")
print_group_counts(groups)


print("\n any group sample ID: ")
print_group_ids(groups, [(False, True), (True, False), (True, True)])


save_json(labeled_data1, file1)
save_json(labeled_data2, file2)
print(f"\n write into {file1} and {file2}")



'''
qwen evqa_train_data
 (False, False): 770 
 (False, True): 16 
 (True, False): 69 
 (True, True): 145 

qwen data_1017
 (False, False): 2668 
 (False, True): 50 
 (True, False): 168 
 (True, True): 459 

qwen okvqa_train_data 992
 (False, False): 432 
 (False, True): 41 
 (True, False): 78 
 (True, True): 441 

qwen data_okvqa
 (False, False): 2010 
 (False, True): 254 
 (True, False): 454 
 (True, True): 2271 

qwen data_0921 3741
 (False, False): 2917 
 (False, True): 102 
 (True, False): 286 
 (True, True): 436 


i3 data_0921 3741 judge_correct
 (False, False): 2227 
 (False, True): 120 
 (True, False): 220 
 (True, True): 20 


i3 prompt okvqa_train_data judge_correct
 (False, False): 169 
 (False, True): 200 
 (True, False): 274 
 (True, True): 357 


i3 prompt evqa_train_data judge_correct
 (False, False): 785 
 (False, True): 61 
 (True, False): 143 
 (True, True): 11 
'''
