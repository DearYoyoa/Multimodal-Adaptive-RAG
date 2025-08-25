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
        print(f"组 {key}: {len(value)} 个样本")

def print_group_ids(groups, group_keys):
    for key in group_keys:
        print(f"\n组 {key} 的ID:")
        for id in groups[key]:
            print(id)

# 文件路径
file1 = '/ossfs/workspace/aml2/aml_ri/fengyi/adaptive-MMRAG/evqa_train_data/evqa_i3_prompt/evqa_train_i3_rir/Qwen25_72B_Instruct_awq_vllm_l20_judged_logs_evqa_train_i3_rir.json'
file2 = '/ossfs/workspace/aml2/aml_ri/fengyi/adaptive-MMRAG/evqa_train_data/evqa_i3_prompt/evqa_train_i3/Qwen25_72B_Instruct_awq_vllm_l20_judged_logs_evqa_train_i3.json'
# file1 = 'data_0921/infoseek_i3/infoseek_train_i3_rir/Qwen25_72B_Instruct_awq_vllm_l20_judged_logs_infoseek_train_i3_rir.json'
# file2 = 'data_0921/infoseek_i3/infoseek_train_i3/Qwen25_72B_Instruct_awq_vllm_l20_judged_logs_infoseek_train_i3.json'

# 比较文件、标注组别并分组
labeled_data1, labeled_data2, groups = compare_and_label_files(file1, file2)

# 打印每组的样本数量
print("每组的样本数量：")
print_group_counts(groups)

# 打印组2、组3和组4的样本ID
print("\n特定组的样本ID: ")
print_group_ids(groups, [(False, True), (True, False), (True, True)])

# 将标注后的数据保存回原始文件
save_json(labeled_data1, file1)
save_json(labeled_data2, file2)
print(f"\n标注后的数据已保存回原始文件 {file1} 和 {file2}")



'''
qwen evqa_train_data
组 (False, False): 770 个样本
组 (False, True): 16 个样本
组 (True, False): 69 个样本
组 (True, True): 145 个样本

qwen data_1017
组 (False, False): 2668 个样本
组 (False, True): 50 个样本
组 (True, False): 168 个样本
组 (True, True): 459 个样本

qwen okvqa_train_data 992
组 (False, False): 432 个样本
组 (False, True): 41 个样本
组 (True, False): 78 个样本
组 (True, True): 441 个样本

qwen data_okvqa
组 (False, False): 2010 个样本
组 (False, True): 254 个样本
组 (True, False): 454 个样本
组 (True, True): 2271 个样本

qwen data_0921 3741
组 (False, False): 2917 个样本
组 (False, True): 102 个样本
组 (True, False): 286 个样本
组 (True, True): 436 个样本


i3 data_0921 3741 judge_correct
组 (False, False): 2227 个样本
组 (False, True): 120 个样本
组 (True, False): 220 个样本
组 (True, True): 20 个样本


i3 prompt okvqa_train_data judge_correct
每组的样本数量：
组 (False, False): 169 个样本
组 (False, True): 200 个样本
组 (True, False): 274 个样本
组 (True, True): 357 个样本


i3 prompt evqa_train_data judge_correct
每组的样本数量：
组 (False, False): 785 个样本
组 (False, True): 61 个样本
组 (True, False): 143 个样本
组 (True, True): 11 个样本
'''