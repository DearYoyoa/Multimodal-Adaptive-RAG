# #!/bin/bash

# # 定义要使用的epoch列表
# EPOCHS=(2 4 6 8 10 15 30)

# # 定义JSON文件路径列表
# json_pathes=("local_data/snake/snake_i2_adaptive_cls2"
# "local_data/snake/snake_i2_adaptive_rir")

# # 遍历每个JSON文件路径
# for json_base_path in ${json_pathes[@]}; do
#     # 遍历每个epoch
#     for EPOCH in "${EPOCHS[@]}"; do
#         # exit when error
#         set -e
#         # 构建完整的JSON文件路径
#         if [[ $json_base_path == *"cls2"* ]]; then
#             # 对于包含"cls2"的路径，处理两个日志文件
#             for LOG_NUM in 1 2; do
#                 json_path="${json_base_path}_epoch_${EPOCH}/logs_${json_base_path##*/}_epoch_${EPOCH}_${LOG_NUM}_epoch_${EPOCH}.json"
#                 echo "Calculating accuracy for ${json_path}"
#                 python calculate_accuracy.py --input_file ${json_path} --metric binomial_recall
#                 python calculate_accuracy.py --input_file ${json_path} --metric genus_recall
#             done
#         else
#             # 对于不包含"cls2"的路径，处理单个日志文件
#             json_path="${json_base_path}_epoch_${EPOCH}/logs_${json_base_path##*/}_epoch_${EPOCH}.json"
#             echo "Calculating accuracy for ${json_path}"
#             python calculate_accuracy.py --input_file ${json_path} --metric binomial_recall
#             python calculate_accuracy.py --input_file ${json_path} --metric genus_recall
#         fi
#     done
# done


# 遍历results/evqa文件夹下的每个json，根据answer_in_pred 计算准确率
for json_path in results/okvqa/*.json; do
    echo "Calculating accuracy for ${json_path}"
    python calculate_accuracy.py --input_file ${json_path} --metric answer_in_pred
done
