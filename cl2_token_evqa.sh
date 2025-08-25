# #!/bin/bash
# BASE_PATH="models/token_probe_infoseek_qwen"
# DATA_DIR="adaptive"
# OUTPUT_DIR="local_data/infoseek_qwen"

# # 运行自适应检索实验
# echo "Running Adaptive Retrieval Experiment..."

# # Define the layers, epochs, and model types
# layers=(-1)
# epochs=(5 6 7 8 9 10 11 12)
# # epochs=(9 10 11 12 13 14 15 16)
# model_types=("original_image_wo_rir")
# # image_rir 24
# # model_types=("original_image_wo_rir" "image_wo_rir" "original_image_rir" "image_rir")
# # model_types=("image_wo_rir" "original_image_wo_rir")
# # Loop through the specified layers, epochs, and model types
# for layer in "${layers[@]}"; do
#     echo "Processing layer: $layer"
#     for model_type in "${model_types[@]}"; do
#         echo "Using model type: $model_type"
#         for epoch in "${epochs[@]}"; do
#             echo "Running epoch: $epoch"
#             # Construct the classifier path for the current configuration
#             CLASSIFIER_PATH="${BASE_PATH}/qwen_dp3_multiclass_layer_${layer}_${model_type}/qwen_dp3_multiclass_layer_${layer}_${model_type}_epoch_${epoch}.pth"
            
#             EXP_NAME="infoseek_qwen_probe_token_layer_${layer}_${model_type}_epoch_${epoch}"
#             echo "Experiment name: $EXP_NAME"
            
#             # Calculate GPU index (0, 1, 2, 3) based on epoch
#             GPU_INDEX=$((epoch % 8))
            
#             CUDA_VISIBLE_DEVICES=$GPU_INDEX python -m experiment.infoseek.run_infoseek_qwen_cls2_test \
#                     --output_root ${OUTPUT_DIR}/${EXP_NAME} \
#                     --log_name logs_${EXP_NAME} \
#                     --sys_msg_filename_with_screenshot query_system_with_screenshot.jinja2 \
#                     --sys_msg_filename_without_screenshot query_system_no_screenshot.jinja2 \
#                     --idx_offset 0 \
#                     --classifier_path ${CLASSIFIER_PATH} &
#         done
#         wait 
#         # for epoch in "${epochs[@]}"; do
#         #     EXP_NAME="evqa_qwen_adaptive_token_cls2_layer_${layer}_${model_type}_epoch_${epoch}"
#         #     echo "Judging experiment: $EXP_NAME"
            
#         #     # Calculate GPU index (0, 1, 2, 3) based on epoch
#         #     GPU_INDEX=$((epoch % 4))
            
#         #     CUDA_VISIBLE_DEVICES=$GPU_INDEX python -m experiment.infoseek.judge_qwen \
#         #         --output_root ${OUTPUT_DIR}/${EXP_NAME} \
#         #         --exp_name logs_${EXP_NAME}_epoch_${epoch} &
#         # done
#         # wait 
#     done
# done

# !/bin/bash

# 定义基础文件夹路径
BASE_FOLDER="data_1017/evqa_i2_ablation_text"  # 要遍历的文件夹路径
OUTPUT_ROOT="data_1017/evqa_i2_ablation_text"  # 输出根路径

# 遍历 BASE_FOLDER 下的每个文件夹
for folder in "$BASE_FOLDER"/*; do
  if [ -d "$folder" ]; then
    # 遍历文件夹中的每个 .json 文件
    for json_file in "$folder"/*.json; do # 8 16的9~16 original_image_wo_rir
      if [[ "$(basename "$json_file")" == *"evqa_i2_probe_token_layer_-1_"* ]]; then
        if [ -f "$json_file" ]; then
          # 获取文件名去除扩展名，作为 EXP_NAME
          EXP_NAME=$(basename "$json_file" .json)
          echo "Processing EXP_NAME: $EXP_NAME"
          
          # 生成命令，不添加 logs_ 前缀
          COMMAND="python -m experiment.evqa.judge_qwen --output_root $folder --exp_name $EXP_NAME"
          echo "Generated command: $COMMAND"
          
          # 执行命令
          $COMMAND
        fi
      fi
    done
  fi
done
