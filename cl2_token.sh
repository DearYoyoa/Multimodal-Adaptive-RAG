#!/bin/bash
BASE_PATH="models/token_probe_infoseek_i3"
DATA_DIR="adaptive"
OUTPUT_DIR="local_data/infoseek_i3"

# 运行自适应检索实验
echo "Running Adaptive Retrieval Experiment..."

# Define the layers, epochs, and model types
layers=(0 8 16 24 -1)
epochs=(1 2 3 4 5 6 7 8)
model_types=("original_image_wo_rir" "image_wo_rir" "original_image_rir" "image_rir")
# model_types=("image_rir")
# Loop through the specified layers, epochs, and model types
for layer in "${layers[@]}"; do
    echo "Processing layer: $layer"
    for model_type in "${model_types[@]}"; do
        echo "Using model type: $model_type"
        for epoch in "${epochs[@]}"; do
            echo "Running epoch: $epoch"
            # Construct the classifier path for the current configuration
            CLASSIFIER_PATH="${BASE_PATH}/qwen_dp3_multiclass_layer_${layer}_${model_type}/qwen_dp3_multiclass_layer_${layer}_${model_type}_epoch_${epoch}.pth"
            
            EXP_NAME="infoseek_i3_adaptive_token_cls2_layer_${layer}_${model_type}_epoch_${epoch}"
            echo "Experiment name: $EXP_NAME"
            
            # Calculate GPU index (0, 1, 2, 3) based on epoch
            GPU_INDEX=$((epoch % 8))
            
            CUDA_VISIBLE_DEVICES=$GPU_INDEX python -m experiment.infoseek.run_sample_i3_cls2_test \
                    --output_root ${OUTPUT_DIR}/${EXP_NAME} \
                    --log_name logs_${EXP_NAME} \
                    --sys_msg_filename_with_screenshot query_system_with_screenshot.jinja2 \
                    --sys_msg_filename_without_screenshot query_system_no_screenshot.jinja2 \
                    --idx_offset 0 \
                    --classifier_path ${CLASSIFIER_PATH} &
        done
        wait 
#         for epoch in "${epochs[@]}"; do
#             EXP_NAME="infoseek_i2_adaptive_token_cls2_layer_${layer}_${model_type}_epoch_${epoch}"
#             echo "Judging experiment: $EXP_NAME"
            
#             # Calculate GPU index (0, 1, 2, 3) based on epoch
#             GPU_INDEX=$((epoch % 4))
            
#             CUDA_VISIBLE_DEVICES=$GPU_INDEX python -m experiment.infoseek.judge_qwen \
#                 --output_root ${OUTPUT_DIR}/${EXP_NAME} \
#                 --exp_name logs_${EXP_NAME}_1_epoch_${epoch} &
#             CUDA_VISIBLE_DEVICES=$GPU_INDEX python -m experiment.infoseek.judge_qwen \
#                 --output_root ${OUTPUT_DIR}/${EXP_NAME} \
#                 --exp_name logs_${EXP_NAME}_2_epoch_${epoch} &
#             wait
#         done
    done
done