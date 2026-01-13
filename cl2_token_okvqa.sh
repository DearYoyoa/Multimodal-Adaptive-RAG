#!/bin/bash
BASE_PATH="models/token_probe_evqa_i2_ablation_text"
DATA_DIR="adaptive"
OUTPUT_DIR="data_1017/evqa_i2_ablation_text"

echo "Running Adaptive Retrieval Experiment..."

# Define the layers, epochs, and model types
layers=(-1)
# epochs=(3)
epochs=(3 4 5 6 7 8 9 10)
model_types=("text_rir" "text_wo_rir")

# Loop through the specified layers, epochs, and model types
for layer in "${layers[@]}"; do
    echo "Processing layer: $layer"
    for model_type in "${model_types[@]}"; do
        echo "Using model type: $model_type"
        for epoch in "${epochs[@]}"; do
            echo "Running epoch: $epoch"
            # Construct the classifier path for the current configuration
            # CLASSIFIER_PATH="${BASE_PATH}/qwen_dp3_multiclass_layer_${layer}_${model_type}/qwen_dp3_multiclass_layer_${layer}_${model_type}_epoch_${epoch}.pth"
            CLASSIFIER_PATH="${BASE_PATH}/qwen_dp3_multiclass_${model_type}/qwen_dp3_multiclass_${model_type}_epoch_${epoch}.pth"
            EXP_NAME="evqa_i2_probe_token_layer_${layer}_${model_type}_epoch_${epoch}"
            echo "Experiment name: $EXP_NAME"
            
            # Calculate GPU index (0, 1, 2, 3) based on epoch
            GPU_INDEX=$((epoch % 8))
            
            CUDA_VISIBLE_DEVICES=$GPU_INDEX python -m experiment.evqa.run_evqa_i2_cls2_test \
                    --output_root ${OUTPUT_DIR}/${EXP_NAME} \
                    --log_name logs_${EXP_NAME} \
                    --sys_msg_filename_with_screenshot query_system_with_screenshot.jinja2 \
                    --sys_msg_filename_without_screenshot query_system_no_screenshot.jinja2 \
                    --idx_offset 0 \
                    --classifier_path ${CLASSIFIER_PATH} &
        done
        wait 
        # for epoch in "${epochs[@]}"; do
        #     EXP_NAME="evqa_qwen_adaptive_token_cls2_layer_${layer}_${model_type}_epoch_${epoch}"
        #     echo "Judging experiment: $EXP_NAME"
            
        #     # Calculate GPU index (0, 1, 2, 3) based on epoch
        #     GPU_INDEX=$((epoch % 4))
            
        #     CUDA_VISIBLE_DEVICES=$GPU_INDEX python -m experiment.infoseek.judge_qwen \
        #         --output_root ${OUTPUT_DIR}/${EXP_NAME} \
        #         --exp_name logs_${EXP_NAME}_epoch_${epoch} &
        # done
        # wait 
    done
done
