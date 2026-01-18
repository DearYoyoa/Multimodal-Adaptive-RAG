#!/bin/bash

export PYTHONPATH="$PYTHONPATH:path/to/your/project"
CLASSIFIER_PATH="models/classifier1/classifier1_epoch_3.pth"
DATA_DIR="adaptive"
OUTPUT_DIR="local_data/evqa"

echo "Running Adaptive Retrieval Experiment for EVQA..."
EXP_NAME=evqa_i2_adaptive_rir
CUDA_VISIBLE_DEVICES=5 python -m experiment.evqa.run_evqa_adaptive \
    --output_root ${OUTPUT_DIR}/${EXP_NAME} \
    --log_name logs_${EXP_NAME} \
    --sys_msg_filename_with_screenshot query_system_with_screenshot.jinja2 \
    --sys_msg_filename_without_screenshot query_system_no_screenshot.jinja2 \
    --idx_offset 0 \
    --data_source local \
    --classifier_path ${CLASSIFIER_PATH} \
    --use_adaptive 
