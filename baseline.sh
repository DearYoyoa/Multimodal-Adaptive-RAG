#!/usr/bin/env bash
set -e

# useful system prompt


SYS_MSG="experiment/okvqa/query_system_no_screenshot.jinja2"

# OKVQA
python -m clip_baseline \
  --data_path data/data_okvqa \
  --output_root data/data_okvqa/clip_baseline_new \
  --log_name clip_okvqa_qwen2vl_final \
  --sys_msg_filename ${SYS_MSG} \
  --idx_offset 0

# EVQA / data_1017
python -m clip_baseline \
  --data_path data/data_1017 \
  --output_root data/data_1017/clip_baseline_new \
  --log_name clip_evqa_qwen2vl_final \
  --sys_msg_filename ${SYS_MSG} \
  --idx_offset 0

# InfoSeek / local_data
python -m clip_baseline_infoseek \
  --data_path data/local_data \
  --output_root data/local_data/clip_baseline_new \
  --log_name clip_infoseek_qwen2vl_final \
  --sys_msg_filename ${SYS_MSG} \
  --idx_offset 0

# SYS_MSG="experiment/okvqa/query_system_no_screenshot.jinja2"

# OKVQA
# python -m p_true_baseline \
#   --data_path data/data_okvqa \
#   --output_root data/data_okvqa/p_true_baseline_new \
#   --log_name p_true_okvqa_qwen2vl \
#   --sys_msg_filename ${SYS_MSG} \
#   --idx_offset 0

# EVQA / data_1017
# python -m p_true_baseline \
#   --data_path data/data_1017 \
#   --output_root data/data_1017/p_true_baseline_new \
#   --log_name p_true_evqa_qwen2vl \
#   --sys_msg_filename ${SYS_MSG} \
#   --idx_offset 0

# # InfoSeek / local_data
# python -m p_true_baseline_infoseek \
#   --data_path data/local_data \
#   --output_root data/local_data/p_true_baseline_new \
#   --log_name p_true_infoseek_qwen2vl \
#   --sys_msg_filename ${SYS_MSG} \
#   --idx_offset 0


# SYS_MSG="experiment/okvqa/query_system_cot_screenshot.jinja2"

# # OKVQA
# python -m cot_baseline_qwen2 \
#   --data_path data/data_okvqa \
#   --output_root data/data_okvqa/cot_baseline_new \
#   --log_name cot_okvqa_qwen2vl \
#   --sys_msg_filename ${SYS_MSG} \
#   --idx_offset 0

# # EVQA / data_1017
# python -m cot_baseline_qwen2 \
#   --data_path data/data_1017 \
#   --output_root data/data_1017/cot_baseline_new \
#   --log_name cot_evqa_qwen2vl \
#   --sys_msg_filename ${SYS_MSG} \
#   --idx_offset 0

# # InfoSeek / local_data
# python -m cot_baseline_infoseek \
#   --data_path data/local_data \
#   --output_root data/local_data/cot_baseline_new \
#   --log_name cot_infoseek_qwen2vl \
#   --sys_msg_filename ${SYS_MSG} \
#   --idx_offset 0
