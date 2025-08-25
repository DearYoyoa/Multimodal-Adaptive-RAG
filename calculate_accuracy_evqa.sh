#!/bin/bash

# 定义基础文件夹路径
BASE_FOLDER="data_okvqa/okvqa_qwen_test"  # 要遍历的文件夹路径
OUTPUT_ROOT="data_okvqa/okvqa_qwen_test"  # 输出根路径

# 遍历 BASE_FOLDER 下的每个文件夹
for folder in "$BASE_FOLDER"/*; do
  if [ -d "$folder" ]; then
    # 遍历文件夹中的每个 .json 文件
    for json_file in "$folder"/*.json; do
      if [[ "$(basename "$json_file")" == *"qwen_probe_token_layer_8"* ]]; then
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
