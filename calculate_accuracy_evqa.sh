#!/bin/bash

BASE_FOLDER="data_okvqa/okvqa_qwen_test" 
OUTPUT_ROOT="data_okvqa/okvqa_qwen_test"

for folder in "$BASE_FOLDER"/*; do
  if [ -d "$folder" ]; then

    for json_file in "$folder"/*.json; do
      if [[ "$(basename "$json_file")" == *"qwen_probe_token_layer_8"* ]]; then
        if [ -f "$json_file" ]; then

          EXP_NAME=$(basename "$json_file" .json)
          echo "Processing EXP_NAME: $EXP_NAME"

          COMMAND="python -m experiment.evqa.judge_qwen --output_root $folder --exp_name $EXP_NAME"
          echo "Generated command: $COMMAND"

          $COMMAND
        fi
      fi
    done
  fi
done
