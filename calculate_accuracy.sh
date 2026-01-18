# #!/bin/bash

# EPOCHS=(2 4 6 8 10 15 30)

# json_pathes=("local_data/snake/snake_i2_adaptive_cls2"
# "local_data/snake/snake_i2_adaptive_rir")

# for json_base_path in ${json_pathes[@]}; do

#     for EPOCH in "${EPOCHS[@]}"; do
#         # exit when error
#         set -e

#         if [[ $json_base_path == *"cls2"* ]]; then

#             for LOG_NUM in 1 2; do
#                 json_path="${json_base_path}_epoch_${EPOCH}/logs_${json_base_path##*/}_epoch_${EPOCH}_${LOG_NUM}_epoch_${EPOCH}.json"
#                 echo "Calculating accuracy for ${json_path}"
#                 python calculate_accuracy.py --input_file ${json_path} --metric binomial_recall
#                 python calculate_accuracy.py --input_file ${json_path} --metric genus_recall
#             done
#         else

#             json_path="${json_base_path}_epoch_${EPOCH}/logs_${json_base_path##*/}_epoch_${EPOCH}.json"
#             echo "Calculating accuracy for ${json_path}"
#             python calculate_accuracy.py --input_file ${json_path} --metric binomial_recall
#             python calculate_accuracy.py --input_file ${json_path} --metric genus_recall
#         fi
#     done
# done

for json_path in results/okvqa/*.json; do
    echo "Calculating accuracy for ${json_path}"
    python calculate_accuracy.py --input_file ${json_path} --metric answer_in_pred
done
