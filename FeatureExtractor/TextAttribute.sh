#!/bin/bash

csv_file=$1
path=$2
max_length=$3
text_column=$4
name=$5
gpu_ids=$6

# 定义要遍历的 model_names
model_names=("bert-large-uncased" "roberta-large" "prajjwal1/bert-tiny" "prajjwal1/bert-mini" "bert-base-uncased" "roberta-base" )

# 循环遍历不同的 model_name
for model_name in "${model_names[@]}"
do
  # 设置不同 model_name 对应的 batch_size
  if [[ "$model_name" == "bert-large-uncased" || "$model_name" == "roberta-large" ]]; then
    batch_size=500
  else
    batch_size=1000
  fi

  CUDA_VISIBLE_DEVICES="$gpu_ids" python TextAttributeExtract.py \
  --csv_file "$csv_file" \
  --model_name "$model_name" \
  --name "$name" \
  --path "$path" \
  --max_length "$max_length" \
  --batch_size "$batch_size" \
  --cls \
  --text_column "$text_column"
done