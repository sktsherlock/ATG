#! /bin/bash

# 获取命令行参数


data_path=$1
name=$2
class_numbers=$3

python data_processing_utils.py --data_path "$data_path" --name "$name" --class_numbers "$class_numbers"