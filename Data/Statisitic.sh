#!/bin/bash

csv_file=$1  # 获取命令行中的第一个参数作为csv_file
column_name=$2  # 获取命令行中的第二个参数作为column_name

python TextStatisitic.py --csv_file "$csv_file" --column_name "$column_name"