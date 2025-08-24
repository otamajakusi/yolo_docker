#!/bin/bash

if [ $# -ne 1 ];then
  echo "usage: $0 weigth.pt"
  exit 1
fi

weight=$1
full_path=$(realpath $weight)
weight_name=$(basename $full_path)
weight_dir=$(dirname $full_path)

docker run -it -v $weight_dir:/data yolo11_export_ncnn python3 yolo11_export_ncnn.py --model /data/$weight_name
