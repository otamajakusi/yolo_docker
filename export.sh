#!/bin/bash

if [ $# -ne 1 ];then
  echo "usage: $0 weigth.pt"
  exit 1
fi

weight=$1
full_path=$(realpath $weight)
weight_name=$(basename $full_path)
weight_dir=$(dirname $full_path)
weight_stem="${weight_name%.*}"

docker run -it --rm -v $weight_dir:/weight yolov5_62_export_ncnn python3 export.py --weights /weight/$weight_name --img 640 --batch 1 --include onnx torchscript --simplify
docker run -it --rm -v $weight_dir:/weight ncnn ./build/tools/onnx/onnx2ncnn /weight/$weight_name /weight/$weight_stem.param /weight/$weight_stem.bin

