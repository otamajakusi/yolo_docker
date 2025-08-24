#!/bin/bash

docker build -t yolov5_62_export_ncnn -f Dockerfile.yolov5_62_export_ncnn .
docker build -t ncnn -f Dockerfile.ncnn .
docker buildx build --platform linux/arm64 -t yolov5_jetson -f Dockerfile.yolov5_jetson .
docker build -t yolo11_export_ncnn -f Dockerfile.yolo11_export_ncnn .
