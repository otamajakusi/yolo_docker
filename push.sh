#!/bin/bash

docker login -u otamajakusi
docker tag yolov5_62_export_ncnn:latest otamajakusi/yolov5_62_export_ncnn:latest
docker tag yolov5_62_export_ncnn:latest otamajakusi/ncnn:latest
docker tag yolov5_jetson:latest otamajakusi/yolov5_jetson:latest
docker push otamajakusi/yolov5_62_export_ncnn:latest
docker push otamajakusi/ncnn:latest
docker push otamajakusi/yolov5_jetson:latest
