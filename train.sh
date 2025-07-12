# docker run -it \
#     -v /usr/local/cuda-12.2:/usr/local/cuda-12.2:ro \
#     -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro \
#     -v $(realpath path/to/dataset):/dataset \
#     -v $(realpath output):/output \
#     -e LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64" \
#     --runtime nvidia \
#     --rm otamajakusi/yolov5_jetson bash

# source venv/bin/activate
# python train.py --data coco128.yaml --cfg models/yolov5s.yaml --weight yolov5s.pt --batch-size 8 --project /output
