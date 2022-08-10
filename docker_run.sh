docker run --entrypoint bash -it --rm --gpus all --shm-size 2G -v ~/enap_yolov6:/workspace2 \
--mount type=bind,source=$PWD/train_params.json,target=/opt/ml/input/config/hyperparameters.json \
--mount type=bind,source=$PWD/../data.zip,target=/opt/ml/input/data/train/data.zip \
 yolov6:latest
#  