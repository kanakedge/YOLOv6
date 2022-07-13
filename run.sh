echo "Unzipping the dataset at /home/en_yolov6/YOLOv6/Dataset"
unzip -qq /opt/ml/input/data/train/data.zip -d /home/en_yolov6/YOLOv6/
rm -rf /home/en_yolov6/YOLOv6/artifacts/
echo "Datasep Unzipped!"
export TRAIN_PARAMS_LOC=/opt/ml/input/config/hyperparameters.json
export DATASET_ROOT=/home/en_yolov6/YOLOv6/
export DATA_PATH=/home/en_yolov6/YOLOv6/data.yaml
python3 enap_train.py
