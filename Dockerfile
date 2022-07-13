# YOLOv6 by https://github.com/meituan/YOLOv6

# Start FROM NVIDIA PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:22.05-py3
RUN rm -rf /opt/pytorch  # remove 1.2GB dir

# RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx
RUN mkdir -p /home/
RUN mkdir -p /home/en_yolov6
RUN mkdir -p /home/en_yolov6/YOLOv6

RUN python3 --version
RUN pip --version
RUN pip install --upgrade pip
RUN python3 -m pip install boto3 botocore
RUN python3 -m pip uninstall -y torch torchvision torchtext Pillow
COPY ./requirements.txt /home/en_yolov6/YOLOv6/
RUN python3 -m pip install --no-cache -r /home/en_yolov6/YOLOv6/requirements.txt
RUN python3 -m pip install opencv-python-headless==4.5.5.64

WORKDIR /home/en_yolov6/YOLOv6
ENTRYPOINT [ "bash","run.sh" ]
COPY . /home/en_yolov6/YOLOv6/