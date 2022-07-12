# YOLOv6 by https://github.com/meituan/YOLOv6

# Start FROM NVIDIA PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:22.05-py3
RUN rm -rf /opt/pytorch  # remove 1.2GB dir

# RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx
RUN mkdir -p /home/
RUN mkdir -p /home/en_yolov6
RUN mkdir -p /home/en_yolov6/YOLOv6

COPY ./* /home/en_yolov6/YOLOv6
# RUN git clone https://github.com/kanakedge/YOLOv6.git /usr/src/YOLOv6 
RUN python3 --version
RUN pip --version
RUN pip install --upgrade pip
RUN python3 -m pip install boto3 botocore
RUN python3 -m pip uninstall -y torch torchvision torchtext Pillow
RUN python3 -m pip install --no-cache -r /usr/src/YOLOv6/requirements.txt
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
# COPY . /usr/src/app


