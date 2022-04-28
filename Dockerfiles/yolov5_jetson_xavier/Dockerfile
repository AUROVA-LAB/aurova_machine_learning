# base image that includes pytorch (ready for cuda use)
FROM nvcr.io/nvidia/l4t-ml:r32.5.0-py3

# cloning yolov5 repo
RUN git clone https://github.com/ultralytics/yolov5

# apt update
RUN apt update

# tkinter
RUN apt-get install python3-tk -y

RUN export LC_CTYPE=en_US.UTF-8
RUN LANG=en_US.UTF-8

# matplotlib
RUN pip3 install matplotlib>=3.2.2

# pillow
RUN pip3 install Pillow

#pyyaml
RUN pip3 install PyYAML>=1.4.1

# tqdm
RUN pip3 install tqdm>=4.41.0

# tensorboard
RUN pip3 install tensorboard>=2.4.1

# seaborn
RUN pip3 install seaborn>=0.11.0

# to run yolo with an example, execute this command after docker run,
# inside the yolov5 repository
# RUN  python3 detect.py --source ./data/images/zidane.jpg --conf 0.5

