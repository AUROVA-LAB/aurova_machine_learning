FROM nvcr.io/nvidia/tensorflow:20.11-tf2-py3

RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends apt-utils

# Install python 3.7.3
RUN apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
RUN apt-get install curl -y
RUN curl -O https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tar.xz
RUN tar -xf Python-3.7.3.tar.xz
RUN cd Python-3.7.3 && ./configure --enable-optimizations && make -j8 && make -j8 altinstall && python3.7 --version

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN apt-get install -y python3-distutils && apt-get install -y python3-apt
RUN python3.7 get-pip.py
RUN apt-get install -y python3.7-dev
RUN python3.7 -V && pip3 -V


RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Install cuda 11.2.0 (https://webnautes.tistory.com/1428)
# Check with "nvcc -V"
RUN export PATH=$PATH:/usr/local/cuda-11.0/bin
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/lib64
RUN export CUDADIR=/usr/local/cuda-11.0

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda-repo-ubuntu1804-11-2-local_11.2.0-460.27.04-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu1804-11-2-local_11.2.0-460.27.04-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu1804-11-2-local/7fa2af80.pub
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install cuda

RUN sh -c "echo 'export PATH=$PATH:/usr/local/cuda-11.2/bin' >> /etc/profile"
RUN sh -c "echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64' >> /etc/profile"
RUN sh -c "echo 'export CUDADIR=/usr/local/cuda-11.2' >> /etc/profile"
RUN source /etc/profile

# Install cudnn 8.1.0 (https://webnautes.tistory.com/1428)
# Check with "ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn"
COPY ./cudnn-11.2-linux-x64-v8.1.0.77.tgz ./../cudnn-11.2-linux-x64-v8.1.0.77.tgz
RUN tar xvzf ./../cudnn-11.2-linux-x64-v8.1.0.77.tgz
RUN cp ./cuda/include/cudnn* /usr/local/cuda/include
RUN cp ./cuda/lib64/libcudnn* /usr/local/cuda/lib64
RUN chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
RUN ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.1.0 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_adv_train.so.8
RUN ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.1.0 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8
RUN ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.1.0 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8
RUN ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.1.0 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
RUN ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.1.0 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_ops_train.so.8
RUN ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.1.0 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8
RUN ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn.so.8.1.0 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn.so.8

# Install Mask RCNN (tf 2.5)
RUN apt-get install -y git
RUN git clone https://github.com/zouyuelin/MASK_RCNN_2.5.0.git
RUN cd MASK_RCNN_2.5.0 && pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt-get install -y python3-pip && pip -V
RUN cd MASK_RCNN_2.5.0 && python3.7 setup.py install

# Install pycocotools
RUN pip3 install pycocotools

# Remove some libraries from path (avoid error "Loaded runtime CuDNN library: 8.0.4 but source was compiled with: 8.1.0.")
# https://stackoverflow.com/questions/61039316/loaded-runtime-cudnn-library-7-1-2-but-source-was-compiled-with-7-6-0-ubuntu
RUN rm /etc/ld.so.conf.d/00-cuda-compat.conf
RUN rm /etc/ld.so.conf.d/nvidia.conf
RUN rm /etc/ld.so.conf.d/x86_64-linux-gnu.conf

# Wandb graphics
RUN pip3 install wandb
#RUN wandb login xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 

# Path CUPTI
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64

# Run program
#CMD ldconfig && cd ../ignacio/maskrcnn && python3.7 program.py train --dataset_train ../dataset/program_split_coco/train/ --dataset_val ../dataset/program_split_coco/val/ --model ./mask_rcnn_coco.h5 --logs ./pruebas/logs_50

#CMD ldconfig && cd ../ignacio/maskrcnn && python3.7 program.py evaluate --dataset_test ../dataset/program_split_coco/test/ --model ./weights/resnet50.h5 --logs ./logs/ --save_test ./img_result/img_result_50_val/
