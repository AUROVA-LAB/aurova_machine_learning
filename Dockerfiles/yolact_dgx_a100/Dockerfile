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

# Install cuda
RUN export PATH=$PATH:/usr/local/cuda-11.0/bin
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/lib64
RUN export CUDADIR=/usr/local/cuda-11.0

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda-repo-ubuntu1804-11-1-local_11.1.0-455.23.05-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu1804-11-1-local_11.1.0-455.23.05-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu1804-11-1-local/7fa2af80.pub
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install cuda

RUN sh -c "echo 'export PATH=$PATH:/usr/local/cuda-11.1/bin' >> /etc/profile"
RUN sh -c "echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1/lib64' >> /etc/profile"
RUN sh -c "echo 'export CUDADIR=/usr/local/cuda-11.1' >> /etc/profile"
RUN source /etc/profile

RUN export PATH="/usr/local/cuda/bin:$PATH" && \
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH" && \
    export CUDA_HOME=/usr/local/cuda/ && \
    export FORCE_CUDA="1"

# LZMA headers
RUN apt install liblzma-dev
RUN cp /usr/lib/python3.7/lib-dynload/_lzma.cpython-37m-x86_64-linux-gnu.so  /usr/local/lib/python3.7/

# Wandb graphics
RUN pip3 install wandb
#RUN wandb login xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Install
RUN git clone https://github.com/dbolya/yolact.git

RUN pip3 install ninja
RUN pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install cython
RUN pip3 install opencv-python
RUN pip3 install pillow 
RUN pip3 install pycocotools
RUN pip3 install matplotlib
RUN pip3 install data 

RUN git clone https://github.com/MatthewHowe/DCNv2.git
RUN cd DCNv2/DCN/ && ls && cp dcn_v2.py ../
RUN python3.7 DCNv2/setup.py build develop

COPY train.py ./yolact/train.py
COPY config.py ./yolact/data/
COPY coco.py ./yolact/data/
COPY eval.py ./yolact/


RUN export CUDA_VISIBLE_DEVICES=[0]
