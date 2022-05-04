# Info about machine learning docker image for dgx
To create this Dockerfile, we used nvcr.io/nvidia/tensorflow:20.11-tf2-py3 as initial docker image. It is specifically optimized for NVIDIA A100 and includes several packages such as Ubuntu 18.04, Tensorflow, NVIDIA CUDA 11.1.0 or NVIDIA cuDNN 8.0.4. All the information can be checked at https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow.

After it, we installed the following packages:
- Python 3.7.3
- Pip
- Cuda 11.2.0
- Cudnn 8.1.0
- Mask RCNN
- Pycocotools
- Wandb

# Example of use 
1. Clone this repository, navigate to the path where the Dockerfile is and build it:
```
git clone https://github.com/AUROVA-LAB/aurova_machine_learning.git
cd aurova_machine_learning/Dockerfiles/maskrcnn_dgx_a100/
docker build -t aurova_maskrcnn .
```
2. Once the image is built, we have to run it by using the following command. 
WARNING: adjust --gpus (which GPU to use) and -v (path to share with docker) flags if needed.
```
docker run --shm-size=6gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus "device=2" --rm -it --name aurova_maskrcnn -v /raid/aurova/docker/:/aurova_maskrcnn
```
3. Inside the running docker, we are able to train or test. 
WARNING: to choose which backbone to use, we must edit program.py in line 250. Available options are resnet50 or resnet101.
WARNING: during the evaluation process match program.py backbone and the one used to train that weights.
WARNING: edit your wandb user ID in line 872 of program.py.
```
ldconfig && cd ../aurova_maskrcnn && python3.7 program.py train

ldconfig && cd ../aurova_maskrcnn && python3.7 program.py evaluate
```