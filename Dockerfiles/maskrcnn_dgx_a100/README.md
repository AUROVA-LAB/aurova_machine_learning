# Info about machine learning docker image for dgx
To create this Dockerfile, we used nvcr.io/nvidia/tensorflow:20.11-tf2-py3 as initial docker image. It is specifically optimized for NVIDIA DGX A100 and includes several packages such as Ubuntu 18.04, Tensorflow, NVIDIA CUDA 11.1.0 or NVIDIA cuDNN 8.0.4. All the information can be checked at https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow. Also, we have a kernel version of 5.4.0-66-generic, NVIDIA-SMI at 450.102.04 version with CUDA at 11.0. 

After it, we installed the following packages:
- python 3.7.3
- pip
- cuda 11.2.0
- cudnn 8.1.0
- mask RCNN
- pycocotools
- wandb

# Weights & biases
For a complete experience, create an account on https://wandb.ai/site. It is free and will allow us to follow the training process from everywhere.

# Example of use 
1. Clone this repository and navigate to the path where the Dockerfile is:
```
cd ~/
git clone https://github.com/AUROVA-LAB/aurova_machine_learning.git
cd aurova_machine_learning/Dockerfiles/maskrcnn_dgx_a100/
```
2. Get cuDNN 8.1.0.77 (cudnn-11.2-linux-x64-v8.1.0.77.tgz) from https://developer.nvidia.com/cudnn and place it at the same level as the Dockerfile (~/aurova_machine_learning/Dockerfiles/maskrcnn_dgx_a100/). 

3. Get coco weights (mask_rcnn_coco.h5) in order to test Mask-RCNN from https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiL5_ei-sf3AhXfgv0HHXyWDTAQFnoECBIQAQ&url=https%3A%2F%2Fgithub.com%2Fmatterport%2FMask_RCNN%2Freleases%2Fdownload%2Fv2.0%2Fmask_rcnn_coco.h5&usg=AOvVaw0nAUAmHpcXDQ6mPgV9NckR and place it at the same level as the Dockerfile (~/aurova_machine_learning/Dockerfiles/maskrcnn_dgx_a100/).

4. Build the Dockerfile:
```
docker build -t aurova_maskrcnn .
```

5. Once the image is built, we have to run it by using the following command. 
- WARNING: adjust --gpus (which GPU to use) and -v (path to share with docker) flags if needed.
```
docker run --shm-size=6gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus "device=2" --rm -it --name aurova_maskrcnn -v /raid/aurova/docker/:/aurova_maskrcnn aurova_maskrcnn
```
6. Inside the running docker, we are able to run a demo. See below two examples.
```
cd ../aurova_maskrcnn/aurova_machine_learning/Dockerfiles/maskrcnn_dgx_a100/

ldconfig && python3.7 mask_rcnn.py --image ./images/athletic.jpg --saved_image ./images/new_athletic.jpg --weights ./mask_rcnn_coco.h5 
ldconfig && python3.7 mask_rcnn.py --image ./images/bilbao_council.jpg --saved_image ./images/new_bilbao_council.jpg --weights ./mask_rcnn_coco.h5 
```
<img src="/Dockerfiles/maskrcnn_dgx_a100/images/new_bilbao_council.jpg" width="503"> <img src="/Dockerfiles/maskrcnn_dgx_a100/images/new_athletic.jpg" width="503">

# Train and eval
After that, we are ready to train or test too. 
1. Edit your new classes in line 31 of program.py.
2. Edit your Mask-RCNN config on lines 220 - 407.
3. Edit your wandb user ID in line 872 of program.py.
4. Call the program taking into account available flags.
```
Train Mask R-CNN on MS COCO.

positional arguments:
  <command>             'train' or 'evaluate' on MS COCO

optional arguments:
  -h, --help            show this help message and exit
  --dataset_train /path/to/coco_dataset_train/
                        Directory of the MS-COCO training dataset
  --dataset_val /path/to/coco_dataset_val/
                        Directory of the MS-COCO validating dataset
  --dataset_test /path/to/coco_dataset_test/
                        Directory of the MS-COCO testing dataset
  --model /path/to/weights.h5
                        Path to weights .h5 file or 'coco'
  --logs /path/to/logs/
                        Logs and checkpoints directory
  --save_test /path/to/result_test_images/
                        Where to save results (analysed images)
```
```
ldconfig && cd ../aurova_maskrcnn && python3.7 program.py train

ldconfig && cd ../aurova_maskrcnn && python3.7 program.py evaluate
```

EXTRA: during the evaluation process match program.py backbone and the one used to train that weights.
