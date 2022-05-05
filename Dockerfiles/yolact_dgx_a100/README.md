# Info about machine learning docker image for dgx
To create this Dockerfile, we used nvcr.io/nvidia/tensorflow:20.11-tf2-py3 as initial docker image. It is specifically optimized for NVIDIA DGX A100 and includes several packages such as Ubuntu 18.04, Tensorflow, NVIDIA CUDA 11.1.0 or NVIDIA cuDNN 8.0.4. All the information can be checked at https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow. Also, we have a kernel version of 5.4.0-66-generic, NVIDIA-SMI at 450.102.04 version with CUDA at 11.0.

After it, we installed the following packages:
- python 3.7.3
- pip
- cuda 11.1.0
- lzma headers
- wandb
- yolact

# Weights & biases
For a complete experience, create an account on https://wandb.ai/site. It is free and will allow us to follow the training process from everywhere.

# Example of use 
1. Clone this repository and navigate to the path where the Dockerfile is:
```
cd ~/
git clone https://github.com/AUROVA-LAB/aurova_machine_learning.git
cd aurova_machine_learning/Dockerfiles/yolact_dgx_a100/
```
2. Get coco weights (yolact_resnet50_54_800000.pth, yolact_darknet53_54_800000.pth, yolact_base_54_800000.pth, yolact_plus_resnet50_54_800000.pth and yolact_plus_base_54_800000.pth) in order to test Yolact from https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EUVpxoSXaqNIlssoLKOEoCcB1m0RpzGq_Khp5n1VX3zcUw, https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/ERrao26c8llJn25dIyZPhwMBxUp2GdZTKIMUQA3t0djHLw, https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EYRWxBEoKU9DiblrWx2M89MBGFkVVB_drlRd_v5sdT3Hgg, https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EcJAtMiEFlhAnVsDf00yWRIBUC4m8iE9NEEiV05XwtEoGw and https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EVQ62sF0SrJPrl_68onyHF8BpG7c05A8PavV4a849sZgEA and place it at the same level as the Dockerfile (~/aurova_machine_learning/Dockerfiles/yolact_dgx_a100/).
3. Build the Dockerfile:
```
docker build -t aurova_yolact .
```
4. Once the image is built, we have to run it by using the following command. 
- WARNING: adjust --gpus (which GPU to use) and -v (path to share with docker) flags if needed.
```
docker run --shm-size=6gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus "device=2" --rm -it --name aurova_yolact -v /raid/aurova/docker/:/aurova_yolact aurova_yolact
```
5. Inside the running docker, we are able to run a demo. See below two examples.
```
cd DCNv2/ && python3.7 setup.py build develop && cd ../yolact && python3.7 eval.py --trained_model ../../aurova_yolact/aurova_machine_learning/Dockerfiles/yolact_dgx_a100/yolact_base_54_800000.pth --image ../../aurova_yolact/aurova_machine_learning/Dockerfiles/yolact_dgx_a100/athletic.jpg:../../aurova_yolact/aurova_machine_learning/Dockerfiles/yolact_dgx_a100/new_athletic.jpg --config=yolact_base_config --top_k=5

cd DCNv2/ && python3.7 setup.py build develop && cd ../yolact && python3.7 eval.py --trained_model ../../aurova_yolact/aurova_machine_learning/Dockerfiles/yolact_dgx_a100/yolact_plus_base_54_800000.pth --image ../../aurova_yolact/aurova_machine_learning/Dockerfiles/yolact_dgx_a100/images/marcha.jpg:../../aurova_yolact/aurova_machine_learning/Dockerfiles/yolact_dgx_a100/images/new_marcha.jpg --config=yolact_plus_base_config --top_k=50
```
<img src="/Dockerfiles/maskrcnn_dgx_a100/images/new_zebra.jpg" width="503"> <img src="/Dockerfiles/maskrcnn_dgx_a100/images/new_marcha.jpg" width="503">

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
