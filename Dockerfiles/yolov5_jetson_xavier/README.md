# Info about machine learning docker image for jetson 
To create this Dockerfile, I use the l4t-ml docker image that contains Tensorflow, PyTorch and other popular ML and data science frameworks.

This image is available for several JetPacks. All the information about JetPack version and package version is in the following link: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-ml

# Docker build
To use it, you just have to clone the repository, run the command "docker build -t docker_image_name ." in the same level where Dockerfile file is placed. 

# Docker run
To run the docker image, just run the following command "docker run --rm -it --gpus all -v /path/to/volume:/volume_name_in_docker_container -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix docker_image_name"

# Run an example of yolov5 with a single image
Once you have executed the previous command, and you are inside of the docker container, you can test yolov5 with the command: "python3 detect.py --source ./data/images/zidane.jpg --conf 0.5"
