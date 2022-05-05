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
<img src="/Dockerfiles/yolact_dgx_a100/images/new_zebra.jpg" width="503"> <img src="/Dockerfiles/yolact_dgx_a100/images/new_marcha.jpg" width="503">

# Train and eval
After that, we are ready to train or test too. 
1. Edit config.py. You should add train/validation and test dataset paths (see lines 177-193 for reference).
2. Edit config.py. You should also add the config for your neural network (see lines 798-869 & lines 917-976)
3. Edit your wandb user ID in line 176 of train.py.
4. Call the program taking into account available flags.
```
Yolact Training Script

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size for training
  --resume RESUME       Checkpoint state_dict file to resume training from. If
                        this is "interrupt", the model will resume training
                        from the interrupt file.
  --start_iter START_ITER
                        Resume training at this iter. If this is -1, the
                        iteration will be determined from the file name.
  --num_workers NUM_WORKERS
                        Number of workers used in dataloading
  --cuda CUDA           Use CUDA to train model
  --lr LR, --learning_rate LR
                        Initial learning rate. Leave as None to read this from
                        the config.
  --momentum MOMENTUM   Momentum for SGD. Leave as None to read this from the
                        config.
  --decay DECAY, --weight_decay DECAY
                        Weight decay for SGD. Leave as None to read this from
                        the config.
  --gamma GAMMA         For each lr step, what to multiply the lr by. Leave as
                        None to read this from the config.
  --save_folder SAVE_FOLDER
                        Directory for saving checkpoint models.
  --log_folder LOG_FOLDER
                        Directory for saving logs.
  --config CONFIG       The config object to use.
  --save_interval SAVE_INTERVAL
                        The number of iterations between saving the model.
  --validation_size VALIDATION_SIZE
                        The number of images to use for validation.
  --validation_epoch VALIDATION_EPOCH
                        Output validation information every n iterations. If
                        -1, do no validation.
  --keep_latest         Only keep the latest checkpoint instead of each one.
  --keep_latest_interval KEEP_LATEST_INTERVAL
                        When --keep_latest is on, don't delete the latest file
                        at these intervals. This should be a multiple of
                        save_interval or 0.
  --dataset DATASET     If specified, override the dataset specified in the
                        config with this one (example: coco2017_dataset).
  --no_log              Don't log per iteration information into log_folder.
  --log_gpu             Include GPU information in the logs. Nvidia-smi tends
                        to be slow, so set this with caution.
  --no_interrupt        Don't save an interrupt when KeyboardInterrupt is
                        caught.
  --batch_alloc BATCH_ALLOC
                        If using multiple GPUS, you can set this to be a comma
                        separated list detailing which GPUs should get what
                        local batch size (It should add up to your total batch
                        size).
  --no_autoscale        YOLACT will automatically scale the lr and the number
                        of iterations depending on the batch size. Set this if
                        you want to disable that.
  --initial_weights INITIAL_WEIGHTS
                        Place where weights during training are stored.

```
```
YOLACT COCO Evaluation

optional arguments:
  -h, --help            show this help message and exit
  --trained_model TRAINED_MODEL
                        Trained state_dict file path to open. If "interrupt",
                        this will open the interrupt file.
  --top_k TOP_K         Further restrict the number of predictions to parse
  --cuda CUDA           Use cuda to evaulate model
  --fast_nms FAST_NMS   Whether to use a faster, but not entirely correct
                        version of NMS.
  --cross_class_nms CROSS_CLASS_NMS
                        Whether compute NMS cross-class or per-class.
  --display_masks DISPLAY_MASKS
                        Whether or not to display masks over bounding boxes
  --display_bboxes DISPLAY_BBOXES
                        Whether or not to display bboxes around masks
  --display_text DISPLAY_TEXT
                        Whether or not to display text (class [score])
  --display_scores DISPLAY_SCORES
                        Whether or not to display scores in addition to
                        classes
  --display             Display qualitative results instead of quantitative
                        ones.
  --shuffle             Shuffles the images when displaying them. Doesn't have
                        much of an effect when display is off though.
  --ap_data_file AP_DATA_FILE
                        In quantitative mode, the file to save detections
                        before calculating mAP.
  --resume              If display not set, this resumes mAP calculations from
                        the ap_data_file.
  --max_images MAX_IMAGES
                        The maximum number of images from the dataset to
                        consider. Use -1 for all.
  --output_coco_json    If display is not set, instead of processing IoU
                        values, this just dumps detections into the coco json
                        file.
  --bbox_det_file BBOX_DET_FILE
                        The output file for coco bbox results if
                        --coco_results is set.
  --mask_det_file MASK_DET_FILE
                        The output file for coco mask results if
                        --coco_results is set.
  --config CONFIG       The config object to use.
  --output_web_json     If display is not set, instead of processing IoU
                        values, this dumps detections for usage with the
                        detections viewer web thingy.
  --web_det_path WEB_DET_PATH
                        If output_web_json is set, this is the path to dump
                        detections into.
  --no_bar              Do not output the status bar. This is useful for when
                        piping to a file.
  --display_lincomb DISPLAY_LINCOMB
                        If the config uses lincomb masks, output a
                        visualization of how those masks are created.
  --benchmark           Equivalent to running display mode but without
                        displaying an image.
  --no_sort             Do not sort images by hashed image ID.
  --seed SEED           The seed to pass into random.seed. Note: this is only
                        really for the shuffle and does not (I think) affect
                        cuda stuff.
  --mask_proto_debug    Outputs stuff for scripts/compute_mask.py.
  --no_crop             Do not crop output masks with the predicted bounding
                        box.
  --image IMAGE         A path to an image to use for display.
  --images IMAGES       An input folder of images and output folder to save
                        detected images. Should be in the format
                        input->output.
  --video VIDEO         A path to a video to evaluate on. Passing in a number
                        will use that index webcam.
  --video_multiframe VIDEO_MULTIFRAME
                        The number of frames to evaluate in parallel to make
                        videos play at higher fps.
  --score_threshold SCORE_THRESHOLD
                        Detections with a score under this threshold will not
                        be considered. This currently only works in display
                        mode.
  --dataset DATASET     If specified, override the dataset specified in the
                        config with this one (example: coco2017_dataset).
  --detect              Don't evauluate the mask branch at all and only do
                        object detection. This only works for --display and
                        --benchmark.
  --display_fps         When displaying / saving video, draw the FPS on the
                        frame
  --emulate_playback    When saving a video, emulate the framerate that you'd
                        get running in real-time mode.
  --map_test MAP_TEST   If you want to get mAP of test dataset, add --map_test

```
```
cd DCNv2/ && python3.7 setup.py build develop && cd ../yolact && python3.7 eval.py

cd DCNv2/ && python3.7 setup.py build develop && cd ../yolact && python3.7 train.py
```

EXTRA: during the evaluation process match eval.py trained model and the config chosen.
