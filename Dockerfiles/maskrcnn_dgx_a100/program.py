#!usr/bin/env python3.7

import os
import sys
import time
import numpy as np
import imgaug
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import zipfile
import urllib.request
import shutil
import pathlib
import argparse
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from datetime import datetime
import wandb
import tensorflow as tf
import subprocess
from mrcnn.utils import compute_ap
from mrcnn.utils import compute_ap_range
import time


class_names = ['BG', 'plastic', 'carton', 'glass', 'metal']

############################################################
#  Paths                                                   #
############################################################

# Actual absolute path of working directory
pwd = pathlib.Path().resolve()
print(pwd)

# Actual directory of file executed
pfd = pathlib.Path(__file__).parent.resolve()
print(pfd)

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
print(ROOT_DIR)

############################################################
#  /Paths                                                  #
############################################################


# Import Mask RCNN (to find local version of the library)
sys.path.append(ROOT_DIR)

# Variables for timers
total_epoch_time = 0.0
mAP_train_time = 0.0
mAP_val_time = 0.0
epoch_train_time = 0.0
stop_time = 0.0


def evaluate_model(dataset, model, model_train, cfg, len=50):
    APs = list()
    precisions_dict = {}
    recall_dict     = {}

    last_weights_path = model.find_last()
    print("Loaded weights for the inference model (last checkpoint of the train model): {0}".format(last_weights_path))
    model_train.load_weights(last_weights_path, by_name=True)

    iou_thresholds = np.arange(0.5,1.0,0.05)
    longitud = 0
    for index, image_id in enumerate(dataset.image_ids):
        longitud = longitud+1

    AP_image = np.zeros((longitud,12))
    i = 0

    for index,image_id in enumerate(dataset.image_ids):
        #if(index > len):
        #    break;
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert image into one sample
        sample = np.expand_dims(image, 0)
        # make prediction
        yhat = model_train.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        #AP, precisions, recalls, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        j=0
        for iou_threshold in iou_thresholds:
            AP_image_pt = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold)
            #print(AP_image_pt)
            AP_image[i][j] = AP_image_pt[0]
            j=j+1

        map_media = 0.0
        for j in range(11):
            map_media = map_media + AP_image[i][j]

        AP_image[i][11] = map_media / 10.0

        i=i+1
        #precisions_dict[image_id] = np.mean(precisions)
        #recall_dict[image_id] = np.mean(recalls)
        # store

    AP_result = np.zeros(12)
    for j in range(12):
        num = 0.0
        for k in range(longitud):
            num = num + AP_image[k][j]
        AP_result [j] = num / longitud


    #print("Fin: "+str(AP_result))
    # calculate the mean AP across all images

    return AP_result #,precisions_dict,recall_dict


###########################################################
# Callback class                                          #
###########################################################

class CustomSaver(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs={}):
        global total_epoch_time
        total_epoch_time  = time.time()

    def on_epoch_end(self, epoch, logs={}):

        global stop_time
        global total_epoch_time
        global mAP_train_time
        global mAP_val_time
        global epoch_train_time

        stop_time = time.time()
        epoch_train_time = stop_time - total_epoch_time

        mAP_train_time = time.time()
        eval_train = evaluate_model(dataset_train, model, model_train, config)
        stop_time = time.time()
        mAP_train_time = stop_time - mAP_train_time

        print(eval_train)
        print("Train_finished")

        mAP_val_time = time.time()
        eval_val = evaluate_model(dataset_val, model, model_train, config)
        stop_time = time.time()
        mAP_val_time = stop_time - mAP_val_time

        print(eval_val)
        print("val_finished")

        stop_time = time.time()
        total_epoch_time = stop_time - total_epoch_time

        #print(logs)
        wandb.log({"loss": logs["loss"]}, commit=False)
        wandb.log({"val_loss": logs["val_loss"]}, commit=False)
        wandb.log({"train mAP 0.5": eval_train[0]}, commit=False)
        wandb.log({"train mAP 0.55": eval_train[1]}, commit=False)
        wandb.log({"train mAP 0.60": eval_train[2]}, commit=False)
        wandb.log({"train mAP 0.65": eval_train[3]}, commit=False)
        wandb.log({"train mAP 0.70": eval_train[4]}, commit=False)
        wandb.log({"train mAP 0.75": eval_train[5]}, commit=False)
        wandb.log({"train mAP 0.80": eval_train[6]}, commit=False)
        wandb.log({"train mAP 0.85": eval_train[7]}, commit=False)
        wandb.log({"train mAP 0.90": eval_train[8]}, commit=False)
        wandb.log({"train mAP 0.95": eval_train[9]}, commit=False)
        wandb.log({"train mAP mean": eval_train[10]}, commit=False)
        wandb.log({"val mAP 0.5": eval_val[0]}, commit=False)
        wandb.log({"val mAP 0.55": eval_val[1]}, commit=False)
        wandb.log({"val mAP 0.60": eval_val[2]}, commit=False)
        wandb.log({"val mAP 0.65": eval_val[3]}, commit=False)
        wandb.log({"val mAP 0.70": eval_val[4]}, commit=False)
        wandb.log({"val mAP 0.75": eval_val[5]}, commit=False)
        wandb.log({"val mAP 0.80": eval_val[6]}, commit=False)
        wandb.log({"val mAP 0.85": eval_val[7]}, commit=False)
        wandb.log({"val mAP 0.90": eval_val[8]}, commit=False)
        wandb.log({"val mAP 0.95": eval_val[9]}, commit=False)
        wandb.log({"val mAP mean": eval_val[10]}, commit=False)
        wandb.log({"epoch_train_time": epoch_train_time}, commit=False)
        wandb.log({"mAP_train_time": mAP_train_time}, commit=False)
        wandb.log({"mAP_val_time": mAP_val_time}, commit=False)
        wandb.log({"total_epoch_time": total_epoch_time}, commit=True)


    #def on_train_batch_end(self, batch, logs={}):

        #model.keras_model.save("hoy.h5")

        #print("In batch..."+str(batch))
        #if (batch == (model.STEPS_PER_EPOCH-1)):
        #    print("READY")



###########################################################
# /Callback class                                         #
###########################################################




############################################################
#  Configurations                                          #
############################################################


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "first_trashnet"

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    BACKBONE = "resnet101"

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    
    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT = 6000

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 1024

    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0

    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = None  # Defaulting to False since batch size is often small

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0


    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

############################################################
#  /Configurations                                         #
############################################################


############################################################
#  Dataset                                                 #
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, class_ids=None,
                  return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, test)
        class_ids: If provided, only loads images that have the given classes.
        return_coco: If True, returns the COCO object.
        """
        print("Load dataset from (def load_coco ~ l250): "+ dataset_dir+"/annotations.json")
        coco = COCO(dataset_dir+"/annotations.json")
        print("Result loading dataset (def load_coco ~ l260)")

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        print("Class_ids (def load_coco ~ l275)")
	
        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])
            print("Loading classes (def load_coco ~ l280): "+ "coco" + ", "+ str(i) + "& " + coco.loadCats(i)[0]["name"])


        image_dir = dataset_dir
        print("Image_dir (def load_coco ~ l285): "+ image_dir)
        print("Image_example (def load_coco ~ l290): " + os.path.join(image_dir, coco.imgs[0]['file_name']) + ", "+str(coco.imgs[0]["width"]) + ", "+ str(coco.imgs[0]["height"]))
	
        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

        print("Images added (def load_coco ~ l295)")

        if return_coco:
            return coco


    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        #print("Type of image to load (def load_mask ~ l315): "+image_info["source"])

        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        #print("Loading annotations (def load_mask ~ l325)")

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        #print("End loading annotations (def load_mask ~ l350)")

        # Pack instance masks into an array
        #print("Class ids (def load_mask ~ l350)")

        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)


    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

###########################################################
#  /Dataset                                                #
############################################################

############################################################
#  COCO Evaluation                                         #
############################################################
def color_map(N=256, normalized=False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255 if normalized else cmap
        return cmap


def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks, path, image, colors, l, coco_images_ids):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                    "image_id": image_id,
                    "category_id": dataset.get_source_class_id(class_id, "coco"),
                    "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                    "score": score,
                    "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)

    if path is None:
        print("Not saving result images (def evaluate_coco ~ l510)")
    else:
        print("Saving analysed images in "+os.path.join(path, str(l)+".png"))


        for image_id in image_ids:
            for i in range(rois.shape[0]):
                class_id = class_ids[i]
                score = scores[i]
                bbox = np.around(rois[i], 1)
                mask = masks[:, :, i]

                color = colors[class_id].astype(np.int)

                class_id = dataset.get_source_class_id(class_id, "coco")
                class_id = class_names[class_id]
                print(class_id)
                cv2.rectangle(image, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), tuple([int(x) for x in color]), 1)
                cv2.putText(image, '{}: {:.3f}'.format(class_id, score), (bbox[1], bbox[0]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, tuple([int(x) for x in color]), 1, cv2.LINE_AA)

                for j in range(0,mask.shape[0]):
                    for k in range(0,mask.shape[1]):
                        if mask[j][k] == 1:
                            image[j][k] = tuple([int(x) for x in color])


                img_float32 = np.float32(image)
                lab_image = cv2.cvtColor(img_float32, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(path, str(l)+".png"), lab_image)

    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", image_ids=None, path=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
        """

    print("In the eval coco function (def evaluate_coco ~ l440)")
    # Pick images
    image_ids = dataset.image_ids

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    print("Get COCO image IDs (def evaluate_coco ~ l450)")

    t_prediction = 0
    t_start = time.time()

    results = []
    print("Started analizing images (def evaluate_coco ~ l455)")

    colors = color_map()

    for i, image_id in enumerate(image_ids):
        print("Analizing "+str(i)+" image...")

        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8), path, image, colors, i, coco_image_ids)
        results.extend(image_results)

    print("Finished analizing images (def evaluate_coco ~ l470)")


    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


def evaluate_own_metrics(dataset, model_train, cfg):

    APs = list()
    precisions_dict = {}
    recall_dict     = {}

    iou_thresholds = np.arange(0.5,1.0,0.05)
    longitud = 0
    for index, image_id in enumerate(dataset.image_ids):
        longitud = longitud+1

    AP_image = np.zeros((longitud,12))
    i = 0

    for index,image_id in enumerate(dataset.image_ids):
        #if(index > len):
        #    break;
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert image into one sample
        sample = np.expand_dims(image, 0)
        # make prediction
        yhat = model_train.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        #AP, precisions, recalls, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        j=0
        for iou_threshold in iou_thresholds:
            AP_image_pt = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold)
            #print(AP_image_pt)
            AP_image[i][j] = AP_image_pt[0]
            j=j+1

        map_media = 0.0
        for j in range(11):
            map_media = map_media + AP_image[i][j]

        AP_image[i][11] = map_media / 10.0

        i=i+1
        #precisions_dict[image_id] = np.mean(precisions)
        #recall_dict[image_id] = np.mean(recalls)
        # store

    AP_result = np.zeros(12)
    for j in range(12):
        num = 0.0
        for k in range(longitud):
            num = num + AP_image[k][j]
        AP_result [j] = num / longitud


    #print("Fin: "+str(AP_result))
    # calculate the mean AP across all images

    return AP_result #,precisions_dict,recall_dict

############################################################
#  /COCO Evaluation                                        #
############################################################

############################################################
#  Training
############################################################


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset_train', required=False,
                        metavar="/path/to/coco_dataset_train/",
                        help='Directory of the MS-COCO training dataset')
    parser.add_argument('--dataset_val', required=False,
                        metavar="/path/to/coco_dataset_val/",
                        help='Directory of the MS-COCO validating dataset')
    parser.add_argument('--dataset_test', required=False,
                        metavar="/path/to/coco_dataset_test/",
                        help='Directory of the MS-COCO testing dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=True,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory')
    parser.add_argument('--save_test', required=False,
                        metavar="/path/to/result_test_images/",
                        help='Where to save results (analysed images)')


    args = parser.parse_args()
    print("Command: ", args.command)
    print("Dataset_train: ", args.dataset_train)
    print("Dataset_val: ", args.dataset_val)
    print("Dataset_test: ", args.dataset_test)
    print("Model: ", args.model)
    print("Logs: ", args.logs)
    print("Save_test: ", args.save_test)

    # Configurations
    if args.command == "train":
        config = CocoConfig()

        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config_train = InferenceConfig()

    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
        model_train = modellib.MaskRCNN(mode="inference", config=config_train,
                                        model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    if args.command == "train":
        model.load_weights(model_path, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask", "rpn_model"])
    else:
        model.load_weights(model_path, by_name=True)

    # Wandb app
    if args.command == "train":

        wandb.init(name='MaskRCNN - '+str(datetime.now()),
                   project='Mask-RCNN training')
        subprocess.run(["wandb", "login", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"])
        wandb.config.name = config.NAME
        wandb.config.images_per_gpu = config.IMAGES_PER_GPU
        wandb.config.steps_per_epoch = config.STEPS_PER_EPOCH
        wandb.config.validation_steps = config.VALIDATION_STEPS
        wandb.config.backbone = config.BACKBONE
        wandb.config.backbone_strides = config.BACKBONE_STRIDES
        wandb.config.fpn_classif_fc_layers_size = config.FPN_CLASSIF_FC_LAYERS_SIZE
        wandb.config.top_down_pyramid_size = config.TOP_DOWN_PYRAMID_SIZE
        wandb.config.num_classes = config.NUM_CLASSES
        wandb.config.rpn_anchor_scales = config.RPN_ANCHOR_SCALES
        wandb.config.rpn_anchor_ratios = config.RPN_ANCHOR_RATIOS
        wandb.config.rpn_anchor_stride = config.RPN_ANCHOR_STRIDE
        wandb.config.rpn_nms_threshold = config.RPN_NMS_THRESHOLD
        wandb.config.rpn_train_anchors_per_image = config.RPN_TRAIN_ANCHORS_PER_IMAGE
        wandb.config.pre_nms_limit = config.PRE_NMS_LIMIT
        wandb.config.post_nms_rois_training = config.POST_NMS_ROIS_TRAINING
        wandb.config.post_nms_rois_inference = config.POST_NMS_ROIS_INFERENCE
        wandb.config.use_mini_mask = config.USE_MINI_MASK
        wandb.config.mini_mask_shape = config.MINI_MASK_SHAPE
        wandb.config.image_resize_mode = config.IMAGE_RESIZE_MODE
        wandb.config.image_min_dim = config.IMAGE_MIN_DIM
        wandb.config.image_max_dim = config.IMAGE_MAX_DIM
        wandb.config.image_min_scale = config.IMAGE_MIN_SCALE
        wandb.config.image_channel_count = config.IMAGE_CHANNEL_COUNT
        wandb.config.mean_pixel = config.MEAN_PIXEL
        wandb.config.train_rois_per_image = config.TRAIN_ROIS_PER_IMAGE
        wandb.config.roi_positive_ratio = config.ROI_POSITIVE_RATIO
        wandb.config.pool_size = config.POOL_SIZE
        wandb.config.mask_pool_size = config.MASK_POOL_SIZE
        wandb.config.mask_shape = config.MASK_SHAPE
        wandb.config.max_gt_instances = config.MAX_GT_INSTANCES
        wandb.config.rpn_bbox_std_dev = config.RPN_BBOX_STD_DEV
        wandb.config.bbox_std_dev = config.BBOX_STD_DEV
        wandb.config.detection_max_instances = config.DETECTION_MAX_INSTANCES
        wandb.config.detection_min_confidence = config.DETECTION_MIN_CONFIDENCE
        wandb.config.detection_nms_threshold = config.DETECTION_NMS_THRESHOLD
        wandb.config.learning_rate = config.LEARNING_RATE
        wandb.config.learning_momentum = config.LEARNING_MOMENTUM
        wandb.config.weight_decay = config.WEIGHT_DECAY
        wandb.config.loss_weights = config.LOSS_WEIGHTS
        wandb.config.use_rpn_rois = config.USE_RPN_ROIS
        wandb.config.train_bn = config.TRAIN_BN
        wandb.config.gradient_clip_norm = config.GRADIENT_CLIP_NORM
        wandb.config.gpu_count = config.GPU_COUNT


    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset_train, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()        
        dataset_val.load_coco(args.dataset_val, "validate")
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        #augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        saver = CustomSaver()

        # Training - Stage 1
        print("Training network heads")
        epoch_train_time = time.time()
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads', custom_callbacks=[saver])

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+', custom_callbacks=[saver])

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all', custom_callbacks=[saver])

    elif args.command == "evaluate":
        # Evaluate dataset
        dataset_test = CocoDataset()
        coco = dataset_test.load_coco(args.dataset_test, "test", return_coco=True)
        dataset_test.prepare()
        print("Prueba: "+ args.save_test)
        evaluate_coco(model, dataset_test, coco, "segm", path = args.save_test)
        eval_val = evaluate_own_metrics(dataset_test, model, config)
        print("Eval_val: "+str(eval_val))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
