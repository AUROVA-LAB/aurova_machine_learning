#!/usr/bin/env python3.7

# example of inference with a pre-trained coco model  
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import numpy as np
import pickle
import argparse

# define classes that the model knowns about
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# define the test configuration
class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(CLASS_NAMES)
    BACKBONE = "resnet101" 
    BATCH_SIZE = 1 


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


def display_results(image, boxes, masks, class_ids, class_names, scores=None,
                        show_mask=True, show_bbox=True, display_img=True,
                        save_img=True, img_name=None, mask_name = None):

	n_instances = boxes.shape[0]
	colors = color_map()
	image_mask = np.zeros((image.shape[0],image.shape[1]),np.uint8)
	print(np.unique(image_mask))
	print(image_mask.shape)
	num = 0
	num1 = 0
	num2 = 0

	for k in range(n_instances):
		color = colors[class_ids[k]].astype(np.uint8)

		if show_bbox:
			box = boxes[k]
			cls = class_names[class_ids[k]]  # Skip the Background
			score = scores[k]
			num1 = box[0]
			num2 = box[1]
			num3 = box[2]
			num4 = box[3]
			image = cv2.rectangle(image, (num2,num1,num4-num2,num3-num1), tuple([int(x) for x in color]), thickness = 1)
			font = cv2.FONT_HERSHEY_SIMPLEX
			image = cv2.putText(image, '{}: {:.3f}'.format(cls, score), (box[1], box[0]-10), font, 0.4, tuple([int(x) for x in color]), 1, cv2.LINE_AA)

		if show_mask:
			mask = masks[:, :, k]
			color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
			color_mask[mask] = color
			image = cv2.addWeighted(color_mask, 0.5, image.astype(np.uint8), 1, 0)


		# Para guardar la mascara para el geograsp (monocromatica, con valores segun numero de objetos)
		mask = masks[:, :, k]
		mono_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
		mono_mask[mask] = k+1
		image_mask = image_mask + mono_mask
		print("Numero asignado")
		print(k+1)
		print(np.unique(mono_mask))
		print(mono_mask.shape)
		print(np.unique(image_mask))
		print(image_mask.shape)


	if save_img:
		cv2.imwrite(img_name, image)

	return None

# Parse command line arguments
parser = argparse.ArgumentParser(
description='Test Mask R-CNN on MS COCO.')
parser.add_argument('--image', required=True,
                metavar="/path/to/image/",
                help='Image to be detected')
parser.add_argument('--saved_image', required=True,
                metavar="/path/to/save/image",
                help='Where to save the detected image')
parser.add_argument('--weights', required=True,
                metavar="/path/to/weights/",
                help='Weights to be used')



args = parser.parse_args()
print("Image: ", args.image)
print("Saved_image: ", args.saved_image)
print("Weights: ", args.weights)


# define the model
model = mrcnn.model.MaskRCNN(mode="inference", config=SimpleConfig(), model_dir=os.getcwd())

# load coco model weights
model.load_weights(filepath=args.weights, by_name=True)

img = cv2.imread(args.image)
# make prediction
results = model.detect([img], verbose=0)

# get dictionary for first prediction
r = results[0]

# Save results in a pkl file
print(r)

# save photo with bounding boxes, masks, class labels and scores
display_results(img, r['rois'], r['masks'], r['class_ids'], CLASS_NAMES, r['scores'], img_name = args.saved_image)
