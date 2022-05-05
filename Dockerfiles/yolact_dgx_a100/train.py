from data import *
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss
from yolact import Yolact
import os
import sys
import time
import math, random
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime
import subprocess
import wandb
import eval as eval_script

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

###################################################################
# Arguments for parsing                                           #
###################################################################

# Possible parsed arguments
parser = argparse.ArgumentParser(description='Yolact Training Script')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from. If this is "interrupt", the model will resume training from the interrupt file.')
parser.add_argument('--start_iter', default=-1, type=int, help='Resume training at this iter. If this is -1, the iteration will be determined from the file name.')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
parser.add_argument('--lr', '--learning_rate', default=None, type=float, help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--momentum', default=None, type=float, help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float, help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--gamma', default=None, type=float, help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
parser.add_argument('--save_folder', default='weights/', help='Directory for saving checkpoint models.')
parser.add_argument('--log_folder', default='logs/', help='Directory for saving logs.')
parser.add_argument('--config', default=None, help='The config object to use.')
parser.add_argument('--save_interval', default=10000, type=int, help='The number of iterations between saving the model.')
parser.add_argument('--validation_size', default=5000, type=int, help='The number of images to use for validation.')
parser.add_argument('--validation_epoch', default=2, type=int, help='Output validation information every n iterations. If -1, do no validation.')
parser.add_argument('--keep_latest', dest='keep_latest', action='store_true', help='Only keep the latest checkpoint instead of each one.')
parser.add_argument('--keep_latest_interval', default=100000, type=int, help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
parser.add_argument('--dataset', default=None, type=str, help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--no_log', dest='log', action='store_false', help='Don\'t log per iteration information into log_folder.')
parser.add_argument('--log_gpu', dest='log_gpu', action='store_true', help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false', help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
parser.add_argument('--batch_alloc', default=None, type=str, help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false', help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')
parser.add_argument('--initial_weights', dest='initial_weights', help='Place where weights during training are stored.')

# Set default parameters
parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
# Get parameters
args = parser.parse_args()

# Checking received parameters
print("Name of yolact chosen -- from config_yolact.py: "+str(args.config))
if args.config is not None:
    set_cfg(args.config)

print("Copying paths of dataset -- from config_yolact.py: "+str(args.dataset))
if args.dataset is not None:
    set_dataset(args.dataset)

print()
print("Change lr and iterations checking the batch size: "+ str(args.autoscale))


if args.autoscale and args.batch_size != 8:
    factor = args.batch_size / 8
    if __name__ == '__main__':
        print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))

    print()
    print("CAREFUL: args. autoscale is set. Changing lr, max_iter and lr_steps.")
    print("Current: "+str(cfg.lr)+", "+str(cfg.max_iter)+", "+str(cfg.lr_steps))

    cfg.lr *= factor
    cfg.max_iter //= factor
    cfg.lr_steps = [x // factor for x in cfg.lr_steps]

    print("New: "+str(cfg.lr)+", "+str(cfg.max_iter)+", "+str(cfg.lr_steps))

# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))
replace('lr')
replace('decay')
replace('gamma')
replace('momentum')

# This is managed by set_lr
cur_lr = args.lr

print()
print("Checking if GPU is available: "+str(torch.cuda.device_count()))
if torch.cuda.device_count() == 0:
    print('No GPUs detected. Exiting...')
    exit(-1)

if args.batch_size // torch.cuda.device_count() < 6:
    if __name__ == '__main__':
        print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    cfg.freeze_bn = True

loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']

print("Checking if cuda is available: "+str(torch.cuda.is_available()))
print("Use CUDA?: "+str(args.cuda))
print()
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """
    
    def __init__(self, net:Yolact, criterion:MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion
    
    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        losses = self.criterion(self.net, preds, targets, masks, num_crowds)
        return losses

class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
            [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])
        
        return out

def train():

    # Wandb app
    wandb.init(name='Yolact - '+str(datetime.datetime.now()),
               project='Yolact training')
    subprocess.run(["wandb", "login", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"])

    #for k, v in vars(cfg).items():
    #    print(k, ' = ', v)

    my_dict = vars(cfg.dataset)
    my_dict2 = vars(cfg.fpn)
    my_dict3 = vars(cfg.backbone)
    my_dict4 = vars(cfg.backbone.transform)

    wandb.config.dataset_name = my_dict['name']
    wandb.config.dataset_train_images = my_dict['train_images']
    wandb.config.dataset_train_info = my_dict['train_info']
    wandb.config.dataset_valid_images = my_dict['valid_images']
    wandb.config.dataset_valid_info = my_dict['valid_info']
    wandb.config.dataset_has_gt = my_dict['has_gt']
    wandb.config.dataset_class_names = my_dict['class_names']
    wandb.config.label_map = my_dict['label_map']
    wandb.config.num_classes = cfg.num_classes
    wandb.config.max_iter = cfg.max_iter
    wandb.config.max_num_detections = cfg.max_num_detections
    wandb.config.lr = cfg.lr
    wandb.config.momentum = cfg.momentum
    wandb.config.decay = cfg.decay
    wandb.config.gamma = cfg.gamma
    wandb.config.lr_steps = cfg.lr_steps
    wandb.config.lr_warmup_init = cfg.lr_warmup_init
    wandb.config.lr_warmup_until = cfg.lr_warmup_until
    wandb.config.conf_alpha = cfg.conf_alpha
    wandb.config.bbox_alpha = cfg.bbox_alpha
    wandb.config.mask_alpha = cfg.mask_alpha
    wandb.config.eval_mask_branch = cfg.eval_mask_branch
    wandb.config.nms_top_k = cfg.nms_top_k
    wandb.config.nms_conf_thresh = cfg.nms_conf_thresh
    wandb.config.nms_thresh = cfg.nms_thresh
    wandb.config.mask_type = cfg.mask_type
    wandb.config.mask_size = cfg.mask_size
    wandb.config.masks_to_train = cfg.masks_to_train
    wandb.config.mask_proto_src = cfg.mask_proto_src
    wandb.config.mask_proto_net = cfg.mask_proto_net
    wandb.config.mask_proto_bias = cfg.mask_proto_bias

    wandb.config.mask_proto_prototype_activation = cfg.mask_proto_prototype_activation
    wandb.config.mask_proto_mask_activation = cfg.mask_proto_mask_activation
    wandb.config.mask_proto_coeff_activation = cfg.mask_proto_coeff_activation

    wandb.config.mask_proto_crop = cfg.mask_proto_crop
    wandb.config.mask_proto_crop_expand = cfg.mask_proto_crop_expand
    wandb.config.mask_proto_loss = cfg.mask_proto_loss
    wandb.config.mask_proto_binarize_downsampled_gt = cfg.mask_proto_binarize_downsampled_gt
    wandb.config.mask_proto_normalize_mask_loss_by_sqrt_area = cfg.mask_proto_normalize_mask_loss_by_sqrt_area
    wandb.config.mask_proto_reweight_mask_loss = cfg.mask_proto_reweight_mask_loss
    wandb.config.mask_proto_grid_file = cfg.mask_proto_grid_file
    wandb.config.mask_proto_use_grid = cfg.mask_proto_use_grid
    wandb.config.mask_proto_coeff_gate = cfg.mask_proto_coeff_gate
    wandb.config.mask_proto_prototypes_as_features = cfg.mask_proto_prototypes_as_features
    wandb.config.mask_proto_prototypes_as_features_no_grad = cfg.mask_proto_prototypes_as_features_no_grad
    wandb.config.mask_proto_remove_empty_masks = cfg.mask_proto_remove_empty_masks
    wandb.config.mask_proto_reweight_coeff = cfg.mask_proto_reweight_coeff
    wandb.config.mask_proto_coeff_diversity_loss = cfg.mask_proto_coeff_diversity_loss
    wandb.config.mask_proto_coeff_diversity_alpha = cfg.mask_proto_coeff_diversity_alpha
    wandb.config.mask_proto_normalize_emulate_roi_pooling = cfg.mask_proto_normalize_emulate_roi_pooling
    wandb.config.mask_proto_double_loss = cfg.mask_proto_double_loss
    wandb.config.mask_proto_double_loss_alpha = cfg.mask_proto_double_loss_alpha
    wandb.config.mask_proto_split_prototypes_by_head = cfg.mask_proto_split_prototypes_by_head
    wandb.config.mask_proto_crop_with_pred_box = cfg.mask_proto_crop_with_pred_box
    wandb.config.augment_photometric_distort = cfg.augment_photometric_distort
    wandb.config.augment_expand = cfg.augment_expand
    wandb.config.augment_random_sample_crop = cfg.augment_random_sample_crop
    wandb.config.augment_random_mirror = cfg.augment_random_mirror
    wandb.config.augment_random_flip = cfg.augment_random_flip
    wandb.config.augment_random_rot90 = cfg.augment_random_rot90
    wandb.config.discard_box_width = cfg.discard_box_width
    wandb.config.discard_box_height = cfg.discard_box_height
    wandb.config.freeze_bn = cfg.freeze_bn

    wandb.config.fpn_num_features = my_dict2['num_features']
    wandb.config.fpn_interpolation_mode = my_dict2['interpolation_mode']
    wandb.config.fpn_num_downsample = my_dict2['num_downsample']
    wandb.config.fpn_use_conv_downsample = my_dict2['use_conv_downsample']
    wandb.config.fpn_pad = my_dict2['pad']
    wandb.config.fpn_relu_downsample_layers = my_dict2['relu_downsample_layers']
    wandb.config.fpn_relu_pred_layers = my_dict2['relu_pred_layers']

    wandb.config.share_prediction_module = cfg.share_prediction_module
    wandb.config.ohem_use_most_confident = cfg.ohem_use_most_confident
    wandb.config.use_focal_loss = cfg.use_focal_loss
    wandb.config.focal_loss_alpha = cfg.focal_loss_alpha
    wandb.config.focal_loss_gamma = cfg.focal_loss_gamma
    wandb.config.focal_loss_init_pi =cfg.focal_loss_init_pi
    wandb.config.use_class_balanced_conf =cfg.use_class_balanced_conf
    wandb.config.use_sigmoid_focal_loss =cfg.use_sigmoid_focal_loss
    wandb.config.use_objectness_score =cfg.use_objectness_score
    wandb.config.use_class_existence_loss =cfg.use_class_existence_loss
    wandb.config.class_existence_alpha =cfg.class_existence_alpha
    wandb.config.use_semantic_segmentation_loss =cfg.use_semantic_segmentation_loss
    wandb.config.semantic_segmentation_alpha =cfg.semantic_segmentation_alpha
    wandb.config.use_mask_scoring =cfg.use_mask_scoring
    wandb.config.mask_scoring_alpha =cfg.mask_scoring_alpha
    wandb.config.use_change_matching =cfg.use_change_matching
    wandb.config.extra_head_net  =cfg.extra_head_net
    wandb.config.head_layer_params =cfg.head_layer_params
    wandb.config.extra_layers =cfg.extra_layers
    wandb.config.positive_iou_threshold =cfg.positive_iou_threshold
    wandb.config.negative_iou_threshold =cfg.negative_iou_threshold
    wandb.config.ohem_negpos_ratio =cfg.ohem_negpos_ratio
    wandb.config.crowd_iou_threshold =cfg.crowd_iou_threshold
    wandb.config.mask_dim =cfg.mask_dim
    wandb.config.max_size =cfg.max_size
    wandb.config.force_cpu_nms =cfg.force_cpu_nms
    wandb.config.use_coeff_nms =cfg.use_coeff_nms
    wandb.config.use_instance_coeff =cfg.use_instance_coeff
    wandb.config.num_instance_coeffs =cfg.num_instance_coeffs
    wandb.config.train_masks =cfg.train_masks
    wandb.config.train_boxes =cfg.train_boxes
    wandb.config.use_gt_bboxes =cfg.use_gt_bboxes
    wandb.config.preserve_aspect_ratio =cfg.preserve_aspect_ratio
    wandb.config.use_prediction_module =cfg.use_prediction_module
    wandb.config.use_yolo_regressors =cfg.use_yolo_regressors
    wandb.config.use_prediction_matching =cfg.use_prediction_matching
    wandb.config.delayed_settings =cfg.delayed_settings
    wandb.config.no_jit =cfg.no_jit

    wandb.config.backbone_name = my_dict3['name']
    wandb.config.backbone_path = my_dict3['path']
    wandb.config.backbone_type = my_dict3['type']
    wandb.config.backbone_args = my_dict3['args']

    wandb.config.backbone_transform_channel_order = my_dict4['channel_order']
    wandb.config.backbone_transform_normalize = my_dict4['normalize']
    wandb.config.backbone_transform_subtract_means = my_dict4['subtract_means']
    wandb.config.backbone_transform_to_float = my_dict4['to_float']

    wandb.config.backbone_selected_layers = my_dict3['selected_layers']
    wandb.config.backbone_pred_scales = my_dict3['pred_scales']
    wandb.config.backbone_pred_aspect_ratios = my_dict3['pred_aspect_ratios']
    wandb.config.backbone_use_pixel_scales = my_dict3['use_pixel_scales']
    wandb.config.backbone_preapply_sqrt = my_dict3['preapply_sqrt']
    wandb.config.backbone_use_square_anchors = my_dict3['use_square_anchors']

    wandb.config.name =cfg.name
    wandb.config.use_maskiou =cfg.use_maskiou
    wandb.config.maskiou_net =cfg.maskiou_net
    wandb.config.discard_mask_area =cfg.discard_mask_area
    wandb.config.maskiou_alpha =cfg.maskiou_alpha
    wandb.config.rescore_mask =cfg.rescore_mask
    wandb.config.rescore_bbox =cfg.rescore_bbox
    wandb.config.maskious_to_train =cfg.maskious_to_train


    if not os.path.exists(args.save_folder):
        #os.mkdir(args.save_folder)
        print("Save folder must be created before running docker")
        return

    print()
    print("Save folder in: "+str( args.save_folder))

    print()
    print("Image path is: "+cfg.dataset.train_images)
    print("Label path is: "+cfg.dataset.train_info)

    dataset = COCODetection(image_path=cfg.dataset.train_images,
                            info_file=cfg.dataset.train_info,
                            transform=SSDAugmentation(MEANS))

    print()
    print("Dataset_train loaded: "+str(dataset))

    print()
    print("Apply validation data every X epochs: "+str(args.validation_epoch))
    print("Size of validation data: "+str(args.validation_size))


    if args.validation_epoch > 0:
        setup_eval()
        val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    transform=BaseTransform(MEANS))
        print()
        print("Dataset_val loaded:"+str(val_dataset))

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    yolact_net = Yolact()
    net = yolact_net
    net.train()

    print()
    print("Created yolact network")

    print()
    print("Log directory provided: "+str(args.log))
    if args.log:
        log = Log(cfg.name, args.log_folder, dict(args._get_kwargs()), overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)
        print("Info to be written in the log file")
        print("Name: "+str(cfg.name))
        print("Path of log: "+str(args.log_folder))
        print("Specified parameters: "+str(args._get_kwargs()))
        print("Clean log folder before training: "+str(args.resume))
        print("Log GPU info - warning, it is slow: "+str(args.log_gpu))


    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check    
    print()
    print("Train from "+str(args.resume))

    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        yolact_net.load_weights(args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        print('Initializing weights. Using: '+str(args.initial_weights))
        yolact_net.init_weights(args.initial_weights) # backbone_path=args.save_folder + cfg.backbone.path)

    print()
    print("Setting optimizer...")
    print("SGD with "+str(net.parameters())+", learning rate "+str(args.lr)+", momentum "+str(args.momentum)+" and weight decay "+str(args.decay))
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay)

    print()
    print("Configuring SSD weighted loss function... ")
    print("Number of classes: "+str(cfg.num_classes)+", pos threshold: "+str(cfg.positive_iou_threshold)+", neg threshold: "+str(cfg.negative_iou_threshold)+" and ratio negpos: "+str(cfg.ohem_negpos_ratio))
    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=cfg.ohem_negpos_ratio)

    print()
    print("Using multiple GPUs? "+str(args.batch_alloc))
    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size))
            exit(-1)

    print()
    print("Inserting SSD weighted loss function in net...")
    net = CustomDataParallel(NetLoss(net, criterion))

    print()
    print("Checking CUDA again: "+str(args.cuda))
    if args.cuda:
        net = net.cuda()
    
    print()
    print("Layers diferent from convolutional in backbone are frozen: "+str(cfg.freeze_bn))
    # Initialize everything
    if not cfg.freeze_bn: yolact_net.freeze_bn() # Freeze bn so we don't kill our means
    yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())
    if not cfg.freeze_bn: yolact_net.freeze_bn(True)

    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = max(args.start_iter, 0)
    last_time = time.time()

    epoch_size = len(dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)
    
    print()
    print("The training will be of "+str(num_epochs)+" epochs and "+str(cfg.max_iter)+" max_iter")

    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0

    print()
    print("Organizing dataset for our network. Params: batch_size: "+str(args.batch_size)+", num_workers: "+str(args.num_workers)+" and dataset: "+str(dataset))
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    
    
    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
    print()
    print("Setting save_path")
    time_avg = MovingAverage()

    global loss_types # Forms the print order
    loss_avgs  = { k: MovingAverage(100) for k in loss_types }


    print()
    print("Legend of while-training parameters: ")
    print("B: Box Location Loss, C: Class Confidence Loss, M: Mask Loss, S: Semantic Segmentation Loss, I: Mask IoU Loss, T: Sum of the mean of the losses, ETA: Estimated Time of Arrival, timer: time on batch size")

    print()
    print('Begin training!')
    print()

    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(num_epochs):
            # Resume from start_iter
            if (epoch+1)*epoch_size < iteration:
                continue
            
            for datum in data_loader:
                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch+1)*epoch_size:
                    break

                # Stop at the configured number of iterations even if mid-epoch
                if iteration == cfg.max_iter:
                    break

                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])

                        # Reset the loss averages because things might have changed
                        for avg in loss_avgs:
                            avg.reset()
                
                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer, (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    print("Changing lr. Step_index: "+str(step_index)+", new_step_index: "+str(step_index+1)+", len(cfg.lr_steps): "+str(len(cfg.lr_steps))+", iteration: "+str(iteration)+", cfg.lr_steps[step_index]: "+str(cfg.lr_steps[step_index]))
                    step_index += 1
                    print("Cur. lr: "+str(cur_lr))
                    set_lr(optimizer, args.lr * (args.gamma ** step_index))
                    print("New lr: "+str(cur_lr))

                
                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()

                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
                losses = net(datum)
                
                losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
                loss = sum([losses[k] for k in losses])
                
                # no_inf_mean removes some components from the loss, so make sure to backward through all of it
                # all_loss = sum([v.mean() for v in losses.values()])

                # Backprop
                loss.backward() # Do this to free up vram even if loss is not finite
                if torch.isfinite(loss).item():
                    optimizer.step()
                
                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_time  = time.time()
                elapsed   = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                if iteration % 10 == 0:
                    eta_str = str(datetime.timedelta(seconds=(cfg.max_iter-iteration) * time_avg.get_avg())).split('.')[0]
                    
                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])
                    
                    print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                            % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)

                if args.log:
                    precision = 5
                    loss_info = {k: round(losses[k].item(), precision) for k in losses}
                    loss_info['T'] = round(loss.item(), precision)

                    if args.log_gpu:
                        log.log_gpu_stats = (iteration % 10 == 0) # nvidia-smi is sloooow
                        
                    log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                        lr=round(cur_lr, 10), elapsed=elapsed)

                    log.log_gpu_stats = args.log_gpu
                
                iteration += 1

                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, cfg.name)

                    print('Saving state, iter:', iteration)
                    yolact_net.save_weights(save_path(epoch, iteration))

                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            os.remove(latest)
            
            # This is done per epoch
            if args.validation_epoch > 0:
                if epoch % args.validation_epoch == 0:
                    compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)

                    # wandb loss and time
                    wandb.log({"Box Location Loss": loss_labels[1]}, commit=False)
                    wandb.log({"Class Confidence Loss": loss_labels[3]}, commit=False)
                    wandb.log({"Mask Loss": loss_labels[5]}, commit=False)
                    wandb.log({"Semantic Segmentation Loss": loss_labels[7]}, commit=False)
                    wandb.log({"Sum of the mean of the losses": total}, commit=False)
                    wandb.log({"Time per batch": elapsed*10}, commit=False)


        # Compute validation mAP after training is finished
        compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)

    except KeyboardInterrupt:

        print()
        print("Save an interrupt? "+str(args.interrupt))
        if args.interrupt:
            print('Stopping early. Saving network...')
            # Delete previous copy of the interrupted network so we don't spam the weights folder
            SavePath.remove_interrupt(args.save_folder)
            yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        exit()

    yolact_net.save_weights(save_path(epoch, iteration))


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    global cur_lr
    cur_lr = new_lr

def gradinator(x):
    x.requires_grad = False
    return x

def prepare_data(datum, devices:list=None, allocation:list=None):
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation)) # The rest might need more/less
        
        images, (targets, masks, num_crowds) = datum

        cur_idx = 0
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx]  = gradinator(images[cur_idx].to(device))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                masks[cur_idx]   = gradinator(masks[cur_idx].to(device))
                cur_idx += 1

        if cfg.preserve_aspect_ratio:
            # Choose a random size from the batch
            _, h, w = images[random.randint(0, len(images)-1)].size()

            for idx, (image, target, mask, num_crowd) in enumerate(zip(images, targets, masks, num_crowds)):
                images[idx], targets[idx], masks[idx], num_crowds[idx] \
                    = enforce_size(image, target, mask, num_crowd, w, h)
        
        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(4)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx]    = torch.stack(images[cur_idx:cur_idx+alloc], dim=0)
            split_targets[device_idx]   = targets[cur_idx:cur_idx+alloc]
            split_masks[device_idx]     = masks[cur_idx:cur_idx+alloc]
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx+alloc]

            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds

def no_inf_mean(x:torch.Tensor):
    """
    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """

    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()

def compute_validation_loss(net, data_loader, criterion):
    global loss_types

    with torch.no_grad():
        losses = {}
        
        # Don't switch to eval mode because we want to get losses
        iterations = 0
        for datum in data_loader:
            images, targets, masks, num_crowds = prepare_data(datum)
            out = net(images)

            wrapper = ScatterWrapper(targets, masks, num_crowds)
            _losses = criterion(out, wrapper, wrapper.make_mask())
            
            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            iterations += 1
            if args.validation_size <= iterations * args.batch_size:
                break
        
        for k in losses:
            losses[k] /= iterations
            
        
        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)

def compute_validation_map(epoch, iteration, yolact_net, dataset, log:Log=None):
    with torch.no_grad():
        yolact_net.eval()
        
        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(yolact_net, dataset, train_mode=True)
        end = time.time()

        # wandb graphics - mAP
        wandb.log({"mAP mean box": val_info['box']['all']}, commit=False)
        wandb.log({"mAP .50 box": val_info['box'][50]}, commit=False)
        wandb.log({"mAP .55 box": val_info['box'][55]}, commit=False)
        wandb.log({"mAP .60 box": val_info['box'][60]}, commit=False)
        wandb.log({"mAP .65 box": val_info['box'][65]}, commit=False)
        wandb.log({"mAP .70 box": val_info['box'][70]}, commit=False)
        wandb.log({"mAP .75 box": val_info['box'][75]}, commit=False)
        wandb.log({"mAP .80 box": val_info['box'][80]}, commit=False)
        wandb.log({"mAP .85 box": val_info['box'][85]}, commit=False)
        wandb.log({"mAP .90 box": val_info['box'][90]}, commit=False)
        wandb.log({"mAP .95 box": val_info['box'][95]}, commit=False)

        wandb.log({"mAP mean mask": val_info['mask']['all']}, commit=False)
        wandb.log({"mAP .50 mask": val_info['mask'][50]}, commit=False)
        wandb.log({"mAP .55 mask": val_info['mask'][55]}, commit=False)
        wandb.log({"mAP .60 mask": val_info['mask'][60]}, commit=False)
        wandb.log({"mAP .65 mask": val_info['mask'][65]}, commit=False)
        wandb.log({"mAP .70 mask": val_info['mask'][70]}, commit=False)
        wandb.log({"mAP .75 mask": val_info['mask'][75]}, commit=False)
        wandb.log({"mAP .80 mask": val_info['mask'][80]}, commit=False)
        wandb.log({"mAP .85 mask": val_info['mask'][85]}, commit=False)
        wandb.log({"mAP .90 mask": val_info['mask'][90]}, commit=False)
        wandb.log({"mAP .95 mask": val_info['mask'][95]}, commit=True)


        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        yolact_net.train()

def setup_eval():
    eval_script.parse_args(['--no_bar', '--max_images='+str(args.validation_size)])

if __name__ == '__main__':
    train()
