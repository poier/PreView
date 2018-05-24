"""
Copyright 2015, 2018 ICG, Graz University of Technology

This file is part of PreView.

PreView is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PreView is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PreView.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import print_function
import cv2 # to be imported before torch (for me; so that libgomp is loaded from the system installation)
import torch
import torch.utils.data
from torch.autograd import Variable

# Project specific
import nets.NetFactory as NetFactory
from nets.NetFactory import NetType
from trainer.SuperTrainer import SuperTrainer, TrainingType
import eval.handpose_evaluation as hape_eval
import data.LoaderFactory as LoaderFactory
from data.LoaderFactory import LoaderMode, DatasetType
from util.output import imwrite
from util.initialization import weights_init
from util.argparse_helper import parse_arguments_generic

from pycrayon import CrayonClient
import torchvision

import numpy as np
import cPickle

import os.path


#%% Set configuration
# General parameters
from config.config import args
# Data specific parameters
from config.config_data_nyu import args_data
# Uncomment the following line to use the MV-hands dataset
#from config.config_data_icg import args_data

# Merge different configuration parameters into single object
args.__dict__ = dict(args.__dict__.items() + args_data.__dict__.items())

# Parse command-line arguments
args = parse_arguments_generic(args)

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.net_type == NetType.POSE_ESTIMATION:
    # pose estimation is solely trained with labeled data
    args.min_samp_prob = 1.0
    args.min_sampling_prob_labeled = 1.0

# Set seed for random number generators
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
args.rng = np.random.RandomState(args.seed)

# Assemble (output-) paths
parent_dir = os.path.dirname(os.path.realpath(__file__))
if not args.out_base_path == "":
    parent_dir = args.out_base_path
args.out_path_results = os.path.join(parent_dir, args.out_path, args.exp_name)
if args.model_filepath == "":
    model_file_default = "model/model.mdl"
    args.model_filepath = os.path.join(args.out_path_results, model_file_default)
args.out_path_results_config = os.path.join(args.out_path_results, args.out_subdir_config)
args.out_path_results_images = os.path.join(args.out_path_results, args.out_subdir_images)


#%% Create directories (if necessary)
if not os.path.exists(args.out_path_results):
    os.makedirs(args.out_path_results)
if not os.path.exists(args.out_path_results_config):
    os.makedirs(args.out_path_results_config)
if not os.path.exists(args.out_path_results_images):
    os.makedirs(args.out_path_results_images)
    
    
#%% Save parameters
filepath_args = os.path.join(args.out_path_results, args.out_subdir_config, "args.pkl")
cPickle.dump(args, open(filepath_args, "wb"), protocol=cPickle.HIGHEST_PROTOCOL)
    

#%% Setup data loaders
if args.do_train:
    train_loader = LoaderFactory.create_dataloader(LoaderMode.TRAIN, args,
                                                   do_use_gpu=args.cuda)
if args.do_train or args.do_produce_previews:
    val_loader = LoaderFactory.create_dataloader(LoaderMode.VAL, args,
                                                 do_use_gpu=args.cuda)
if args.do_test:
    test_loader = LoaderFactory.create_dataloader(LoaderMode.TEST, args,
                                                  do_use_gpu=args.cuda)


#%% Create logger
if args.do_train:
    # Connect Crayon Logger (TensorBoard "wrapper") to the server
    cc = CrayonClient(hostname="localhost", port=8889)
    tb_log_exp_name = args.exp_name
    # Remove previous experiment
    try:
        cc.remove_experiment(tb_log_exp_name)
    except ValueError:
        # experiment doesn't already exist - nothing to be done here
        print("Experiment '{}' didn't exist already (nothing to be done).".format(\
            tb_log_exp_name))
    # Create a new experiment
    tb_log = cc.create_experiment(tb_log_exp_name)


#%% Train (Load) model
# Create and init predictor
model = NetFactory.create_net(net_type=args.net_type,
                              params=args, 
                              num_prior_dims=args.num_embedding_dims, 
                              num_cond_dims=args.num_cond_dims, 
                              num_joints=args.num_joints, 
                              num_features=args.num_features)
model.apply(weights_init)
if args.cuda:
    model.cuda()

# Create and init discriminator
if TrainingType.is_training_type_adversarial(args.training_type):
    model_d = NetFactory.create_net_discriminator(params=args, num_joints=args.num_joints, 
                                                  num_features=args.num_features)
    model_d.apply(weights_init)
    if args.cuda:
        model_d.cuda()
else:
    model_d = None


if args.do_train:
    trainer = SuperTrainer()
    trainer.train(model, train_loader, val_loader, args, tb_log, 
                  model_discriminator=model_d)
    
    # Backup experiment log
    results_log_filepath = os.path.join(args.out_path_results, "crayon_results_log")
    filename = tb_log.to_zip(results_log_filepath)
    print("Stored log in file: {}".format(filename))
    
# Load stored (best/last) model file
descr_str = "best" if args.do_use_best_model else "last"
print("Loading {} model from file...".format(descr_str))
print("  from file {}".format(args.model_filepath))
model.load_state_dict(torch.load(args.model_filepath))
if args.cuda:
    model.cuda()
    
    
#%% Evaluate
if args.do_test:
    print("Evaluate model on test set...")
    # Evaluate joint positions
    if NetType.is_output_pose(args.net_type):
        targets, predictions, crop_transforms, coms, data = hape_eval.evaluatePytorchModel(
            model, test_loader, do_use_gpu=args.cuda)
        if args.dataset_type == DatasetType.NYU:
            hpe = hape_eval.NYUHandposeEvaluation(targets, predictions, joints=args.num_joints)
        elif args.dataset_type == DatasetType.ICG:
            hpe = hape_eval.ICGHandposeEvaluation(targets, predictions)
        hpe.outputPath = args.out_path_results
        mean_error = hpe.getMeanError()
        max_error = hpe.getMaxError()
        num_train_samples = "?"
        if args.do_train:
            num_train_samples = len(train_loader.sampler)
        print("Train samples: {}, test samples: {}".format(
            num_train_samples, len(test_loader.sampler)))
        print("Mean error: {}mm, max error: {}mm".format(mean_error, max_error))
        print("MD score: {}".format(hpe.getMDscore(80)))
        
        print("{}".format([hpe.getJointMeanError(j) for j in range(targets[0].shape[0])]))
        print("{}".format([hpe.getJointMaxError(j) for j in range(targets[0].shape[0])]))
                
        # Write results to textfile
        out_result_filepath = os.path.join(args.out_path_results, args.out_filename_result)
        hpe.writeResults2Textfile(out_result_filepath)
        
        # Save results
        filepath_est = os.path.join(args.out_path_results, args.out_filename_joint_positions_estimated)
        cPickle.dump(predictions, open(filepath_est, "wb"), protocol=cPickle.HIGHEST_PROTOCOL)
        
        # Save ground truth
        filepath_gt = os.path.join(args.out_path_results, args.out_filename_joint_positions_groundtruth)
        cPickle.dump(targets, open(filepath_gt, "wb"), protocol=cPickle.HIGHEST_PROTOCOL)
        
        hpe.plotEvaluation(args.exp_name, methodName='Ours')
        
    
#%% Produce some predictions for test/validation samples
if args.do_produce_previews and (NetType.get_num_output_views(args.net_type) > 0):
    model.eval()
    num_images_to_show = 32
    padding = 0
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    args.rng = np.random.RandomState(args.seed)
    
    test_iter = iter(val_loader)
    data_cam1, data_cam2, data_cam3,_,_,_,_,_,_, com_cam1,_,_ = test_iter.next()
    list_camviews = [data_cam1, data_cam2, data_cam3]
    # Select input image
    data_input = data_cam1
    com_input = torch.from_numpy(
                    val_loader.dataset.normalize_3D(com_cam1.numpy()))
    # Select target image
    data_target = list_camviews[args.output_cam_ids_test[0] - 1] # 1-based IDs
    
    if args.cuda:
        data_input, data_target = data_input.cuda(), data_target.cuda()
        com_input = com_input.cuda()
    data_input = Variable(data_input, volatile=True)
    data_target = Variable(data_target, volatile=True)
    com_input = Variable(com_input, volatile=True)
    # Do prediction
    if NetType.is_output_pose(args.net_type):
        if NetType.get_num_output_views(args.net_type) == 1:
            _, img_pred, _ = model(data_input, com_input)
        else:
            raise UserWarning("not implemented (trying to show samples for \
                prediction of multiple views and pose)")
    else:
        if NetType.get_num_output_views(args.net_type) == 1:
            img_pred, _ = model(data_input, com_input)
        elif NetType.get_num_output_views(args.net_type) == 2:
            img_pred, img_pred2, _ = model(data_input, com_input)
        
    # Extract samples 
    input_cam1 = data_input.data.cpu()[:num_images_to_show]
    input_cam2 = data_target.data.cpu()[:num_images_to_show]
    estim = img_pred.data.cpu().view(-1,1,args.out_crop_size[0],args.out_crop_size[1])[:num_images_to_show]
    if NetType.get_num_output_views(args.net_type) == 2:
        estim2 = img_pred2.data.cpu().view(-1,1,args.out_crop_size[0],args.out_crop_size[1])[:num_images_to_show]
    # Assemble samples 
    input_aligned_image_grid = torchvision.utils.make_grid(input_cam1, padding=padding)[:,1:,1:]
    target_aligned_image_grid = torchvision.utils.make_grid(input_cam2, padding=padding)[:,1:,1:]
    estimate_aligned_image_grid = torchvision.utils.make_grid(estim, padding=padding)[:,1:,1:]
    if NetType.get_num_output_views(args.net_type) == 2:
        estimate_aligned_image_grid2 = torchvision.utils.make_grid(estim2, padding=padding)[:,1:,1:]
    # Write samples 
    imwrite(input_aligned_image_grid, 
            os.path.join(args.out_path_results, "test_sample_input.png"), 
            do_normalize=True)
    imwrite(target_aligned_image_grid, 
            os.path.join(args.out_path_results, "test_sample_target.png"), 
            do_normalize=True)
    imwrite(estimate_aligned_image_grid, 
            os.path.join(args.out_path_results, "test_sample_estimate.png"), 
            do_normalize=True)
    if NetType.get_num_output_views(args.net_type) == 2:
        imwrite(estimate_aligned_image_grid2, 
                os.path.join(args.out_path_results, "test_sample_estimate_view2.png"), 
                do_normalize=True)
    
print("Finished experiment.")
