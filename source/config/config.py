"""
Configuration

Copyright 2018 ICG, Graz University of Technology

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

# Project specific
import data.basetypes as basetypes


args = basetypes.Arguments()

# Number of additional input dimensions for the decoder
args.num_cond_dims = 3
# Number of (base-) feature channels of the model (this is usually increased after downsampling)
args.num_features = 64
# Use the model with best val. error (over epochs)? if False last model is used
args.do_use_best_model = True

# How many (CPU) workers for loading data
args.num_loader_workers = 5

# Output parameters
args.do_save_model = True
args.save_model_epoch_interval = 10
args.out_filename_result = "results.txt"
args.out_filename_joint_positions_estimated = "results_joint_pos_estimated.pkl"
args.out_filename_joint_positions_groundtruth = "results_joint_pos_gt.pkl"
args.out_subdir_images = "images"
args.out_subdir_config = "config"
