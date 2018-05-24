"""
Collected dataset configuration

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

##############################################################################
# Definition/Convention:
# the first cam (e.g. in a list of two cams, or cam-ID 1) 
# is the right cam from user perspective, i.e., left from camera perspective
##############################################################################

# Project specific
import data.basetypes as basetypes
from data.LoaderFactory import DatasetType
from util.camera_definitions import *

# Other libs
import numpy as np

# builtin
import os


args_data = basetypes.Arguments()

args_data.dataset_type = DatasetType.ICG

# Change args_data.base_path_true to point to the data
args_data.base_path_true = "/path/to/mvhands/data/"

# Don't change this path (its used for simplicity, since there are absolute paths in textfiles)
args_data.base_path_in_files = "/media/seagatehdd/ICG/Datasets/HandPose/IGT"

# Unlabeled data
args_data.icg_unlabeled_files_list = [
    os.path.join(args_data.base_path_true, "Gener8_20161003/all_valid_files.txt"),
    os.path.join(args_data.base_path_true, "Gener8_20161116/all_valid_files.txt"),
    os.path.join(args_data.base_path_true, "Gener8_20170112/all_valid_files.txt"),
    os.path.join(args_data.base_path_true, "Gener8_20170206/all_valid_files.txt")
    ]
# Camera parameters corresponding to icg_unlabeled_files_list
args_data.icg_unlabeled_data_intrinsics_list = [
    [iCalibCam1221, iCalibCam1819],
    [iCalibCam1819, iCalibCam1221],
    [iCalibCam1819, iCalibCam1221],
    [iCalibCam0819, iCalibCam1617]
    ]
args_data.icg_unlabeled_data_R_list = [
    [R1_0310, R2_0310],
    [R1_1611, R2_1611],
    [R1_1201, R2_1201],
    [R1_0602, R2_0602]
    ]
args_data.icg_unlabeled_data_t_list = [
    [t1_0310, t2_0310],
    [t1_1611, t2_1611],
    [t1_1201, t2_1201],
    [t1_0602, t2_0602]
    ]
    
# Define train, val., test sets for unlabeled data
args_data.list_indices_unlabeled_train  = [0,1,3]
args_data.list_indices_unlabeled_val    = [2]
args_data.list_indices_unlabeled_test   = []
    
# Labeled data
basepath_icg_anno_train = os.path.join(args_data.base_path_true, "Annotation/Selection01")
args_data.icg_labeled_files_list_leftcam = [
    # 1003
    os.path.join(basepath_icg_anno_train, "AT_Pose_A_A03LeftIndeCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_A_A03LeftIThuCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_F_A03LeftClosCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_F_A03LeftSplaCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_F_A03RighIndeCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_F_A03RighIThuCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_F_A03RighSplaCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_F_A03RighStndCam1221.txt"),
    # for val.
    os.path.join(basepath_icg_anno_train, "AT_Pose_F_A03RighClosCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_F_A03LeftStndCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_A_A03TestFreeCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_3Dok_A_A03TestFreeCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_2cam_A_A03TestFreeCam1221.txt"),
    # 0112
    os.path.join(basepath_icg_anno_train, "AT_Pose_G_A01LeftClosCam1819.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_G_A01LeftFistCam1819.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_G_A01LeftStndCam1819.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_G_A01RighClosCam1819.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_G_A01RighFistCam1819.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_G_A01RighIndeCam1819.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_G_A01RighStndCam1819.txt"),
    # 0206
    os.path.join(basepath_icg_anno_train, "AT_Pose_20170206_G_A01LeftStndCam0819.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_20170206_G_A01LeftIndeCam0819.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_20170206_G_A01RighStndCam0819.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_20170206_G_A02RighIndeCam0819.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_20170206_G_A02LeftStndCam0819.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_20170206_G_A02LeftIndeCam0819.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_20170206_G_A03LeftIndeCam0819.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_20170206_G_A03RighStndCam0819.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_20170206_G_A03LeftStndCam0819.txt"),
    # for val
    os.path.join(basepath_icg_anno_train, "AT_Pose_20170206_G_A01LeftClosCam0819.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_20170206_G_A02RighStndCam0819.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_20170206_G_A03RighIndeCam0819.txt"),
    # 1003 Test set 
    os.path.join(basepath_icg_anno_train, "AT_Pose_S_A04RighIThuCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_S_A04LeftStndCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_S_A04RighSplaCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_A_A04LeftIndeCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_A_A04LeftIThuCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_S_A04LeftClosCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_S_A04LeftSplaCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_S_A04RighClosCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_S_A04RighIndeCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_S_A04RighStndCam1221.txt"),
    #
    os.path.join(basepath_icg_anno_train, "AT_Pose_A_A04TestFreeCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_A_A05TestFreeCam1221(left).txt"),
    os.path.join(basepath_icg_anno_train, "AT_Pose_A_A05TestFreeCam1221.txt"),
    # 1003 Test set (propagated, being checked/"ok" in 3D (3Dok), or checked/"ok" in 2. cam (2cam))
    os.path.join(basepath_icg_anno_train, "AT_3Dok_A_A04TestFreeCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_3Dok_A_A05TestFreeCam1221(left).txt"),
    os.path.join(basepath_icg_anno_train, "AT_3Dok_A_A05TestFreeCam1221.txt"),
    os.path.join(basepath_icg_anno_train, "AT_2cam_A_A04TestFreeCam1221.txt")
    ]
    
# Corresponding second cams and camera params (to icg_labeled_files_list_leftcam)
args_data.icg_labeled_data_camnames = []
args_data.icg_labeled_data_intrinsics_list = []
args_data.icg_labeled_data_R_list = []
args_data.icg_labeled_data_t_list = []
for i in range(13):     # for 1003
    args_data.icg_labeled_data_camnames.append(["Cam_0005-1207-0034-1221", "Cam_0005-1207-0034-1819"])
    args_data.icg_labeled_data_intrinsics_list.append([iCalibCam1221, iCalibCam1819])
    args_data.icg_labeled_data_R_list.append([R1_0310, R2_0310])
    args_data.icg_labeled_data_t_list.append([t1_0310, t2_0310])
for i in range(13,20):  # for 0112
    args_data.icg_labeled_data_camnames.append(["Cam_0005-1207-0034-1819", "Cam_0005-1207-0034-1221"])
    args_data.icg_labeled_data_intrinsics_list.append([iCalibCam1819, iCalibCam1221])
    args_data.icg_labeled_data_R_list.append([R1_1201, R2_1201])
    args_data.icg_labeled_data_t_list.append([t1_1201, t2_1201])
for i in range(20,32):  # for 0206
    args_data.icg_labeled_data_camnames.append(["Cam_0005-1207-0034-0819", "Cam_0005-1207-0034-1617"])
    args_data.icg_labeled_data_intrinsics_list.append([iCalibCam0819, iCalibCam1617])
    args_data.icg_labeled_data_R_list.append([R1_0602, R2_0602])
    args_data.icg_labeled_data_t_list.append([t1_0602, t2_0602])
for i in range(32,49):  # for 1003 Test set
    args_data.icg_labeled_data_camnames.append(["Cam_0005-1207-0034-1221", "Cam_0005-1207-0034-1819"])
    args_data.icg_labeled_data_intrinsics_list.append([iCalibCam1221, iCalibCam1819])
    args_data.icg_labeled_data_R_list.append([R1_0310, R2_0310])
    args_data.icg_labeled_data_t_list.append([t1_0310, t2_0310])
    
# Define train, val., test sets for labeled data
#args_data.list_indices_labeled_train  = range(0,8) + range(13,29)
args_data.list_indices_labeled_train  = range(0,32)     # use val.set also for training (=> no model selection based on val.set)
args_data.list_indices_labeled_val    = range(8,13) + range(29,32)
args_data.list_indices_labeled_test   = range(32,49)

args_data.input_cam_id_train = 1
args_data.input_cam_id_test = 1

args_data.max_num_labeled_samples = np.inf
# Ratio of validation set to train set (to simulate realistic scenario with small amount of labeled data)
# (=> makes usually no sense for this dataset => hence, set to high value)
args_data.max_val_train_ratio = 10.3

# Only use labeled data for validation or test
args_data.do_val_only_with_labeled = True
args_data.do_test_only_with_labeled = True

args_data.num_joints = 24

args_data.in_crop_size = (64, 64)
args_data.out_crop_size = (64, 64)
args_data.do_jitter_com = [True, False]
args_data.do_jitter_com_test = [False, False]
args_data.sigma_com = [10., 0.]       # in millimeter
args_data.do_add_white_noise = [True, False]
args_data.do_add_white_noise_test = [False, False]
args_data.sigma_noise = [5., 0.]     # in millimeter

# Minimum ratio of the detected hand crop which have to be inside image boundaries
# only check for a minimum here, filelist should only contain valid samples (but jitter not considered)
args_data.min_ratio_inside = 0.3

# Value normalization; if True \in [0,1], else \in [-1,1]
args_data.do_norm_zero_one = False
