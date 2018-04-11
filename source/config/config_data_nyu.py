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

# Project specific
import data.basetypes as basetypes
from data.LoaderFactory import DatasetType
from data.NyuHandPoseDataset import NyuAnnoType


args_data = basetypes.Arguments()

args_data.dataset_type = DatasetType.NYU

# Change to point to the original NYU dataset
args_data.nyu_data_basepath = "/path/to/NyuDataset/original_data"

# If a "cache" should be used (=> faster loading/training), change the path
args_data.use_pickled_cache = True
args_data.nyu_data_basepath_pickled = "/path/to/NyuDataset/original_data_pickled"

args_data.input_cam_id_train = 1
args_data.input_cam_id_test = 1

# frame IDs
args_data.id_start_train, args_data.id_end_train = 29116, 72756     # 0-based IDs (largest part with approx. same setup)
args_data.id_start_val, args_data.id_end_val = 0, 2439              # 0-based IDs
args_data.id_start_test, args_data.id_end_test = 0, 8251            # 0-based IDs (all test)
# Samples in full train set (i.e., indexable)
args_data.num_all_samples_train = 72757
# Number of used samples from validation set
args_data.max_val_train_ratio = 0.3   # ratio of validation samples over train samples
args_data.max_num_samples_val = 2440  # maximum number of used validation samples

# args_data.anno_type=NyuHandPoseDataset.AnnoType.ALL_JOINTS
args_data.anno_type = NyuAnnoType.EVAL_JOINTS_ORIGINAL

args_data.num_joints = args_data.anno_type

args_data.in_crop_size = (64, 64)
args_data.out_crop_size = (64, 64)
args_data.do_jitter_com = [True, False, False]
args_data.do_jitter_com_test = [False, False, False]
args_data.sigma_com = [10., 0., 0.]       # in millimeter
args_data.do_add_white_noise = [True, False, False]
args_data.do_add_white_noise_test = [False, False, False]
args_data.sigma_noise = [5., 0., 0.]     # in millimeter

# Minimum ratio of the detected hand crop which have to be inside image boundaries
args_data.min_ratio_inside = 0.3

# Value normalization; if True \in [0,1], else \in [-1,1]
args_data.do_norm_zero_one = False

# Whether to use the real or synthetic data
args_data.do_use_real_samples = True
