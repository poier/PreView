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

# Project
from detector.handdetector import HandDetectorICG
from util.transformations import transformPoint2D, pointsImgTo3D, pointImgTo3D, \
                                    point3DToImg, transform_points_to_other_cam
from data.basetypes import ICGFrame

# PyTorch
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

# Other Libs
from PIL import Image
import numpy as np

# builtin
import os.path
import time
import copy


class IcgHandPoseMultiViewDataset(Dataset):
    """
    Provides functionality to load data from two camera dataset created by ICG
    """

    def __init__(self, args_data):
        """
        Constructor
        
        Arguments:
            args_data (basetypes.DataArguments) collection of parameters
        """
        print("Init ICG Dataset...")
        
        self.joint_id_com = 10
        self.depth_sub_dir = "DepthImagesDistorted"
        self.gray_sub_dir = "GrayImagesDistorted"
        self.min_depth_cam = 50
        self.max_depth_cam = 900
        self.min_ir_value = 50
        self.background_value = 10000
        self.do_normalize_com = True
        
        self.args_data = copy.deepcopy(args_data)
        
        # Load list of labeled samples and annotations
        self.labeled_files = self.load_labeled_filelist(
                                self.args_data.labeled_files_list_leftcam_cur,
                                self.args_data.labeled_data_intrinsics_list_cur,
                                self.args_data.labeled_data_R_list_cur,
                                self.args_data.labeled_data_t_list_cur,
                                self.args_data.labeled_data_camnames_cur,
                                self.args_data.base_path_in_files,
                                self.args_data.base_path_true)
                            
        # Load list of unlabeled samples
        self.unlabeled_files = self.load_unlabeled_filelist(
                                self.args_data.unlabeled_files_list_cur,
                                self.args_data.base_path_in_files,
                                self.args_data.base_path_true)
                            
        self.num_samples_labeled = len(self.labeled_files)
        self.num_samples_unlabeled = len(self.unlabeled_files)
        self.num_samples = self.num_samples_labeled + self.num_samples_unlabeled
        
        # Precomputations for normalization of 3D point
        self.precompute_normalization_factors()
        
        print("ICG Dataset init done.")
        
        
    def __getitem__(self, index):
        trfm = transforms.ToTensor()
        
        if index < self.num_samples_labeled:
            # labeled sample
            is_labeled = True
            ind = index
            sample_data_allcams = self.labeled_files[ind]
            id_camparams = self.labeled_files[ind][0].id_camparams
            intrinsics = self.args_data.labeled_data_intrinsics_list_cur[id_camparams]
        else:
            # unlabeled sample
            is_labeled = False
            ind = index - self.num_samples_labeled
            sample_data_allcams = self.unlabeled_files[ind]
            id_camparams = self.unlabeled_files[ind][0].id_camparams
            intrinsics = self.args_data.unlabeled_data_intrinsics_list_cur[id_camparams]
            
        if self.args_data.do_load_cam1:
            camid = 0
            fx, fy, cx, cy, k1, k2, p1, p2, k3 = intrinsics[camid]
            
            # Load sample
            sample_cam1 = load_sample_icg(sample_data_allcams[camid], 
                                         fx=fx, fy=fy, cx=cx, cy=cy, 
                                         crop_size_mm=self.args_data.crop_size_3d_tuple, 
                                         crop_size_px=self.args_data.in_crop_size,
                                         is_labeled=is_labeled, 
                                         num_joints=self.args_data.num_joints,
                                         joint_id_com=self.joint_id_com,
                                         rng=self.args_data.rng,
                                         do_jitter_com=self.args_data.do_jitter_com_cur[camid],
                                         sigma_com=self.args_data.sigma_com[camid],
                                         do_add_white_noise=self.args_data.do_white_noise_cur[camid],
                                         sigma_noise=self.args_data.sigma_noise[camid],
                                         min_depth_cam=self.min_depth_cam,
                                         max_depth_cam=self.max_depth_cam, 
                                         min_ir_value=self.min_ir_value, 
                                         background_value=self.background_value,
                                         depth_sub_dir=self.depth_sub_dir, 
                                         gray_sub_dir=self.gray_sub_dir,
                                         min_ratio_inside=self.args_data.min_ratio_inside)
                                            
            # Normalize
            if self.args_data.do_norm_zero_one:
                img_cam1, target_cam1 = normalize_zero_one(sample_cam1)
            else:
                img_cam1, target_cam1 = normalize_minus_one_one(sample_cam1)
                
            # Convert to suitable format (dims, Tensor)
            img_cam1 = np.expand_dims(img_cam1, axis=2) * 255.0     # Image need to be HxWxC and \in [0,255] for transform.ToTensor()
            img_cam1 = trfm(img_cam1)
            
            target_cam1 = torch.from_numpy(target_cam1.astype('float32'))
            transform_crop_cam1 = torch.from_numpy(sample_cam1.T)
            com_cam1 = torch.from_numpy(sample_cam1.com_3D)
        else:
            img_cam1 = []
            target_cam1 = []
            transform_crop_cam1 = []
            com_cam1 = []
                                     
        if self.args_data.do_load_cam2:
            camid = 1
            fx, fy, cx, cy, k1, k2, p1, p2, k3 = intrinsics[camid]
            
            # Load sample
            sample_cam2 = load_sample_icg(sample_data_allcams[camid], 
                                         fx=fx, fy=fy, cx=cx, cy=cy, 
                                         crop_size_mm=self.args_data.crop_size_3d_tuple, 
                                         crop_size_px=self.args_data.in_crop_size,
                                         is_labeled=is_labeled, 
                                         num_joints=self.args_data.num_joints,
                                         joint_id_com=self.joint_id_com,
                                         rng=self.args_data.rng,
                                         do_jitter_com=self.args_data.do_jitter_com_cur[camid],
                                         sigma_com=self.args_data.sigma_com[camid],
                                         do_add_white_noise=self.args_data.do_white_noise_cur[camid],
                                         sigma_noise=self.args_data.sigma_noise[camid],
                                         min_depth_cam=self.min_depth_cam,
                                         max_depth_cam=self.max_depth_cam, 
                                         min_ir_value=self.min_ir_value, 
                                         background_value=self.background_value,
                                         depth_sub_dir=self.depth_sub_dir, 
                                         gray_sub_dir=self.gray_sub_dir,
                                         min_ratio_inside=self.args_data.min_ratio_inside)
                                            
            # Normalize
            if self.args_data.do_norm_zero_one:
                img_cam2, target_cam2 = normalize_zero_one(sample_cam2)
            else:
                img_cam2, target_cam2 = normalize_minus_one_one(sample_cam2)
                
            # Convert to suitable format (dims, Tensor)
            img_cam2 = np.expand_dims(img_cam2, axis=2) * 255.0     # Image need to be HxWxC and \in [0,255] for transform.ToTensor()
            img_cam2 = trfm(img_cam2)
            
            target_cam2 = torch.from_numpy(target_cam2.astype('float32'))
            transform_crop_cam2 = torch.from_numpy(sample_cam2.T)
            com_cam2 = torch.from_numpy(sample_cam2.com_3D)
        else:
            img_cam2 = []
            target_cam2 = []
            transform_crop_cam2 = []
            com_cam2 = []
        
        return img_cam1, img_cam2, [], target_cam1, target_cam2, [], \
            transform_crop_cam1, transform_crop_cam2, [], \
            com_cam1, com_cam2, []


    def __len__(self):
        return self.num_samples
        
        
    def get_num_samples_labeled(self):
        return self.num_samples_labeled
        
        
    def get_num_samples_unlabeled(self):
        return self.num_samples_unlabeled
        

    def get_type_string(self):
        return "ICG"
            
        
    def load_labeled_filelist(self, filelist, cam_intrinsics_list, cam_R_list, 
                              cam_t_list, camnames_list,
                              base_path_in_files, base_path_true):
        """
        Reads all labeled files from each file in filelist
        (and assemble together with corresponding file from second camera)
        """
        t_0 = time.clock()
            
        data = []
        for li in range(len(filelist)):
            fx1, fy1, cx1, cy1, k1c1, k2c1, p1c1, p2c1, k3c1 = cam_intrinsics_list[li][0]
            fx2, fy2, cx2, cy2, k1c2, k2c2, p1c2, p2c2, k3c2 = cam_intrinsics_list[li][1]
            R_cam1 = cam_R_list[li][0]
            R_cam2 = cam_R_list[li][1]
            t_cam1 = cam_t_list[li][0]
            t_cam2 = cam_t_list[li][1]
            camname1 = camnames_list[li][0]
            camname2 = camnames_list[li][1]
            
            filename_txt = filelist[li]
            with open(filename_txt) as f:
                lines = f.readlines()
                f.seek(0)
                for i in range(len(lines)):
                    line = lines[i]
                    # Extract filename and annotation
                    parts = line.split(' ')
                    filename_img1 = parts[0]
                    filename_img1 = self.exchange_substring(filename_img1, 
                                                            base_path_in_files, base_path_true)
                    gt_uvd_cam1 = self.parse_annotation_string(parts[1:])
                    gt_3D_cam1 = pointsImgTo3D(gt_uvd_cam1, fx1, fy1, cx1, cy1)
                    # Assemble filename of second cam image
                    filename_img2 = self.exchange_substring(filename_img1, 
                                                            camname1, camname2)
                    # Transform annotation to other cam
                    gt_uvd_cam2 = transform_points_to_other_cam(gt_uvd_cam1, 
                                    cam_intrinsics_list[li][0], R_cam1, t_cam1,
                                    cam_intrinsics_list[li][1], R_cam2, t_cam2)
                    gt_3D_cam2 = pointsImgTo3D(gt_uvd_cam2, fx2, fy2, cx2, cy2)
                    
                    data.append([
                                ICGFrame(None, gt_uvd_cam1, None, None, 
                                         gt_3D_cam1, None, None, filename_img1,
                                         None, None, li),
                                ICGFrame(None, gt_uvd_cam2, None, None, 
                                         gt_3D_cam2, None, None, filename_img2,
                                         None, None, li)
                                ])
                
        t_1 = time.clock()
        print("  labeled: found {} sample(s) in {} file(s) in {}sec.".format(
            len(data), len(filelist), (t_1 - t_0)))
        
        return data
        
        
    def load_unlabeled_filelist(self, filelist, base_path_in_files, base_path_true):
        """
        Loads all labled files from each file in filelist
        """
        t_0 = time.clock()
        
        data = []
        for li in range(len(filelist)):
            filename_txt = filelist[li]
            with open(filename_txt) as f:
                lines = f.readlines()
                f.seek(0)
                for i in range(len(lines)):
                    line = lines[i]
                    # Extract filename and annotation
                    parts = line.split(' ')
                    filename_img1 = parts[0]
                    filename_img1 = self.exchange_substring(filename_img1, 
                                                            base_path_in_files, base_path_true)
                    camname1 = parts[1]
                    com1_string = parts[2:5]
                    camname2 = parts[5]
                    com2_string = parts[6:9]
                    com1 = self.parse_com_string(com1_string)
                    com2 = self.parse_com_string(com2_string)
                    # Assemble filename of second cam image
                    filename_img2 = self.exchange_substring(filename_img1, 
                                                            camname1, camname2)
                    
                    data.append([
                                ICGFrame(None, None, None, None, 
                                         None, None, com1, filename_img1, 
                                         None, None, li),
                                ICGFrame(None, None, None, None, 
                                         None, None, com2, filename_img2, 
                                         None, None, li)
                                ])
        
        t_1 = time.clock()
        print("  unlabeled: found {} sample(s) in {} file(s) in {}sec.".format(
            len(data), len(filelist), (t_1 - t_0)))
        
        return data
            
            
    def parse_annotation_string(self, annostring):
        """
        Parse string specifying the joint positions
        """
        anno = np.zeros((self.args_data.num_joints,3), np.float32)
        for j in range(self.args_data.num_joints):
            for xyz in range(0, 3):
                anno[j,xyz] = annostring[j*3+xyz]
                
        return anno
        
        
    def parse_com_string(self, comstring):
        """
        Parse string specifying the com (center of mass)
        """
        com = np.zeros((3), np.float32)
        com[0] = comstring[0]
        com[1] = comstring[1]
        com[2] = comstring[2]
        
        return com
        
        
    def exchange_substring(self, string_orig, substring_remove, substring_include):
        """
        Find a given sub-string within a string and exchange it for a new sub-string
        """
        string_new = string_orig
        i_start  = string_orig.find(substring_remove)
        i_end    = i_start + len(substring_remove)
        if i_start >= 0:
            string_new = string_orig[0:i_start] + substring_include + string_orig[i_end:]
        else:
            UserWarning("String to be exchanged not found in exchange_substring()")
            
        return string_new
        
        
    def precompute_normalization_factors(self):
        min_depth_in = self.min_depth_cam
        max_depth_in = self.max_depth_cam
        
        if self.args_data.do_norm_zero_one:
            depth_range_out = 1.
            self.norm_min_out = 0.
        else:
            depth_range_out = 2.
            self.norm_min_out = -1.
            
        depth_range_in = float(max_depth_in - min_depth_in)
        
        self.norm_max_out = 1.
        self.norm_min_in = min_depth_in
        self.norm_scale_3Dpt_2_norm = depth_range_out / depth_range_in

    
    def normalize_3D(self, points_3D):
        """
        Normalize depth to a desired range; x and y are normalized accordingly
        range for x,y is double the depth range 
        This essentially assumes only positive z, but pos/neg x and y as input
        
        Arguments:
            points_3D (Nx3 numpy array): array of N 3D points
        """
        pt = np.asarray(copy.deepcopy(points_3D), 'float32')
        
        pt[:,0] = pt[:,0] * self.norm_scale_3Dpt_2_norm
        pt[:,1] = pt[:,1] * self.norm_scale_3Dpt_2_norm
        pt[:,2] = (pt[:,2] - self.norm_min_in) * self.norm_scale_3Dpt_2_norm + self.norm_min_out
        
        np.clip(pt, self.norm_min_out, self.norm_max_out, out=pt)
        
        return pt

    
    def normalize_and_jitter_3D(self, points_3D):
        """
        Normalize depth to a desired range; x and y are normalized accordingly
        range for x,y is double the depth range 
        This essentially assumes only positive z, but pos/neg x and y as input
        Additionally noise is added
        
        Arguments:
            points_3D (Nx3 numpy array): array of N 3D points
        """
        # FIXME: encapsulate code (parts), reduce redundancy with this dataset and also with other datasets
        pt = np.asarray(copy.deepcopy(points_3D), 'float32')
        
        # Add noise to original 3D coords (in mm)
        sigma = 15.
        pt += sigma * self.args_data.rng.randn(pt.shape[0], pt.shape[1])
        
        pt[:,0] = pt[:,0] * self.norm_scale_3Dpt_2_norm
        pt[:,1] = pt[:,1] * self.norm_scale_3Dpt_2_norm
        pt[:,2] = (pt[:,2] - self.norm_min_in) * self.norm_scale_3Dpt_2_norm + self.norm_min_out
        
        np.clip(pt, self.norm_min_out, self.norm_max_out, out=pt)
        
        return pt
        
        
    def denormalize_joint_pos(self, joint_pos):
        """
        Re-scale the given joint positions to metric distances (in mm)
        
        Arguments:
            joint_pos (numpy.ndarray): normalized joint positions, 
                as provided by the dataset
            config (optional): config with 'cube' setting (i.e., hand cube size)
        """
        return denormalize_joint_pos(joint_pos, self.args_data.do_norm_zero_one, 
                                     self.args_data.crop_size_3d_tuple)
        
        
def load_depth_map(filepath):
    """
    Load depth map
    
    Arguments:
        filepath (string): full path and filename 
        
    Returns:
        depth map
    """
    with open(filepath) as f:
        img = Image.open(filepath)
        assert len(img.getbands()) == 1
        img_np = np.asarray(img, np.float32)
    
    return img_np
        
        
def load_and_preproc_depth_sample(depth_file_path, 
                   fx, fy, cx, cy, 
                   min_depth=50, max_depth=1000, min_ir=50, background_value=10000,
                   depth_sub_dir="DepthImagesDistorted", 
                   gray_sub_dir="GrayImagesDistorted", 
                   ext_depth=".png"):
    """
    Loads data and prepares corresponding annotations
    :param depth_file_path    full file-path of depth image file
    :param depth_sub_dir      the sub-directory containing the depth file, i.e., 
                            the parent directory of depth file contained in depth_file_path
                            (This is changed to find the corresponding gray-image, pcl, ...)
    """
    # Assemble filename for gray/IR image
    i_start = depth_file_path.find(depth_sub_dir)
    gray_file_path = ''
    if i_start > 0:
        # Gray image
        gray_file_path = depth_file_path[0:i_start] + gray_sub_dir + depth_file_path[(i_start+len(depth_sub_dir)):]
        
    # Ensure that files exist
    assert os.path.isfile(depth_file_path), "(Depth-)File {} does not exist!".format(depth_file_path)
    assert os.path.isfile(gray_file_path), "(Gray/IR-)File {} does not exist! Needed for IGT dataset".format(gray_file_path)

    dpt = load_depth_map(depth_file_path)
    gry = load_depth_map(gray_file_path)
    
    # Pre-processing
    # Get rid of invalid, unwanted points
    dpt[gry < min_ir] = background_value   
    dpt[dpt > max_depth] = background_value
    dpt[dpt < min_depth] = background_value
    
    return dpt
    
    
def load_sample_icg(sample_data, 
                    fx, fy, cx, cy, 
                    crop_size_mm, crop_size_px,
                    is_labeled, num_joints, joint_id_com,
                    rng, do_jitter_com, sigma_com, do_add_white_noise, sigma_noise,
                    min_depth_cam, max_depth_cam, min_ir_value, background_value,
                    depth_sub_dir, gray_sub_dir, min_ratio_inside):#,
#                    do_use_cache, cache_path):
    """
    """
    # Load from original file
    filepath = copy.deepcopy(sample_data.filename)
    if not os.path.isfile(filepath):
        raise UserWarning("Image file from ICG dataset does not exist \
            (Filename: {})".format(filepath))
    img = load_and_preproc_depth_sample(filepath, fx, fy, cx, cy, 
               min_depth=min_depth_cam, max_depth=max_depth_cam, 
               min_ir=min_ir_value, background_value=background_value,
               depth_sub_dir=depth_sub_dir, 
               gray_sub_dir=gray_sub_dir)
    
    # Add white noise
    if do_add_white_noise:
        img = img + sigma_noise * rng.randn(img.shape[0], img.shape[1])
        
    # Detect hand
    dtor = HandDetectorICG()

    com_uvd = []
    if is_labeled:
        com_uvd = copy.deepcopy(np.asarray(sample_data.gt_uvd[joint_id_com], dtype=np.float32))
    else:
        com_uvd = copy.deepcopy(point3DToImg(sample_data.com_3D, fx, fy, cx, cy))
        
    # Add detection noise
    com_offset = np.zeros((3), np.float32)
    if do_jitter_com:
        com_offset = sigma_com * rng.randn(3)
        # Transform x/y to pixel coords (since com is in uvd coords)
        com_offset[0] = (fx * com_offset[0]) / (com_uvd[2] + com_offset[2])
        com_offset[1] = (fy * com_offset[1]) / (com_uvd[2] + com_offset[2])
    com_uvd += com_offset

    # Crop sample
    try:
        img, M, com = dtor.cropArea3D(imgDepth=img, com=com_uvd, fx=fx, fy=fy, 
                                      minRatioInside=min_ratio_inside, 
                                      size=crop_size_mm, dsize=crop_size_px)
    except:
        lab_str = "unlabeled"
        if is_labeled:
            lab_str = "labeled"
        print("Issue with {} file: {}".format(lab_str, filepath))
                                  
    # Update label info according to crop
    com_3D = pointImgTo3D(com, fx, fy, cx, cy)
    if is_labeled:
        gt_3D = copy.deepcopy(sample_data.gt_3D)
        gt_3D_crop = gt_3D - com_3D
        gt_uvd = copy.deepcopy(sample_data.gt_uvd)
        gt_uvd_crop = np.zeros((gt_uvd.shape[0], 3), np.float32)
        for j in range(gt_uvd.shape[0]):
            t = transformPoint2D(gt_uvd[j], M)
            gt_uvd_crop[j,0] = t[0]
            gt_uvd_crop[j,1] = t[1]
            gt_uvd_crop[j,2] = gt_uvd[j,2]
    else:
        gt_3D       = np.zeros((num_joints,3), np.float32)
        gt_3D_crop  = np.zeros((num_joints,3), np.float32)
        gt_uvd      = np.zeros((num_joints,3), np.float32)
        gt_uvd_crop = np.zeros((num_joints,3), np.float32)
        
    config = {'cube': crop_size_mm}
    handtype = copy.deepcopy(sample_data.handtype)
    id_camparams = copy.deepcopy(sample_data.id_camparams)
    return ICGFrame(img, gt_uvd, gt_uvd_crop, M, 
                    gt_3D, gt_3D_crop, com_3D, filepath,
                    config, handtype, id_camparams)
    
    
def normalize_zero_one(sample):
    img = np.asarray(sample.img_depth.copy(), 'float32')
    img[img == 0] = sample.com_3D[2] + (sample.config['cube'][2] / 2.)
    img -= (sample.com_3D[2] - (sample.config['cube'][2] / 2.))
    img /= sample.config['cube'][2]
    
    target = np.clip(
                np.asarray(sample.gt_3D_crop, dtype='float32') 
                / sample.config['cube'][2], -0.5, 0.5) + 0.5
                
    return img, target
    
    
def normalize_minus_one_one(sample):
    img = np.asarray(sample.img_depth.copy(), 'float32')
    img[img == 0] = sample.com_3D[2] + (sample.config['cube'][2] / 2.)
    img -= sample.com_3D[2]
    img /= (sample.config['cube'][2] / 2.)
    
    target = np.clip(
                np.asarray(sample.gt_3D_crop, dtype='float32') 
                / (sample.config['cube'][2] / 2.), -1, 1)
                
    return img, target
       
        
def denormalize_joint_pos(joint_pos, de_norm_zero_one, crop_size_tuple_3d_mm):
    """
    Re-scale the given joint positions to metric distances (in mm)
    
    Arguments:
        joint_pos (numpy.ndarray): normalized joint positions, 
            as provided by the dataset
        de_norm_zero_one (boolean): whether jointPos are [0,1] normalized or [-1,1]
        crop_size_tuple_3d_mm: 3D 'cube' size in mm (i.e., hand cube size)
    """
    offset = 0
    scale_factor = crop_size_tuple_3d_mm[2] / 2.0
    if de_norm_zero_one:
        offset = -0.5
        scale_factor = crop_size_tuple_3d_mm[2]
        
    return ((joint_pos + offset) * scale_factor)
    