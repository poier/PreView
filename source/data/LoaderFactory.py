# -*- coding: utf-8 -*-
"""
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
from data.NyuHandPoseDataset import NyuHandPoseMultiViewDataset
from data.IcgHandPoseDataset import IcgHandPoseMultiViewDataset

# PyTorch
import torch
from torchvision import transforms
import torch.utils.data.sampler as smpl

# Libs
import numpy as np

# builtin
from enum import IntEnum


#%% General definitions
class LoaderMode(IntEnum):
    TRAIN = 0
    VAL = 1
    TEST = 2
    
    
class DatasetType(IntEnum):
    NYU = 0
    ICG = 1


#%% Functions
def create_dataloader(loader_type, args_data, do_use_gpu):
    """
    Create a data loader according to the parameters
    """
    
    kwargs = {'num_workers': args_data.num_loader_workers, 'pin_memory': True} if do_use_gpu else {}
    
    if args_data.dataset_type == DatasetType.NYU:
        if loader_type == LoaderMode.TRAIN:
            # Set up sample IDs to sample from
            ids_train = np.arange(args_data.id_start_train, args_data.id_end_train+1)
            ids_train_permuted = args_data.rng.permutation(ids_train)
            ids_train_labeled = ids_train_permuted[:args_data.num_labeled_samples]
            ids_train_unlabeled = ids_train_permuted[args_data.num_labeled_samples:]
            # Ensure a minimum sampling probability for labeled samples
            ratio_labeled = len(ids_train_labeled) / float(len(ids_train))
            prob_labeled = max(args_data.min_sampling_prob_labeled, ratio_labeled)
            prob_unlabeled = 1.0 - prob_labeled
            # Set up distribution/weights to sample from (considering un-/labeled samples)
            scale_weights = float(len(ids_train))   # value to which weights will sum up
            sample_weight_labeled = prob_labeled * scale_weights / float(len(ids_train_labeled))
            sample_weight_unlabeled = prob_unlabeled * scale_weights \
                                        / float(len(ids_train_unlabeled)) \
                                        if len(ids_train_unlabeled) > 0 else 0.0
            sampling_weights = np.zeros((args_data.num_all_samples_train))
            sampling_weights[ids_train_labeled] = sample_weight_labeled
            sampling_weights[ids_train_unlabeled] = sample_weight_unlabeled
            num_samples_used_for_train = np.count_nonzero(sampling_weights)
            
            loader = torch.utils.data.DataLoader(
                NyuHandPoseMultiViewDataset(args_data.nyu_data_basepath, train=True, 
                                            cropSize=args_data.in_crop_size,
                                            doJitterCom=args_data.do_jitter_com,
                                            sigmaCom=args_data.sigma_com,
                                            doAddWhiteNoise=args_data.do_add_white_noise,
                                            sigmaNoise=args_data.sigma_noise,
                                            doLoadRealSamples=args_data.do_use_real_samples,
                                            unlabeledSampleIds=ids_train_unlabeled,
                                            transform=transforms.ToTensor(),
                                            useCache=args_data.use_pickled_cache,
                                            cacheDir=args_data.nyu_data_basepath_pickled, 
                                            annoType=args_data.anno_type,
                                            neededCamIds=args_data.needed_cam_ids_train,
                                            randomSeed=args_data.seed,
                                            cropSize3D=args_data.crop_size_3d_tuple,
                                            args_data=args_data),
                batch_size=args_data.batch_size, 
                sampler=smpl.WeightedRandomSampler(sampling_weights, 
                                                   num_samples=num_samples_used_for_train, 
                                                   replacement=True),
                **kwargs)
                        
            print("Using {} samples for training".format(num_samples_used_for_train))
            if sample_weight_labeled > 0.:
                print("  {} labeled".format(len(ids_train_labeled)))
            if sample_weight_unlabeled > 0.:
                print("  {} unlabeled".format(len(ids_train_unlabeled)))
                
        elif loader_type == LoaderMode.VAL:
            num_samples_val = min(int(round(args_data.max_val_train_ratio * args_data.num_labeled_samples)), 
                                  args_data.max_num_samples_val)
            ids_val = np.arange(args_data.id_start_val, args_data.id_end_val+1)
            ids_val = args_data.rng.permutation(ids_val)
            ids_val = ids_val[:num_samples_val]
            loader = torch.utils.data.DataLoader(
                NyuHandPoseMultiViewDataset(args_data.nyu_data_basepath, train=False, 
                                            cropSize=args_data.in_crop_size,
                                            doJitterCom=args_data.do_jitter_com_test,
                                            sigmaCom=args_data.sigma_com,
                                            doAddWhiteNoise=args_data.do_add_white_noise_test,
                                            sigmaNoise=args_data.sigma_noise,
                                            doLoadRealSamples=args_data.do_use_real_samples,
                                            transform=transforms.ToTensor(),
                                            useCache=args_data.use_pickled_cache,
                                            cacheDir=args_data.nyu_data_basepath_pickled, 
                                            annoType=args_data.anno_type,
                                            neededCamIds=args_data.needed_cam_ids_test,
                                            randomSeed=args_data.seed,
                                            cropSize3D=args_data.crop_size_3d_tuple,
                                            args_data=args_data),
                batch_size=args_data.batch_size,
                sampler=smpl.SubsetRandomSampler(ids_val), **kwargs)
                        
            print("Using {} samples for validation".format(len(ids_val)))
                
        elif loader_type == LoaderMode.TEST:
            ids_test = np.arange(args_data.id_start_test, args_data.id_end_test+1)
            loader = torch.utils.data.DataLoader(
                NyuHandPoseMultiViewDataset(args_data.nyu_data_basepath, train=False, 
                                            cropSize=args_data.in_crop_size,
                                            doJitterCom=args_data.do_jitter_com_test,
                                            sigmaCom=args_data.sigma_com,
                                            doAddWhiteNoise=args_data.do_add_white_noise_test,
                                            sigmaNoise=args_data.sigma_noise,
                                            doLoadRealSamples=args_data.do_use_real_samples,
                                            transform=transforms.ToTensor(),
                                            useCache=args_data.use_pickled_cache,
                                            cacheDir=args_data.nyu_data_basepath_pickled, 
                                            annoType=args_data.anno_type,
                                            neededCamIds=args_data.needed_cam_ids_test,
                                            randomSeed=args_data.seed,
                                            cropSize3D=args_data.crop_size_3d_tuple,
                                            args_data=args_data),
                batch_size=args_data.batch_size,
                sampler=smpl.SubsetRandomSampler(ids_test), **kwargs)
                        
            print("Using {} samples for test".format(len(ids_test)))
            
        else:
            raise UserWarning("LoaderMode unknown.")
            
    elif args_data.dataset_type == DatasetType.ICG:
        args_data = set_loader_type_specific_settings_icg(args_data, loader_type)
        dataset = IcgHandPoseMultiViewDataset(args_data)
        if loader_type == LoaderMode.TRAIN:
            num_samples = len(dataset)
            num_samples_labeled_all = dataset.get_num_samples_labeled()
            num_samples_unlabeled = dataset.get_num_samples_unlabeled()
            num_samples_labeled_used = min(num_samples_labeled_all, args_data.num_labeled_samples)
            num_samples_used = num_samples_labeled_used + num_samples_unlabeled
            # Set up sample IDs to sample from
            ids_train = np.arange(num_samples)
            ids_train_labeled_all = ids_train[:num_samples_labeled_all]
            ids_train_labeled_perm = args_data.rng.permutation(ids_train_labeled_all)
            ids_train_labeled = ids_train_labeled_perm[:num_samples_labeled_used]
            ids_train_unlabeled = ids_train[num_samples_labeled_all:]
            # Ensure a minimum sampling probability for labeled samples
            ratio_labeled = len(ids_train_labeled) / float(num_samples_used)
            prob_labeled = max(args_data.min_sampling_prob_labeled, ratio_labeled)
            prob_unlabeled = 1.0 - prob_labeled
            # Set up distribution/weights to sample from (considering un-/labeled samples)
            scale_weights = float(num_samples_used)   # value to which weights will sum up
            sample_weight_labeled = prob_labeled * scale_weights / float(len(ids_train_labeled))
            sample_weight_unlabeled = prob_unlabeled * scale_weights \
                                         / float(len(ids_train_unlabeled)) \
                                         if len(ids_train_unlabeled) > 0 else 0.0
            sampling_weights = np.zeros(len(ids_train))
            sampling_weights[ids_train_labeled] = sample_weight_labeled
            sampling_weights[ids_train_unlabeled] = sample_weight_unlabeled
            num_samples_used_for_train = np.count_nonzero(sampling_weights)
                
            loader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=args_data.batch_size,
                        sampler=smpl.WeightedRandomSampler(sampling_weights, 
                                                           num_samples=num_samples_used_for_train,
                                                           replacement=True),
                        **kwargs)
                        
            print("Using {} samples for training".format(num_samples_used_for_train))
            if sample_weight_labeled > 0.:
                print("  {} labeled".format(len(ids_train_labeled)))
            if sample_weight_unlabeled > 0.:
                print("  {} unlabeled".format(len(ids_train_unlabeled)))
                        
        elif loader_type == LoaderMode.VAL:
            # Prepare val. sample IDs
            ids_val = np.arange(len(dataset))
            if args_data.do_val_only_with_labeled:
                num_samples_labeled_all = dataset.get_num_samples_labeled()
                ids_val = np.arange(num_samples_labeled_all)
            # Use subset?                
            max_num_samples_val = int(round(args_data.max_val_train_ratio * args_data.num_labeled_samples))
            num_samples_val = min(max_num_samples_val, len(ids_val))
            ids_val = args_data.rng.permutation(ids_val)
            ids_val = ids_val[:num_samples_val]
                
            loader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=args_data.batch_size, 
                        sampler=smpl.SubsetRandomSampler(ids_val),
                        **kwargs)
                        
            print("Using {} samples for validation".format(len(ids_val)))
            print("  {} labeled (might be incorrect if a subset is used (e.g., wrt. train set size))".format(
                dataset.get_num_samples_labeled()))
            if not args_data.do_val_only_with_labeled:
                print("  {} unlabeled (might be incorrect if a subset is used (e.g., wrt. train set size))".format(
                    dataset.get_num_samples_unlabeled()))
                        
        elif loader_type == LoaderMode.TEST:
            # Prepare test sample IDs
            ids_test = np.arange(len(dataset))
            if args_data.do_test_only_with_labeled:
                num_samples_labeled_all = dataset.get_num_samples_labeled()
                ids_test = np.arange(num_samples_labeled_all)
                
            loader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=args_data.batch_size, 
                        sampler=smpl.SubsetRandomSampler(ids_test),
                        **kwargs)
                        
            print("Using {} samples for test".format(len(ids_test)))
            print("  {} labeled".format(dataset.get_num_samples_labeled()))
            if not args_data.do_test_only_with_labeled:
                print("  {} unlabeled".format(dataset.get_num_samples_unlabeled()))
            
    else:
        raise UserWarning("DatasetType unknown.")
            
    return loader
        
        
def set_loader_type_specific_settings_icg(args_data, loader_type):
    if loader_type == LoaderMode.TRAIN:
        # Set-specific list IDs
        ids_l = args_data.list_indices_labeled_train
        ids_u = args_data.list_indices_unlabeled_train
        # Needed cam IDs
        args_data.do_load_cam1 = 1 in args_data.needed_cam_ids_train
        args_data.do_load_cam2 = 2 in args_data.needed_cam_ids_train
        args_data.do_load_cam3 = 3 in args_data.needed_cam_ids_train
        # Jitter mode
        args_data.do_jitter_com_cur = args_data.do_jitter_com
        args_data.do_white_noise_cur = args_data.do_add_white_noise
        
    elif loader_type == LoaderMode.VAL:
        # Set-specific list IDs
        ids_l = args_data.list_indices_labeled_val
        ids_u = args_data.list_indices_unlabeled_val
        # Needed cam IDs
        args_data.do_load_cam1 = 1 in args_data.needed_cam_ids_train
        args_data.do_load_cam2 = 2 in args_data.needed_cam_ids_train
        args_data.do_load_cam3 = 3 in args_data.needed_cam_ids_train
        # Jitter mode
        args_data.do_jitter_com_cur = args_data.do_jitter_com_test
        args_data.do_white_noise_cur = args_data.do_add_white_noise_test
        # 
        
    elif loader_type == LoaderMode.TEST:
        # Set-specific list IDs
        ids_l = args_data.list_indices_labeled_test
        ids_u = args_data.list_indices_unlabeled_test
        # Needed cam IDs
        args_data.do_load_cam1 = 1 in args_data.needed_cam_ids_test
        args_data.do_load_cam2 = 2 in args_data.needed_cam_ids_test
        args_data.do_load_cam3 = 3 in args_data.needed_cam_ids_test
        # Jitter mode
        args_data.do_jitter_com_cur = args_data.do_jitter_com_test
        args_data.do_white_noise_cur = args_data.do_add_white_noise_test
        
    # Set lists according to IDs
    args_data.labeled_files_list_leftcam_cur    = [args_data.icg_labeled_files_list_leftcam[i] for i in ids_l]
    args_data.labeled_data_camnames_cur         = [args_data.icg_labeled_data_camnames[i] for i in ids_l]
    args_data.labeled_data_intrinsics_list_cur  = [args_data.icg_labeled_data_intrinsics_list[i] for i in ids_l]
    args_data.labeled_data_R_list_cur           = [args_data.icg_labeled_data_R_list[i] for i in ids_l]
    args_data.labeled_data_t_list_cur           = [args_data.icg_labeled_data_t_list[i] for i in ids_l]
    
    # Unlabeled data
    args_data.unlabeled_files_list_cur                  = [args_data.icg_unlabeled_files_list[i] for i in ids_u]
    args_data.unlabeled_data_intrinsics_list_cur    = [args_data.icg_unlabeled_data_intrinsics_list[i] for i in ids_u]
    args_data.unlabeled_data_R_list_cur             = [args_data.icg_unlabeled_data_R_list[i] for i in ids_u]
    args_data.unlabeled_data_t_list_cur             = [args_data.icg_unlabeled_data_t_list[i] for i in ids_u]
    
    return args_data


