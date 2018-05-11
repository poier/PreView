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
from util.transformations import transformPoint2D
from data.basetypes import NamedImgSequence, ICVLFrame

# PyTorch
import torch
from torch.utils.data.dataset import Dataset

# General
from PIL import Image
import numpy as np
import os.path
import cPickle
import gzip
import scipy.io
from enum import IntEnum
import copy


#%% General definitions for NYU dataset
class NyuAnnoType(IntEnum):
    ALL_JOINTS = 36             # all 36 joints
    EVAL_JOINTS_ORIGINAL = 14   # original 14 evaluation joints
    
nyuRestrictedJointsEval = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]


#%%
class NyuHandPoseMultiViewDataset(Dataset):
    """
    """
    
    def __init__(self, basepath,
                 train=True, 
                 cropSize=(128, 128),
                 doJitterCom=[False, False, False],
                 sigmaCom=[20., 20., 20.],
                 doAddWhiteNoise=[False, False, False],
                 sigmaNoise=[10., 10., 10.],
                 doLoadRealSamples=True,
                 unlabeledSampleIds=None,
                 transform=None, targetTransform=None, 
                 useCache=True, cacheDir='./cache/single_samples',
                 annoType=0,
                 neededCamIds=[1, 2, 3],
                 randomSeed = 123456789,
                 cropSize3D=(250,250,250),
                 args_data=None):
        """
        Initialize the dataset
        
        Arguments:
            basepath (string): base path, containing sub-folders "train" and "test"
            camId (int, optional): camera ID, as used in filename; \in {1,2,3}; 
                default: 1
            train (boolean, optional): True (default): use train set; 
                False: use test set
            cropSize (2-tuple, optional): size of cropped patches in pixels;
                default: 128x128
            doJitterCom (boolean, optional): 3-element list; one for each cam; 
                default = [False, False, False]
            sigmaCom (float, optional): sigma for center of mass samples 
                (in millimeters); 3-element list; one for each cam; 
                only relevant if doJitterCom is True
            doAddWhiteNoise (boolean, optional):  3-element list; one for 
                each cam; add normal distributed noise to the depth image?; 
                default: False
            sigmaNoise (float, optional): sigma for additive noise; 
                3-element list; one for each cam; 
                only relevant if doAddWhiteNoise is True
            doLoadRealSamples (boolean, optional): whether to load the real 
                sample, i.e., captured by the camera (True; default) or 
                the synthetic sample, rendered using a hand model (False)
            unlabeledSampleIds (list, optional): list of sample-IDs for 
                which the label should NOT be used;
                default: None means for all samples the labels are used
            useCache (boolean, optional): True (default): store in/load from pickle file
            cacheDir (string, optional): path to store/load pickle
            annoType (NyuAnnoType, optional): Type of annotation, i.e., 
                which joints are used.
            neededCamIds (list, optional): list of camera IDs (\in {1,2,3}),
                for which samples should be loaded (can save loading time 
                if not all cameras are needed)
            randomSeed (int, optional): seed for random number generator, 
                e.g., used for jittering, default: 123456789
            cropSize3D (tuple, optional): metric crop size in mm, 
                default: (250,250,250)
        """
        print("Init NYU Dataset...")
        
        # Sanity checks
        if (not type(doJitterCom) == list) or (not type(sigmaCom) == list):
            raise UserWarning("Parameter 'doJitterCom'/'sigmaCom' \
                must be given in a list (for each camera).")
        if (not len(doJitterCom) == 3) or (not len(sigmaCom) == 3):
            raise UserWarning("Parameters 'doJitterComnumSamples'/'sigmaCom' \
                must be 3-element lists.")
        if (not type(doAddWhiteNoise) == list) or (not type(sigmaNoise) == list):
            raise UserWarning("Parameter 'doAddWhiteNoise'/'sigmaNoise' \
                must be given in a list (for each camera).")
        if (not len(doAddWhiteNoise) == 3) or (not len(sigmaNoise) == 3):
            raise UserWarning("Parameters 'doAddWhiteNoise'/'sigmaNoise' \
                must be 3-element lists.")
                
        self.min_depth_cam = 50.
        self.max_depth_cam = 1500.

        self.args_data = args_data
        
        self.rng = np.random.RandomState(randomSeed)
        
        self.basepath = basepath
        # Same parameters for all three cameras (seem to be used by Tompson)
        self.cam1 = Camera(camid=1, fx=588.03, fy=587.07, ux=320., uy=240.)
        self.cam2 = Camera(camid=2, fx=588.03, fy=587.07, ux=320., uy=240.)
        self.cam3 = Camera(camid=3, fx=588.03, fy=587.07, ux=320., uy=240.)
        self.doLoadCam1 = 1 in neededCamIds
        self.doLoadCam2 = 2 in neededCamIds
        self.doLoadCam3 = 3 in neededCamIds
        self.useCache = useCache
        self.cacheBaseDir = cacheDir
        self.restrictedJointsEval = nyuRestrictedJointsEval
        self.config = {'cube':cropSize3D}
        self.cropSize = cropSize
        self.doJitterCom = doJitterCom
        self.doAddWhiteNoise = doAddWhiteNoise
        self.sigmaNoise = sigmaNoise
        self.sigmaCom = sigmaCom
        self.doLoadRealSamples = doLoadRealSamples
        
        self.doNormZeroOne = self.args_data.do_norm_zero_one  # [-1,1] or [0,1]
                
        self.transform = transform
        self.targetTransform = targetTransform
        
        self.seqName = ""
        if train:
            self.seqName = "train"
        else:
            self.seqName = "test"
            
        self.annoType = annoType
        self.doUseAllJoints = True
        if self.annoType == NyuAnnoType.EVAL_JOINTS_ORIGINAL:
            self.doUseAllJoints = False
            
        self.numJoints = annoType
        
        # Load labels
        trainlabels = '{}/{}/joint_data.mat'.format(basepath, self.seqName)
        self.labelMat = scipy.io.loadmat(trainlabels)
        
        # Get number of samples from annotations (test: 8252; train: 72757)
        numAllSamples = self.labelMat['joint_xyz'][self.cam1.camid-1].shape[0]
        self.numSamples = numAllSamples

        self.isSampleLabeled = np.ones((self.numSamples), dtype=bool)
        if not unlabeledSampleIds is None:
            self.isSampleLabeled[unlabeledSampleIds] = False
        
        if self.useCache:
            # Assemble and create cache dir if necessary
            synthString = "real" if self.doLoadRealSamples else "synthetic"
            # Cam1
            camString = "cam{}".format(self.cam1.camid)
            self.cacheDirCam1 = os.path.join(
                self.cacheBaseDir, self.seqName, synthString, camString)
            if not os.path.exists(self.cacheDirCam1):
                os.makedirs(self.cacheDirCam1)
            # Cam2
            camString = "cam{}".format(self.cam2.camid)
            self.cacheDirCam2 = os.path.join(
                self.cacheBaseDir, self.seqName, synthString, camString)
            if not os.path.exists(self.cacheDirCam2):
                os.makedirs(self.cacheDirCam2)
            # Cam3numSamples
            camString = "cam{}".format(self.cam3.camid)
            self.cacheDirCam3 = os.path.join(
                self.cacheBaseDir, self.seqName, synthString, camString)
            if not os.path.exists(self.cacheDirCam3):
                os.makedirs(self.cacheDirCam3)
                
        # Precomputations for normalization of 3D point
        self.precompute_normalization_factors()
        
        print("NYU Dataset init done.")


    def __getitem__(self, index):
        config = self.config
        # Cam1
        if self.doLoadCam1:
            dataCam1 = loadSingleSampleNyu(self.basepath, self.seqName, index, 
                                           self.rng,
                                           doLoadRealSample=self.doLoadRealSamples,
                                            camId=self.cam1.camid, 
                                            fx=self.cam1.fx, fy=self.cam1.fy, 
                                            ux=self.cam1.ux, uy=self.cam1.uy,
                                            allJoints=self.doUseAllJoints, 
                                            config=config,
                                            cropSize=self.cropSize,
                                            doJitterCom=self.doJitterCom[0],
                                            sigmaCom=self.sigmaCom[0],
                                            doAddWhiteNoise=self.doAddWhiteNoise[0],
                                            sigmaNoise=self.sigmaNoise[0],
                                            useCache=self.useCache,
                                            cacheDir=self.cacheDirCam1,
                                            labelMat=self.labelMat,
                                            doUseLabel=self.isSampleLabeled[index],
                                            minRatioInside=self.args_data.min_ratio_inside).data[0]
                                            
            if self.doNormZeroOne:
                imgCam1, targetCam1 = normalizeZeroOne(dataCam1)
            else:
                imgCam1, targetCam1 = normalizeMinusOneOne(dataCam1)
                
            # Image need to be HxWxC and \in [0,255] for transform
            imgCam1 = np.expand_dims(imgCam1, axis=2) * 255.0
            
            if self.transform is not None:
                imgCam1 = self.transform(imgCam1)
                
            if self.targetTransform is not None:
                targetCam1 = self.targetTransform(targetCam1)
                
            targetCam1 = torch.from_numpy(targetCam1.astype('float32'))
            transformCropCam1 = torch.from_numpy(dataCam1.T)
            comCam1 = torch.from_numpy(dataCam1.com)
        else:
            imgCam1 = []
            targetCam1 = []
            transformCropCam1 = []
            comCam1 = []
        # Cam2
        if self.doLoadCam2:
            dataCam2 = loadSingleSampleNyu(self.basepath, self.seqName, index, 
                                           self.rng,
                                           doLoadRealSample=self.doLoadRealSamples,
                                            camId=self.cam2.camid, 
                                            fx=self.cam2.fx, fy=self.cam2.fy, 
                                            ux=self.cam2.ux, uy=self.cam2.uy,
                                            allJoints=self.doUseAllJoints, 
                                            config=config,
                                            cropSize=self.cropSize,
                                            doJitterCom=self.doJitterCom[1],
                                            sigmaCom=self.sigmaCom[1],
                                            doAddWhiteNoise=self.doAddWhiteNoise[1],
                                            sigmaNoise=self.sigmaNoise[1],
                                            useCache=self.useCache,
                                            cacheDir=self.cacheDirCam2,
                                            labelMat=self.labelMat,
                                            doUseLabel=self.isSampleLabeled[index],
                                            minRatioInside=self.args_data.min_ratio_inside).data[0]
            if self.doNormZeroOne:
                imgCam2, targetCam2 = normalizeZeroOne(dataCam2)
            else:
                imgCam2, targetCam2 = normalizeMinusOneOne(dataCam2)
                
            # Image need to be HxWxC and \in [0,255] for transform
            imgCam2 = np.expand_dims(imgCam2, axis=2) * 255.0
            
            if self.transform is not None:
                imgCam2 = self.transform(imgCam2)
    
            if self.targetTransform is not None:
                targetCam2 = self.targetTransform(targetCam2)
                
            targetCam2 = torch.from_numpy(targetCam2.astype('float32'))
            transformCropCam2 = torch.from_numpy(dataCam2.T)
            comCam2 = torch.from_numpy(dataCam2.com)
        else:
            imgCam2 = []
            targetCam2 = []
            transformCropCam2 = []
            comCam2 = []
        # Cam3                
        if self.doLoadCam3:                            
            dataCam3 = loadSingleSampleNyu(self.basepath, self.seqName, index, 
                                           self.rng,
                                           doLoadRealSample=self.doLoadRealSamples,
                                            camId=self.cam3.camid, 
                                            fx=self.cam3.fx, fy=self.cam3.fy, 
                                            ux=self.cam3.ux, uy=self.cam3.uy,
                                            allJoints=self.doUseAllJoints, 
                                            config=config,
                                            cropSize=self.cropSize,
                                            doJitterCom=self.doJitterCom[2],
                                            sigmaCom=self.sigmaCom[2],
                                            doAddWhiteNoise=self.doAddWhiteNoise[1],
                                            sigmaNoise=self.sigmaNoise[1],
                                            useCache=self.useCache,
                                            cacheDir=self.cacheDirCam3,
                                            labelMat=self.labelMat,
                                            doUseLabel=self.isSampleLabeled[index],
                                            minRatioInside=self.args_data.min_ratio_inside).data[0]
            if self.doNormZeroOne:
                imgCam3, targetCam3 = normalizeZeroOne(dataCam3)
            else:
                imgCam3, targetCam3 = normalizeMinusOneOne(dataCam3)
                
            # Image need to be HxWxC and \in [0,255] for transform
            imgCam3 = np.expand_dims(imgCam3, axis=2) * 255.0
               
            if self.transform is not None:
                imgCam3 = self.transform(imgCam3)
    
            if self.targetTransform is not None:
                targetCam3 = self.targetTransform(targetCam3)
                    
            targetCam3 = torch.from_numpy(targetCam3.astype('float32'))
            transformCropCam3 = torch.from_numpy(dataCam3.T)
            comCam3 = torch.from_numpy(dataCam3.com)
        else:
            imgCam3 = []
            targetCam3 = []
            transformCropCam3 = []
            comCam3 = []
            
        return imgCam1, imgCam2, imgCam3, targetCam1, targetCam2, targetCam3, \
            transformCropCam1, transformCropCam2, transformCropCam3, \
            comCam1, comCam2, comCam3


    def __len__(self):
        return self.numSamples
        

    def get_type_string(self):
        return "NYU"
        
        
    def denormalize_joint_pos(self, jointPos, config=None):
        """
        Re-scale the given joint positions to metric distances (in mm)
        
        Arguments:
            jointPos (numpy.ndarray): normalized joint positions, 
                as provided by the dataset
            config (optional): config with 'cube' setting (i.e., hand cube size)
        """
        if config is None:
            config = self.config
        return denormalizeJointPositions(jointPos, self.doNormZeroOne, config)
        
        
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


#%% Helpers
class Camera(object):
    """
    Just encapsulating some camera information/parameters
    """
    
    def __init__(self, camid=0, fx=None, fy=None, ux=None, uy=None):
        self.camid = camid
        self.fx = fx
        self.fy = fy
        self.ux = ux
        self.uy = uy
    
        
def loadDepthMap(filename):
    """
    Read a depth-map
    :param filename: file name to load
    :return: image data of depth image
    """

    with open(filename) as f:
        img = Image.open(filename)
        # top 8 bits of depth are packed into green channel and lower 8 bits into blue
        assert len(img.getbands()) == 3
        r, g, b = img.split()
        r = np.asarray(r,np.int32)
        g = np.asarray(g,np.int32)
        b = np.asarray(b,np.int32)
        dpt = np.bitwise_or(np.left_shift(g,8),b)
        imgdata = np.asarray(dpt,np.float32)

    return imgdata


def loadSingleSampleNyu(basepath, seqName, index, rng,
                        doLoadRealSample=True,
                        camId=1, fx=588.03, fy=587.07, ux=320., uy=240.,
                        allJoints=False, 
                        config={'cube': (250,250,250)},
                        cropSize=(128,128),
                        doJitterCom=False,
                        sigmaCom=20.,
                        doAddWhiteNoise=False,
                        sigmaNoise=10.,
                        useCache=True,
                        cacheDir="./cache/single_samples",
                        labelMat=None,
                        doUseLabel=True,
                        minRatioInside=0.3):
    """
    Load an image sequence from the NYU hand pose dataset
    
    Arguments:
        basepath (string): base path, containing sub-folders "train" and "test"
        seqName: sequence name, e.g. train
        index (int): index of image to be loaded
        rng (random number generator): as returned by numpy.random.RandomState
        doLoadRealSample (boolean, optional): whether to load the real 
            sample, i.e., captured by the camera (True; default) or 
            the synthetic sample, rendered using a hand model (False)
        camId (int, optional): camera ID, as used in filename; \in {1,2,3}; 
            default: 1
        fx, fy (float, optional): camera focal length; default for cam 1
        ux, uy (float, optional): camera principal point; default for cam 1
        allJoints (boolean): use all 36 joints or just the eval.-subset
        config (dictionary, optional): need to have a key 'cube' whose value 
            is a 3-tuple specifying the cube-size (x,y,z in mm) 
            which is extracted around the found hand center location
        cropSize (2-tuple, optional): size of cropped patches in pixels;
            default: 128x128
        doJitterCom (boolean, optional): default: False
        sigmaCom (float, optional): sigma for center of mass samples 
            (in millimeters); only relevant if doJitterCom is True
        sigmaNoise (float, optional): sigma for additive noise; 
            only relevant if doAddWhiteNoise is True
        doAddWhiteNoise (boolean, optional): add normal distributed noise 
            to the depth image; default: False
        useCache (boolean, optional): True (default): store in/load from 
            pickle file
        cacheDir (string, optional): path to store/load pickle
        labelMat (optional): loaded mat file; (full file need to be loaded for 
            each sample if not given)
        doUseLabel (bool, optional): specify if the label should be used;
            default: True
        
    Returns:
        named image sequence
    """
        
    # Load the dataset
    objdir = '{}/{}/'.format(basepath,seqName)

    if labelMat == None:
        trainlabels = '{}/{}/joint_data.mat'.format(basepath, seqName)
        labelMat = scipy.io.loadmat(trainlabels)
        
    joints3D = labelMat['joint_xyz'][camId-1]
    joints2D = labelMat['joint_uvd'][camId-1]
    if allJoints:
        eval_idxs = np.arange(36)
    else:
        eval_idxs = nyuRestrictedJointsEval

    numJoints = len(eval_idxs)
    
    data = []
    line = index
    
    # Assemble original filename
    prefix = "depth" if doLoadRealSample else "synthdepth"
    dptFileName = '{0:s}/{1:s}_{2:1d}_{3:07d}.png'.format(objdir, prefix, camId, line+1)
    # Assemble pickled filename
    cacheFilename = "frame_{}_all{}.pgz".format(index, allJoints)
    pickleCacheFile = os.path.join(cacheDir, cacheFilename)
        
    # Load image
    if useCache and os.path.isfile(pickleCacheFile):
        # Load from pickle file
        with gzip.open(pickleCacheFile, 'rb') as f:
            try:
                dpt = cPickle.load(f)
            except:
                print("Data file exists but failed to laod. File: {}".format(pickleCacheFile))
                raise
        
    else:
        # Load from original file
        if not os.path.isfile(dptFileName):
            raise UserWarning("Desired image file from NYU dataset does not exist \
                (Filename: {})".format(dptFileName))
        dpt = loadDepthMap(dptFileName)
    
        # Write to pickle file
        if useCache:
            with gzip.GzipFile(pickleCacheFile, 'wb') as f:
                cPickle.dump(dpt, f, protocol=cPickle.HIGHEST_PROTOCOL)
    
    # Add noise?
    if doAddWhiteNoise:
        dpt = dpt + sigmaNoise * rng.randn(dpt.shape[0], dpt.shape[1])
    
    # joints in image coordinates
    gtorig = np.zeros((numJoints, 3), np.float32)
    jt = 0
    for ii in range(joints2D.shape[1]):
        if ii not in eval_idxs:
            continue
        gtorig[jt,0] = joints2D[line,ii,0]
        gtorig[jt,1] = joints2D[line,ii,1]
        gtorig[jt,2] = joints2D[line,ii,2]
        jt += 1

    # normalized joints in 3D coordinates
    gt3Dorig = np.zeros((numJoints,3),np.float32)
    jt = 0
    for jj in range(joints3D.shape[1]):
        if jj not in eval_idxs:
            continue
        gt3Dorig[jt,0] = joints3D[line,jj,0]
        gt3Dorig[jt,1] = joints3D[line,jj,1]
        gt3Dorig[jt,2] = joints3D[line,jj,2]
        jt += 1
        
    # Detect hand
    hdOwn = HandDetectorICG()
            
    comGT = copy.deepcopy(gtorig[13])
    if allJoints:
        comGT = copy.deepcopy(gtorig[34])
        
    # Jitter com?
    comOffset = np.zeros((3), np.float32)
    if doJitterCom:
        comOffset = sigmaCom * rng.randn(3)
        # Transform x/y to pixel coords (since com is in uvd coords)
        comOffset[0] = (fx * comOffset[0]) / (comGT[2] + comOffset[2])
        comOffset[1] = (fy * comOffset[1]) / (comGT[2] + comOffset[2])
    comGT = comGT + comOffset
    
    dpt, M, com = hdOwn.cropArea3D(imgDepth=dpt, com=comGT, fx=fx, fy=fy, 
                                   minRatioInside=minRatioInside, \
                                   size=config['cube'], dsize=cropSize)
                                    
    com3D = jointImgTo3D(com, fx, fy, ux, uy)
    gt3Dcrop = gt3Dorig - com3D     # normalize to com
    gtcrop = np.zeros((gtorig.shape[0], 3), np.float32)
    for joint in range(gtorig.shape[0]):
        t=transformPoint2D(gtorig[joint], M)
        gtcrop[joint, 0] = t[0]
        gtcrop[joint, 1] = t[1]
        gtcrop[joint, 2] = gtorig[joint, 2]
                
    if not doUseLabel:
        gtorig = np.zeros(gtorig.shape, gtorig.dtype)
        gtcrop = np.zeros(gtcrop.shape, gtcrop.dtype)
        gt3Dorig = np.zeros(gt3Dorig.shape, gt3Dorig.dtype)
        gt3Dcrop = np.zeros(gt3Dcrop.shape, gt3Dcrop.dtype)
        
    data.append(ICVLFrame(
        dpt.astype(np.float32),gtorig,gtcrop,M,gt3Dorig,
        gt3Dcrop,com3D,dptFileName,'',config) )
        
    # To reuse the code from DeepPrior just use NamedImgSequence as container for the sample
    return NamedImgSequence(seqName,data,config)


def jointImgTo3D(sample, fx, fy, ux, uy):
    """
    Normalize sample to metric 3D
    :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
    :return: normalized joints in mm
    """
    ret = np.zeros((3,), np.float32)
    # convert to metric using f, see Thomson et al.
    ret[0] = (sample[0] - ux) * sample[2] / fx
    ret[1] = (uy - sample[1]) * sample[2] / fy
    ret[2] = sample[2]
    return ret
    
    
def normalizeZeroOne(sample):
    imgD = np.asarray(sample.dpt.copy(), 'float32')
    imgD[imgD == 0] = sample.com[2] + (sample.config['cube'][2] / 2.)
    imgD -= (sample.com[2] - (sample.config['cube'][2] / 2.))
    imgD /= sample.config['cube'][2]
    
    target = np.clip(
                np.asarray(sample.gt3Dcrop, dtype='float32') 
                / sample.config['cube'][2], -0.5, 0.5) + 0.5
                
    return imgD, target
    
    
def normalizeMinusOneOne(sample):
    imgD = np.asarray(sample.dpt.copy(), 'float32')
    imgD[imgD == 0] = sample.com[2] + (sample.config['cube'][2] / 2.)
    imgD -= sample.com[2]
    imgD /= (sample.config['cube'][2] / 2.)
    
    target = np.clip(
                np.asarray(sample.gt3Dcrop, dtype='float32') 
                / (sample.config['cube'][2] / 2.), -1, 1)
                
    return imgD, target
        
        
def denormalizeJointPositions(jointPos, deNormZeroOne, config):
    """
    Re-scale the given joint positions to metric distances (in mm)
    
    Arguments:
        jointPos (numpy.ndarray): normalized joint positions, 
            as provided by the dataset
        deNormZeroOne (boolean): whether jointPos are [0,1] normalized or [-1,1]
        config: config with 'cube' setting (i.e., hand cube size)
    """
    offset = 0
    scaleFactor = config['cube'][2] / 2.0
    if deNormZeroOne:
        offset = -0.5
        scaleFactor = config['cube'][2]
        
    return ((jointPos + offset) * scaleFactor)
    