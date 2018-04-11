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

import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from enum import Enum
import time
import math
import scipy.misc


fingercolors = ["r", "g", "b", "c", "m"]

class AnnoStyle(Enum):
    ICVL24 = 0
    NYU_ALL36 = 1
    NYU_EVAL14 = 2
        
        
# Shows an image
def imshow(img):
    img = img / 2 + 0.5     # de-normalize
    npimg = img.numpy()
    
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    
# Saves an image
def imwrite(img, filepath, do_normalize=False):
    if do_normalize:
        img = img / 2 + 0.5
    npimg = img.numpy()

    scipy.misc.imsave(filepath, np.transpose(npimg, (1, 2, 0)))


def show_kernels(kernels, min_weight=-1.0, max_weight=1.0, title=""):
    """
    Plot the kernel weights
    
    Arguments:
        kernels (numpy.ndarray): array of size KxCxHxW, where 
            K is number of output channels (i.e., number of kernels)
            C is number of input channels
            H is kernel height
            W is kernel width
        min_weight (float, optional): clipping boundary; default = -1
        max_weight (float, optional): clipping boundary; default = 1
        title (string, optional): plot title
    """
    
    if len(kernels.shape) != 4:
        raise UserWarning("show_kernels(): kernel dimension must be 4")
    
    if kernels.shape[1] == 1:
        show_kernels_1inchannel(kernels, min_weight, max_weight, title)
    else:
        raise UserWarning("show_kernels(): Not implemented")
    
    
def show_kernels_1inchannel(kernels, min_weight=-1.0, max_weight=1.0, title=""):
    print(title)
    # Plot layerweights in single figure
    numFilters = kernels.shape[0]
    numPlotrows = np.int32(np.floor( np.sqrt(numFilters) ))
    numPlotcols = np.int32(np.ceil( numFilters / numPlotrows ))
    fig, axes = plt.subplots(numPlotrows, numPlotcols)
    fig.suptitle(title)
    for w in range(len(axes.flat)):
        ax = axes.flat[w]
        im = ax.imshow(kernels[w,0], vmin=min_weight, vmax=max_weight, 
                       cmap="jet", interpolation="nearest")
        
    # Add colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    # Stats
    print("min: {}".format(np.min(kernels)))
    print("max: {}".format(np.max(kernels)))
    
    
def plot_pose_and_image(pose, image, anno_type=AnnoStyle.ICVL24):
    """
    Plots the given pose in 3D and the image
    
    Arguments:
        pose (numpy.ndarray): Nx3 array of joint positions
        image ():
        anno_type (AnnoStyle, optional): annotation type
    """
    indices_list = [[]]
    if anno_type == AnnoStyle.ICVL24:
        indices_list = [range(0,4),             # thumb
                        range(4,9),             # index
                        range(9,14),            # middle
                        range(14,19),           # ring
                        range(19,24)]           # pinky
    elif anno_type == AnnoStyle.NYU_ALL36:
        indices_list = [range(24,30) + [34],    # thumb
                        range(18,24) + [32],    # index 
                        range(12,18) + [32],    # middle
                        range(6,12) + [32],     # ring
                        range(0,6) + [33]]      # pinky
    else:
        raise UserWarning("Not implemented.")
            
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(121, projection='3d')
    indices = indices_list[0]
    ax2.plot(pose[indices,0], pose[indices,1], pose[indices,2], color=fingercolors[0])
    indices = indices_list[1]
    ax2.plot(pose[indices,0], pose[indices,1], pose[indices,2], color=fingercolors[1])
    indices = indices_list[2]
    ax2.plot(pose[indices,0], pose[indices,1], pose[indices,2], color=fingercolors[2])
    indices = indices_list[3]
    ax2.plot(pose[indices,0], pose[indices,1], pose[indices,2], color=fingercolors[3])
    indices = indices_list[4]
    ax2.plot(pose[indices,0], pose[indices,1], pose[indices,2], color=fingercolors[4])
    
    # Make axis equal by setting the limits accordingly
    max_range = np.array([pose[:,0].max()-pose[:,0].min(), pose[:,1].max()-pose[:,1].min(), pose[:,2].max()-pose[:,2].min()]).max()
    min_x = 0.5 * (pose[:,0].max()+pose[:,0].min()) - 0.5 * max_range
    max_x = 0.5 * (pose[:,0].max()+pose[:,0].min()) + 0.5 * max_range
    min_y = 0.5 * (pose[:,1].max()+pose[:,1].min()) - 0.5 * max_range
    max_y = 0.5 * (pose[:,1].max()+pose[:,1].min()) + 0.5 * max_range
    min_z = 0.5 * (pose[:,2].max()+pose[:,2].min()) - 0.5 * max_range
    max_z = 0.5 * (pose[:,2].max()+pose[:,2].min()) + 0.5 * max_range
    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(min_y, max_y)
    ax2.set_zlim(min_z, max_z)
    
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
   
    
    ax2 = fig2.add_subplot(122)
    ax2.imshow(image)
    plt.show()
    
   
    fig2.show()
            
            
def log_gradient_distributions(logger, model, step=-1):
    """
    Logs the gradient distributions for each model layer using the given 
    crayon logger
    
    Arguments:
        logger (CrayonExperiment): crayon logger, 
            i.e., created by CrayonClient.create_experiment()
        model (torch.nn.Module)
        step (integer, optional): if not specified previous step is 
            incremented by one, or step is set to zero for the first log
    """
    for i, param in enumerate(model.parameters()):
        layer_idx = (i / 2) + 1     # bias should get same index
        name = "gradient-distribution layer {} (shape: {})".format( \
            layer_idx, param.grad.data.cpu().numpy().shape)
        logger.add_histogram_value(name, \
                 param.grad.data.cpu().numpy().flatten().tolist(), \
                 tobuild=True, wall_time=time.clock(), step=step)
        
        
def log_weight_distributions(logger, model, step=-1):
    """
    Logs the weight/parameter distributions for each model layer 
    using the given crayon logger
    
    Arguments:
        logger (CrayonExperiment): crayon logger, 
            i.e., created by CrayonClient.create_experiment()
        model (torch.nn.Module)
        step (integer, optional): if not specified previous step is 
            incremented by one, or step is set to zero for the first log
    """
    for i, param in enumerate(model.parameters()):
        layer_idx = (i / 2) + 1     # bias should get same index
        name = "weight-distribution layer {} (shape: {})".format( \
            layer_idx, param.grad.data.cpu().numpy().shape)
        logger.add_histogram_value(name, \
                 param.data.cpu().numpy().flatten().tolist(), \
                 tobuild=True, wall_time=time.clock(), step=step)
                 
                 
def scale_batch_avg(data, out_size):
    """
    Scales the images in a batch using average pooling
    
    Note, this is only used for ease of implementation and should be used 
    with  care. E.g., it's especially unsuitable for small scale changes.
    
    Arguments:
        data (torch.Tensor): mini batch (NxCxHxW); (on CPU)
        out_size (2-tuple): (width, height)
    """
    if (not data.size()[2] == data.size()[3]) or \
        (not out_size[0] == out_size[1]):
        raise UserWarning("Scaling with average pooling only implemented \
            for squarred patch sizes")
                
    in_size_h = data.size()[2]
    out_size_h = out_size[1]
    
    if in_size_h == out_size_h:
        return data
    else:
        k_size = int(math.ceil(float(in_size_h) / float(out_size_h)))
        stride = k_size
        padding = max(0, int(math.ceil( \
            0.5 * (stride * (out_size_h - 1) + k_size - in_size_h))))
        scale = nn.AvgPool2d(k_size, stride=stride, padding=padding, \
            count_include_pad=False)
        return scale(data)
        