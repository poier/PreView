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

from nets.PredViewAndHaPeNetDcGan import PredViewAndHaPeNetDcGan
from nets.DiscriminatorNetDcGan import DiscriminatorNetDcGan, ConditioningType

from enum import IntEnum


#%% General definitions
class NetType(IntEnum):
    PREVIEW = 0
    PREVIEW2 = 1                    # predict two views
    PREVIEW_SEMI_LINEAR = 2         # semi-superv., linear map to target
    PREVIEW_SEMI_FROM_TARGET = 3    # semi-superv., view pred. from target
    POSE_ESTIMATION = 4             # only pose estimation, i.e., only trained from labeled data
    
    
    @classmethod
    def is_output_pose(self, net_type):   # is_training_type_supervised
        return_val = False
        if (net_type == self.PREVIEW_SEMI_LINEAR) \
            or (net_type == self.PREVIEW_SEMI_FROM_TARGET) \
            or (net_type == self.POSE_ESTIMATION):
            return_val = True
        return return_val
    
    
    @classmethod
    def get_num_output_views(self, net_type):
        return_val = 1
        if net_type == self.PREVIEW2:
            return_val = 2
        elif net_type == self.POSE_ESTIMATION:
            return_val = 0
        return return_val
    
    
#%%
def create_net(net_type, 
               params, num_prior_dims, num_cond_dims, num_joints, 
               num_features):
    """
    Create a network model of given type
    """
    
    model = None
    if net_type == NetType.PREVIEW_SEMI_LINEAR:
        model = PredViewAndHaPeNetDcGan(do_use_gpu=params.cuda, 
                                        num_prior_dim=num_prior_dims,
                                        num_cond_dim=num_cond_dims, 
                                        num_output_dim=(num_joints * 3), 
                                        num_features=num_features)
    else:
        print("NetType (={}) not implemented!".format(net_type))
        raise UserWarning("NetType not implemented.")
        
    return model
    
    
def create_net_discriminator(params, num_joints, num_features):
    """
    Create a network model for a discriminator
    """
    num_in_channels = get_num_discriminator_input_channels(
        params.discriminator_condition_type, (num_joints * 3))
    model = DiscriminatorNetDcGan(do_use_gpu=params.cuda, 
                                  num_features=num_features,
                                  num_input_channels=num_in_channels)
    return model
    
    
def get_num_discriminator_input_channels(conditioning_type,  
                                         num_pose_dims=72):
    """
    Get input channels of discriminator according to conditioning
    """
    switcher = {
        ConditioningType.OFF: 1,
        ConditioningType.INPUT: 2,
        ConditioningType.POSE: (1 + num_pose_dims),
        ConditioningType.INPUT_AND_POSE: (2 + num_pose_dims),
    }
    return switcher.get(conditioning_type, 1)
    
    