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

import util.output as output
import util.losses as losses

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torchvision.utils as vutils

import numpy as np

import copy
import time
import os.path



recon_l1 = nn.L1Loss(size_average=True)
#reconstruction_function = nn.SmoothL1Loss(size_average=True)
recon_mse = nn.MSELoss(size_average=True)


class ViewPredAndHaPeTrainer(object):
    """
    """
    
    def __init__(self, train_data_loader, val_data_loader, logger, 
                 trainer_parameters=None):
        """
        Initialize trainer
        
        Arguments:
            train_data_loader (torch.utils.data.DataLoader): data loader 
                for training data
            val_data_loader (torch.utils.data.DataLoader): data loader 
                for validation data
            logger (CrayonExperiment): crayon logger, 
                i.e., created by CrayonClient.create_experiment()
            trainer_parameters (object): object with parameters as attributes
        """
        self.train_loader = train_data_loader
        self.val_loader = val_data_loader
        self.log = logger
        self.train_params = trainer_parameters
        
    
    def loss_function_train(self, joints_pred, joints_target, img_pred, img_target, step_id):
        # Reconstruction loss
        loss_preview = 0
        if self.train_params.recon_loss_type == losses.LossType.L1:
            loss_preview = recon_l1(img_pred, img_target)
        elif self.train_params.recon_loss_type == losses.LossType.L2:
            loss_preview = recon_mse(img_pred, img_target)
        elif self.train_params.recon_loss_type == losses.LossType.HUBER:
            loss_preview = losses.huber_loss(img_pred, img_target, size_average=True, 
                                     delta=self.train_params.recon_huber_delta)
        else:
            raise UserWarning("Reconstruction loss type unknown!")
        
        # Joint position loss
        loss_joints = losses.joint_pos_huber_loss(joints_pred, joints_target,
                                                  ignore_zero_label_samples=True)
                  
        self.log.add_scalar_value("train-loss-joints", loss_joints.data[0], 
                                  wall_time=time.clock(), step=step_id)
        self.log.add_scalar_value("train-loss-preview", loss_preview.data[0], 
                                  wall_time=time.clock(), step=step_id)
        
        return loss_preview + self.train_params.lambda_supervisedloss * loss_joints
        
    
    def loss_function_test(self, estimate, target):
        return losses.joint_pos_distance_loss(estimate, target)
    
    
    def train(self, model, num_epochs, 
              lr=1e-3, weight_decay=0.001, optim_type=0,
              out_crop_size=None):
        if optim_type == 0:
            optimizer = optim.Adam(model.parameters(), 
                                   lr=lr, weight_decay=weight_decay)
        elif optim_type == 1:
            optimizer = optim.RMSprop(model.parameters(), 
                                      lr=lr, weight_decay=weight_decay)
        elif optim_type == 2:
            optimizer = optim.SGD(model.parameters(), 
                                  lr=lr, momentum=0.9, weight_decay=weight_decay)
                                  
        scheduler = lr_scheduler.MultiStepLR(optimizer, 
                                             milestones=self.train_params.lr_decay_steps,
                                             gamma=self.train_params.lr_decay)
        
        if self.train_params.do_save_model \
                and not os.path.exists(
                os.path.dirname(self.train_params.model_filepath)):
            os.makedirs(os.path.dirname(self.train_params.model_filepath))
                                      
        model.train()
        
        t_0 = time.time()
        
        val_error_best = np.inf
        epoch_best = np.NaN
        for epoch in range(1, num_epochs + 1):
            scheduler.step()
            
            t_0_epoch = time.time()
            self.train_epoch(epoch, model, optimizer, out_crop_size)
            t_1_epoch = time.time()
            val_error = self.test_epoch(epoch, model, out_crop_size)
            
            print("Time (wall) for training epoch: {}".format(t_1_epoch - t_0_epoch))
            
            # Store (intermediate) model
            if ((epoch % self.train_params.save_model_epoch_interval) == 0) \
                    and self.train_params.do_save_model:
                filename = self.train_params.model_filepath + "_epoch{}".format(epoch)
                torch.save(model.state_dict(), filename)
                
            if self.train_params.do_use_best and (val_error < val_error_best):
                model_best = copy.deepcopy(model)
                val_error_best = val_error
                epoch_best = epoch
                
        t_1 = time.time()
        
        if self.train_params.do_use_best:
            model = copy.deepcopy(model_best)
            print("Best model from epoch {}/{} (val. error: {})".format(
                epoch_best, num_epochs, val_error_best))
           
        if self.train_params.do_save_model:
            torch.save(model.state_dict(), self.train_params.model_filepath)
    
        print("Time (wall) for train: {}".format(t_1 - t_0))
        
        
    def train_epoch(self, epoch, model, optimizer, out_crop_size):
        model.train()
        train_loss = 0
        num_samples_done = 0
        for batch_idx, (data_cam1, data_cam2, data_cam3, joints_target,_,_,_,_,_, com_cam1,_,_) \
                in enumerate(self.train_loader):
            list_camviews = [data_cam1, data_cam2, data_cam3]
            # Select input image
            data_input = data_cam1
            com_input = torch.from_numpy(
                self.train_loader.dataset.normalize_and_jitter_3D(com_cam1.numpy()))
            # Select target image
            data_target = list_camviews[self.train_params.output_cam_ids_train[0] - 1] # 1-based IDs
                
            data_input, data_target = Variable(data_input), Variable(data_target)
            joints_target = Variable(joints_target)
            com_input = Variable(com_input)
            if self.train_params.cuda:
                data_input, data_target = data_input.cuda(), data_target.cuda()
                com_input = com_input.cuda()
                joints_target = joints_target.cuda()
            optimizer.zero_grad()
            joints_pred, view_pred, _ = model(data_input, com_input)
            if not out_crop_size is None:
                data_target = output.scale_batch_avg(data_target, out_crop_size)
            step = (epoch-1) * len(self.train_loader) + batch_idx # for logging
            loss = self.loss_function_train(joints_pred, joints_target, view_pred, data_target, step)
            loss.backward()
            train_loss += loss.data[0]
            optimizer.step()
            
            # Logging
            num_samples_done += len(data_input)
            if batch_idx % self.train_params.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss/sample: {:.6f}'.format(
                    epoch, num_samples_done, len(self.train_loader.sampler),
                    100. * (batch_idx+1) / len(self.train_loader),
                    loss.data[0]))
                self.log.add_scalar_value("train-loss", 
                                          loss.data[0], 
                                          wall_time=time.clock(), step=step)
    
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.train_loader)))

        if (epoch % 4) == 0:
            # Write some target images
            out_filepath = os.path.join(self.train_params.out_path_results_images, 
                                        "target_samples.png")
            vutils.save_image(data_target.data, out_filepath, normalize=True)
            # Write some generated images
            out_filepath = os.path.join(self.train_params.out_path_results_images, 
                                        "generated_samples_e{}.png".format(epoch))
            vutils.save_image(view_pred.data, out_filepath, normalize=True)


    def test_epoch(self, epoch, model, out_crop_size):
        model.eval()
        test_loss = 0
        for data_cam1, data_cam2, data_cam3, joints_target,_,_,_,_,_, com_cam1,_,_ \
                in self.val_loader:
            list_camviews = [data_cam1, data_cam2, data_cam3]
            # Select input image
            data_input = data_cam1
            com_input = torch.from_numpy(
                self.train_loader.dataset.normalize_3D(com_cam1.numpy()))
            # Select target image
            data_target = list_camviews[self.train_params.output_cam_ids_test[0] - 1] # 1-based IDs
                
            if self.train_params.cuda:
                data_input, data_target = data_input.cuda(), data_target.cuda()
                com_input = com_input.cuda()
                joints_target = joints_target.cuda()
            data_input = Variable(data_input, volatile=True)
            data_target = Variable(data_target, volatile=True)
            com_input = Variable(com_input, volatile=True)
            joints_target = Variable(joints_target, volatile=True)
            joints_pred, view_pred, _ = model(data_input, com_input)
            if not out_crop_size is None:
                data_target = output.scale_batch_avg(data_target, out_crop_size)
            test_loss += (self.loss_function_test(joints_pred, joints_target).data[0]
                * joints_target.size()[0])
    
        test_loss /= len(self.val_loader.sampler)
        # Denormalize the loss, to have it corresponding to test error
        test_loss *= self.val_loader.dataset.args_data.crop_size_3d_tuple[2]
        if not self.val_loader.dataset.args_data.do_norm_zero_one:
            test_loss /= 2.0
        # Output
        print('====> Validation set loss: {:.4f}'.format(test_loss))
        step = epoch * len(self.train_loader)
        self.log.add_scalar_value("val-loss", test_loss, 
                                wall_time=time.clock(), step=step)
                                
        return test_loss
        