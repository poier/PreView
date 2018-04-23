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

from nets.DiscriminatorNetDcGan import ConditioningType

import util.output as output
import util.losses as losses

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils

import numpy as np

import copy
import time
import os.path



#reconstruction_function = nn.L1Loss(size_average=False)
reconstruction_function = nn.SmoothL1Loss(size_average=True)
#reconstruction_function = nn.MSELoss(size_average=False)
criterion_mse = nn.MSELoss()


class AdversarialViewPredAndHaPeTrainer(object):
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
        
    
    def loss_function_train(self, joints_pred, joints_target, img_pred, img_target, 
                            discriminator_pred, discriminator_target, step_id):
        loss_discriminator = self.loss_function_discriminator(
            discriminator_pred, discriminator_target)
        loss_preview = reconstruction_function(img_pred, img_target)
        loss_joints = losses.joint_pos_huber_loss(joints_pred, joints_target,
                                                  ignore_zero_label_samples=True)
                  
        self.log.add_scalar_value("train-loss-joints", loss_joints.data[0], 
                                  wall_time=time.clock(), step=step_id)
        self.log.add_scalar_value("train-loss-preview", loss_preview.data[0], 
                                  wall_time=time.clock(), step=step_id)
        self.log.add_scalar_value("train-loss-adversarial", loss_discriminator.data[0],
                                  wall_time=time.clock(), step=step_id)
        
        return loss_preview \
            + self.train_params.lambda_supervisedloss * loss_joints \
            + self.train_params.lambda_adversarialloss * loss_discriminator
        
        
    def loss_function_discriminator(self, prediction, target):
        return criterion_mse(prediction, target)
        
    
    def loss_function_test(self, estimate, target):
        return losses.joint_pos_distance_loss(estimate, target)
    
    
    def train(self, model, model_discriminator, num_epochs, 
              lr=1e-3, weight_decay=0.001, optim_type=0,
              out_crop_size=None):
        if optim_type == 0:
            optimizer = optim.Adam(model.parameters(), 
                                   lr=lr, weight_decay=weight_decay)
            optimizer_d = optim.Adam(model_discriminator.parameters(), 
                                   lr=lr, weight_decay=weight_decay)
        elif optim_type == 1:
            optimizer = optim.RMSprop(model.parameters(), 
                                      lr=lr, weight_decay=weight_decay)
            optimizer_d = optim.RMSprop(model_discriminator.parameters(), 
                                      lr=lr, weight_decay=weight_decay)
        elif optim_type == 2:
            optimizer = optim.SGD(model.parameters(), 
                                  lr=lr, momentum=0.9, weight_decay=weight_decay)
            optimizer_d = optim.SGD(model_discriminator.parameters(), 
                                  lr=lr, momentum=0.9, weight_decay=weight_decay)
                                  
        if self.train_params.do_save_model \
                and not os.path.exists(
                os.path.dirname(self.train_params.model_filepath)):
            os.makedirs(os.path.dirname(self.train_params.model_filepath))
        
        t_0 = time.time()
        
        val_error_best = np.inf
        epoch_best = np.NaN
        for epoch in range(1, num_epochs + 1):
#            self.adjust_learning_rate(optimizer, lr, (epoch-1), 50)
            t_0_epoch = time.time()
            self.train_epoch(epoch, model, model_discriminator, optimizer, optimizer_d, out_crop_size)
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
                model_discriminator_best = copy.deepcopy(model_discriminator)
                val_error_best = val_error
                epoch_best = epoch
            
        t_1 = time.time()
           
        if self.train_params.do_use_best:
            model = copy.deepcopy(model_best)
            model_discriminator = copy.deepcopy(model_discriminator_best)
            print("Best model from epoch {}/{} (val. error: {})".format(
                epoch_best, num_epochs, val_error_best))
            
        if self.train_params.do_save_model:
            torch.save(model.state_dict(), self.train_params.model_filepath)
    
        print("Time (wall) for train: {}".format(t_1 - t_0))
        
        
    def train_epoch(self, epoch, model, model_d, 
                    optimizer, optimizer_d, out_crop_size):
        model.train()
        model_d.train()
        
        label_real_d = 1
        label_generated_d = 0
        
        target_d = torch.FloatTensor(self.train_params.batch_size)
        if self.train_params.cuda:
            target_d = target_d.cuda()
        
        train_loss = 0
        num_samples_done = 0
        for batch_idx, (data_cam1, data_cam2, data_cam3, joints_target,_,_,_,_,_, com_cam1,_,_) \
                in enumerate(self.train_loader):
            batch_size = data_cam1.size(0)
            list_camviews = [data_cam1, data_cam2, data_cam3]
            # Select input image
            data_input = data_cam1
            com_input = com_cam1
            # Select target image
            data_target = list_camviews[self.train_params.output_cam_ids_train[0] - 1] # 1-based IDs
                
            data_input, data_target = Variable(data_input), Variable(data_target)
            joints_target = Variable(joints_target)
            com_input = Variable(com_input)
            if self.train_params.cuda:
                data_input, data_target = data_input.cuda(), data_target.cuda()
                com_input = com_input.cuda()
                joints_target = joints_target.cuda()

            # Update discriminator
            model_d.zero_grad()
            
            # Mini-batch with real data
            # Create label vector for batch with real data
            target_d.resize_(batch_size).fill_(label_real_d)
            target_d_v = Variable(target_d)
            # Compute fprop/backprop
            data_in_discr = self.get_discriminator_input(data_target, 
                                                         data_input, 
                                                         joints_target)
            prediction_d = model_d(data_in_discr)
            errD_real = self.loss_function_discriminator(prediction_d, target_d_v)
            errD_real.backward()
            D_x = prediction_d.data.mean()
            
            # Mini-batch with predicted/generated data
            joints_pred, view_pred, _ = model(data_input, com_input)
            target_d_v = Variable(target_d.fill_(label_generated_d))
#            data_cond = torch.cat((view_pred, data_input), 1) # cond. on input            
            data_in_discr = self.get_discriminator_input(view_pred, 
                                                         data_input, 
                                                         joints_pred)
            prediction_d = model_d(data_in_discr.detach())
            errD_gen = self.loss_function_discriminator(prediction_d, target_d_v)
            errD_gen.backward()
            D_G_z1 = prediction_d.data.mean()
            
            errD = errD_real + errD_gen
            optimizer_d.step()
            
            # Update generator
            optimizer.zero_grad()
            target_d_v = Variable(target_d.fill_(label_real_d))    # targets for generated samples are "real" for generator cost
            # fprop through discriminator on generated samples again but want to backprop through generator also this time, i.e., no detach()
            # detach the joints_pred for this call of discriminator (we don't want to propagate through the joint pos., do we?, the joint positions would probably just be changed to match the generated image!?)
            data_in_discr = self.get_discriminator_input(view_pred, 
                                                         data_input,
                                                         joints_pred.detach())
            prediction_d = model_d(data_in_discr)
            step = (epoch-1) * len(self.train_loader) + batch_idx # for logging
            loss = self.loss_function_train(joints_pred, joints_target, 
                                            view_pred, data_target, 
                                            prediction_d, target_d_v, step)
            loss.backward()
            D_G_z2 = prediction_d.data.mean()
            train_loss += loss.data[0]
            optimizer.step()
            
            # Logging
            num_samples_done += len(data_input)
            if batch_idx % self.train_params.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss/sample: {:.6f}'.format(
                    epoch, num_samples_done, len(self.train_loader.sampler),
                    100. * (batch_idx+1) / len(self.train_loader),
                    loss.data[0]))
                print('Loss_D: {:.4f}; Loss_G: {:.4f}; D(x): {:.4f}; D(G(z)): {:.4f} / {:.4f}'.format(
                    errD.data[0], loss.data[0], D_x, D_G_z1, D_G_z2))
                step = (epoch-1) * len(self.train_loader) + batch_idx
                self.log.add_scalar_value("train-loss", 
                                          loss.data[0], 
                                          wall_time=time.clock(), step=step)
                self.log.add_scalar_value("prediction-real", D_x, 
                                          wall_time=time.clock(), step=step)
                self.log.add_scalar_value("prediction-gen", D_G_z1, 
                                          wall_time=time.clock(), step=step)
    
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.train_loader)))
              
        if (epoch % 4) == 0:
            # Write some real images
            out_filepath = os.path.join(self.train_params.out_path_results_images, 
                                        "real_samples.png")
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
            com_input = com_cam1
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
                           
                           
#    # Use the lr-scheduler now
#    def adjust_learning_rate(self, optimizer, init_lr, epoch, decay_step):
#        """
#        Sets the learning rate to the initial LR decayed by 10 every 
#        decay_step epochs
#        """
#        lr = init_lr * (0.1 ** (epoch // decay_step))
#        for param_group in optimizer.param_groups:
#            param_group['lr'] = lr
            
            
    def get_discriminator_input(self, discr_in_image, enc_in_image, joint_pos):
        if self.train_params.discriminator_condition_type == ConditioningType.OFF:
            data = discr_in_image
        elif self.train_params.discriminator_condition_type == ConditioningType.INPUT:
            data = torch.cat((discr_in_image, enc_in_image), 1)
        elif self.train_params.discriminator_condition_type == ConditioningType.POSE:
            w = discr_in_image.size(2)
            h = discr_in_image.size(3)
            # create a channel per target dimension/joint-position 
            # with the joint-position coordinate repeated in the whole channel
            data = torch.cat((discr_in_image, 
                              joint_pos.view(joint_pos.size(0), -1).unsqueeze(
                                  2).unsqueeze(2).repeat(1,1,w,h)),
                              1)
        elif self.train_params.discriminator_condition_type == ConditioningType.INPUT_AND_POSE:
            w = discr_in_image.size(2)
            h = discr_in_image.size(3)
            # create a channel per target dimension/joint-position 
            # with the joint-position coordinate repeated in the whole channel
            data = torch.cat((discr_in_image, enc_in_image,
                              joint_pos.view(joint_pos.size(0), -1).unsqueeze(
                                  2).unsqueeze(2).repeat(1,1,w,h)),
                              1)
        else:
            raise UserWarning("ERROR: conditioning type not known")
        return data
            