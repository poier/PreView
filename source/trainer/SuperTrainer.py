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

from nets.NetFactory import NetType

from trainer.ViewPredAndHaPeTrainer import ViewPredAndHaPeTrainer
from trainer.AdversarialViewPredAndHaPeTrainer import AdversarialViewPredAndHaPeTrainer

from enum import IntEnum


#%% General definitions
class TrainingType(IntEnum):
    STANDARD = 0
    ADVERSARIAL = 2
    
    @classmethod
    def is_training_type_adversarial(self, training_type):
        return_val = False
        if training_type == self.ADVERSARIAL:
            return_val = True
        return return_val
        

#%%
class SuperTrainer(object):
    """
    """
    
    def __init__(self):
        """
        Initialize trainer
        """
        
    
    def train(self, model, train_loader, val_loader, args, tb_log,
              model_discriminator=None):
        """
        """
        if args.net_type == NetType.PREVIEW_SEMI_LINEAR:
            if not args.training_type == TrainingType.ADVERSARIAL:
                trainer = ViewPredAndHaPeTrainer(train_loader, val_loader, 
                                                 tb_log, args)
                trainer.train(model, args.epochs, 
                              lr=args.lr, weight_decay=args.weight_decay, 
                              optim_type=args.optim_type)
                              
            else:
                trainer = AdversarialViewPredAndHaPeTrainer(train_loader, val_loader, 
                                                            tb_log, args)
                trainer.train(model, model_discriminator, args.epochs, 
                              lr=args.lr, weight_decay=args.weight_decay, 
                              optim_type=args.optim_type)
                              
        else:
            raise UserWarning("No Training for specified net-type implemented.")
            