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

import argparse


def parse_arguments_generic(args=None):
    """
    Parses command-line arguments of the generic train/test script
    
    Arguments:
        args (Object, optional): existing object with attributes to which the 
            parsed arguments are added (default: None)
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description=
        'Generic script for view prediction (train, test, analysis, ...)')
    parser = add_arguments_for_generic_script(parser)
    args = parser.parse_args(namespace=args)
    
    # Adapt necessary parameters (names, types, ...)
    args.do_train = not args.no_train
    args.do_test = not args.no_test
    args.do_produce_previews = not args.no_preview_samples
    args.lambda_adversarial = args.lambda_adversarialloss
    args.do_use_best = args.do_use_best_model
    args.num_labeled_samples = args.num_samples
    args.min_sampling_prob_labeled = args.min_samp_prob
    args.crop_size_3d_tuple = (args.crop_size_3d, args.crop_size_3d, args.crop_size_3d)
    # Ensure lists
    if not type(args.lr_decay_steps) == list:
        args.lr_decay_steps = [args.lr_decay_steps]
    if not type(args.output_cam_ids_train) == list:
        args.output_cam_ids_train = [args.output_cam_ids_train]
    if not type(args.output_cam_ids_test) == list:
        args.output_cam_ids_test = [args.output_cam_ids_test]
    args.needed_cam_ids_train = [args.input_cam_id_train]
    args.needed_cam_ids_train.extend(args.output_cam_ids_train)
    args.needed_cam_ids_test = [args.input_cam_id_test]
    args.needed_cam_ids_test.extend(args.output_cam_ids_test)
    
    return args
    

def add_arguments_for_generic_script(parser):
    parser.add_argument('--net-type', type=int, default=2,
                        help='Network Type (see nets.NetFactory.NetType)')
    parser.add_argument('--num-embedding-dims', type=int, default=50,
                        help='Number of dimensions of latent representation/\
                        embedding (default: 50).')
    parser.add_argument('--optim-type', type=int, default=0,
                        help='Optimizer (e.g., 0: Adam (default), 1: RMSprop, 2: SGD)')
    parser.add_argument('--recon-loss-type', type=int, default=0,
                        help='Loss type for (image) reconstruction loss \
                        (default: 0 (=L1-loss), see util.losses.LossType)')
    parser.add_argument('--recon-huber-delta', type=float, default=0.2, metavar='huber-delta',
                        help='delta for huber loss if used for reconstruction (default: 0.2)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='g',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr-decay', type=float, default=1.0, metavar='g_d',
                        help='decay of learning rate after lr-decay-step epochs \
                        (default: 1.0, i.e., no decay)')
    parser.add_argument('--lr-decay-steps', type=int, default=50, nargs='*', metavar='s_g_d',
                        help='list of steps/epochs after which lr is adapted, \
                        i.e., multiply learning rate with lr-decay at every step, \
                        see PyTorch docs for lr_scheduler.MultiStepLR')
    parser.add_argument('--lambda-supervisedloss', type=float, default=10, metavar='l_s',
                        help='weight for supervised loss term (default: 10)')
    parser.add_argument('--training-type', type=int, default=0,
                        help='Training Type. Standard or adversarial \
                        (default: 0 (=standard), see trainer.SuperTrainer.TrainingType)')
    parser.add_argument('--discriminator-condition-type', type=int, default=1,
                        help='Discriminator condition type. E.g., \
                        no conditioning, input, pose, ... (default: 1 (=input) \
                        see nets.DiscriminatorNetDcGan.ConditioningType)')
    parser.add_argument('--lambda-adversarialloss', type=float, default=1e-2, metavar='l_a',
                        help='weight for adversarial loss term (default: 1e-2)')
    parser.add_argument('--weight-decay', type=float, default=0.001, metavar='d',
                        help='weight decay (default: 1e-3)')
    parser.add_argument('--output-cam-ids-train', type=int, default=3, nargs='*',
                        help='output camera IDs for training set (default: 3)')
    parser.add_argument('--output-cam-ids-test', type=int, default=2, nargs='*',
                        help='output camera IDs for test set (default: 2)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='n_e',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num-samples', type=int, default=100000, metavar='N',
                        help='number of labeled training samples used at most \
                        (default: 100000)')
    parser.add_argument('--min-samp-prob', type=float, default=0.5, metavar='p',
                        help='minimum sampling probability for labeled samples, \
                        i.e., how large the approx. ratio of labeled samples \
                        is at least in each mini-batch (default: 0.5)')
    parser.add_argument('--crop-size-3d', type=int, default=250,
                        help='crop size in 3d, in mm, same is used in each dimension \
                        (default: 250).')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='do NOT train on (cuda) GPU? (default: False, \
                        i.e., do training on GPU')
    parser.add_argument('--seed', type=int, default=123456789, metavar='S',
                        help='random seed (default: 123456789)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='n_i',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--exp-name', default="run00_default",
                        help='name for the experiment (default: "run00_default")')
    parser.add_argument('--no-train', action='store_true', default=False,
                        help='do NOT train? (default: False, i.e., do training)')
    parser.add_argument('--no-test', action='store_true', default=False,
                        help='do NOT test? (default: False, i.e., do testing)')
    parser.add_argument('--model-filepath', default="",
                        help='filename (and full path) to store/load the model.\
                        default location is within the respective result folder \
                        (in folder model: "./model/model.mdl").')
    parser.add_argument('--out-base-path', default="",
                        help='(absolute) base path for --out-path; default = "" \
                        meaning that the directory containing the script is used.')
    parser.add_argument('--out-path', default="./results/default",
                        help='relative path to store outputs (results, model, ...).')
    parser.add_argument('--no-preview-samples', action='store_true', default=False,
                        help='do NOT produce view prediction samples? \
                        (default: False, i.e., do produce predictions)')
                        
    return parser
    