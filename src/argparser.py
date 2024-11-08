import argparse
import numpy as np
from typing import Callable
from dataclasses import field

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = argparse.ArgumentParser(description='linear_RNN')

    parser.add_argument('--data_out_dir', type=str, default='data/',
                        help='Path to output folder for saving data.')
    parser.add_argument('--results_out_dir', type=str, default='results/',
                        help='Path to output folder for saving results.')
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='Num model seeds to perform experiment on')

    parser.add_argument('--trajectory_length', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples trained on')
    parser.add_argument('--num_epochs', type=int, default=5000,
                        help='Number of epochs trained on')
    parser.add_argument('--lr', type=float, default=10,
                        help='Learning rate')
    parser.add_argument('--initialization_scale', type=float, default=1e-1,
                        help='Initialization scale')

    parser.add_argument('--input_size', type=int, default=4,
                        help='Input dimensionality')
    parser.add_argument('--hidden_size', type=int, default=4,
                        help='Hidden layer size')
    parser.add_argument('--output_size', type=int, default=4,
                        help='Output dimensionality')
    parser.add_argument('--s_curriculum', type=str, default='',
                        help='Flag to specify which curriculum being trained on.')
    
    parser.add_argument('--name', type=str, default=None,
                        help='Name of run (for saving)')
    
    parser.add_argument('--zerobalanced', type=str2bool, default=True,
                        help='')
    parser.add_argument('--frozen_hidden_layer', type=str2bool, default=True,
                        help='')
    parser.add_argument('--frozen_input_output', type=str2bool, default=False,
                        help='')
    parser.add_argument('--single_output_y', type=str2bool, default=True,
                        help='')

    parser.add_argument('--inout_mode_init', type=list, default=[1,1,1,1],
                        help='')
    parser.add_argument('--in_mode_init', type=list, default=[1,1,1,1],
                        help='')
    parser.add_argument('--out_mode_init', type=list, default=[1,1,1,1],
                        help='')
    parser.add_argument('--rec_mode_init', type=list, default=[1,1,-0.99,0.1],
                        help='')
    parser.add_argument('--num_SV', type=int, default=4)
    parser.add_argument('--sv', type=list, default=[2,1,1,0.05])
    
    

    # parser.add_argument('--t_tot', type=float, default=None)
    # parser.add_argument('--T_tot', type=int, default=None)
    # parser.add_argument('--tau', type=float, default=None)
    # parser.add_argument('--dt', type=float, default=1.)
    # parser.add_argument('--T_tape', type=int, default=-1)
    # parser.add_argument('--dt_tape', type=float, default=None)


    return parser