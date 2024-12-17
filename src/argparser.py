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
    
def str2list(v):
    return v.split(',')

def str2floatlist(v):
    return float(v.split(','))

def get_parser():
    parser = argparse.ArgumentParser(description='linear_RNN')

    parser.add_argument('--data_out_dir', type=str, default='data/',
                        help='Path to output folder for saving data.')
    parser.add_argument('--results_out_dir', type=str, default='results/',
                        help='Path to output folder for saving results.')
    parser.add_argument('--num_seeds', type=int, default=5,
                        help='Num model seeds to perform experiment on')
    parser.add_argument('--override_load', type=str2bool, default=False,
                        help="Overrides load and re-runs simulation even if arguments match previous file.")
    parser.add_argument('--log_every_e_epoch', type=int, default=100,
                        help='Set arg to 1 to log for every epoch')
    parser.add_argument('--log_H', type=str2bool, default=False,
                        help='Whether to log hidden layer and its gradient')
    parser.add_argument('--compute_NTK', type=str2bool, default=False)
    parser.add_argument('--nonlinear_RNN', type=str2bool, default=False)

    parser.add_argument('--trajectory_length', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples trained on')
    parser.add_argument('--num_epochs', type=int, default=5000,
                        help='Number of epochs trained on')
    parser.add_argument('--lr', type=float, default=10,
                        help='Learning rate')
    parser.add_argument('--initialization_scale', type=float, default=1e-2, ## TODO: make it be based on the number of neurons?
                        help='Initialization scale')

    parser.add_argument('--input_size', type=int, default=4,
                        help='Input dimensionality')
    parser.add_argument('--hidden_size', type=int, default=4,
                        help='Hidden layer size')
    parser.add_argument('--output_size', type=int, default=4,
                        help='Output dimensionality')
    # parser.add_argument('--s_curriculum', type=str, default='',
    #                     help='Flag to specify which curriculum being trained on.')
    parser.add_argument('--S_config', type=str2list, default=['constant','constant','constant','constant'],
                        help='Flag to specify what configuration of SV trajectories being trained on for each dimension.') 
    parser.add_argument('--delta', type=str2floatlist, default=[1,1,1,1])
    parser.add_argument('--lambda_0', type=str2floatlist, default=[1,1,1,1])
    parser.add_argument('--S_T', type=str2floatlist, default=[1,1,1,1])
    parser.add_argument('--S_TeqS_t', type=str2bool, default=True,
                        help='Flag to specify whether S_T evolves according to S_t dynamics or is separate. Overrides S_T parameter if mismatch.')

    
    parser.add_argument('--name', type=str, default=None,
                        help='Name of run (for saving)')
    
    parser.add_argument('--aligned_student_weights', type=str2bool, default=True)
    parser.add_argument('--zero_balanced', type=str2bool, default=True,
                        help='')
    parser.add_argument('--frozen_hidden_layer', type=str2bool, default=False,
                        help='')
    parser.add_argument('--frozen_input_output', type=str2bool, default=False,
                        help='')
    parser.add_argument('--single_output_y', type=str2bool, default=True,
                        help='')

    parser.add_argument('--inout_mode_init', type=str2floatlist, default=[1,1,1,1],
                        help='')
    parser.add_argument('--in_mode_init', type=str2floatlist, default=[1,1,1,1],
                        help='')
    parser.add_argument('--out_mode_init', type=str2floatlist, default=[1,1,1,1],
                        help='')
    parser.add_argument('--rec_mode_init', type=str2floatlist, default=[1,1,1,1], #default=[1,1,-0.99,0.1],
                        help='')
    parser.add_argument('--num_SV', type=int, default=4)
    # parser.add_argument('--sv', type=list, default=[2,1,1,0.05])
    
    

    # parser.add_argument('--t_tot', type=float, default=None)
    # parser.add_argument('--T_tot', type=int, default=None)
    # parser.add_argument('--tau', type=float, default=None)
    # parser.add_argument('--dt', type=float, default=1.)
    # parser.add_argument('--T_tape', type=int, default=-1)
    # parser.add_argument('--dt_tape', type=float, default=None)


    return parser