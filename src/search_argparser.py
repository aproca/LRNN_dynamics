import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
# TODO: add args for rich and lazy searches

def get_parser():
    parser = argparse.ArgumentParser(description='SV_search')

    parser.add_argument('--trajectory_length', type=int, default=10)
    parser.add_argument('--config_name', type=str, default='constant')
    parser.add_argument('--aligned_student_weights', type=str2bool, default=True)
    parser.add_argument('--num_SV', type=int, default=4)
    return parser