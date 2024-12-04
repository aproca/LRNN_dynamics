import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='SV_search')

    parser.add_argument('--trajectory_length', type=int, default=10)
    parser.add_argument('--config_name', type=str, default='constant')
    return parser