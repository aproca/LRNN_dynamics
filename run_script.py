from src import argparser
from src import simulate
from src import SV_configs
from src import utils
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    parser = argparser.get_parser() 
    args = parser.parse_args()
    # commment-out and hard-code for non-default trajectories
    args.S = SV_configs.create_SV_trajectories(args)
    data_path = Path(args.data_out_dir) / args.name

    # Prevents re-running simulation if already exists
    run = True
    try:
        if args.override_load:
            raise TypeError
        args_ = utils.load_dict(data_path / 'args.pkl')
        if utils._hash(args) != utils._hash(args_):
            raise ValueError
        run = False
        tape = utils.load_dict(data_path / 'tape.pkl')
        data_tape = utils.load_dict(data_path / 'data_tape.pkl')
        # logger.info(f"Data found at {data_path}. Loaded data.")
        print(f"Data found at {data_path}. Loaded data.")
    except (FileNotFoundError, ValueError, TypeError) as e:
        if type(e) == FileNotFoundError:
            print(f"Data not found at {data_path}. Running simulation.")
        elif type(e) == ValueError:
            print(f"Data found at {data_path} but with different arguments. Re-running simulation.")
        elif type(e) == TypeError:
            print(f"{e}. Override argument on. Re-running simulation.")

    if run:
        tape, data_tape = simulate.run_simulation_multiseed(args)

    # Saving data and config      
    data_path.mkdir(parents=True, exist_ok=True)
    utils.save_dict(tape, data_path / 'tape.pkl')
    utils.save_dict(data_tape, data_path / 'data_tape.pkl')
    utils.save_dict(args, data_path / 'args.pkl')

        
    # Loading
    # tape = utils.load_dict(data_path / 'tape.pkl')
    # data_tape = utils.load_dict(data_path / 'data_tape.pkl')
    # args = utils.load_dict(data_path / 'args.pkl')