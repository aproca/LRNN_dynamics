from src import argparser
from src import search_argparser
from src import simulate
from src import SV_configs
from src import utils
from configs import SV_search_config
from pathlib import Path
import numpy as np
from types import SimpleNamespace
from scipy import stats


if __name__ == "__main__":
    config_parser = search_argparser.get_parser() 
    config_args = config_parser.parse_args()

    if config_args.config_name == 'constant':
        args = SimpleNamespace(**SV_search_config.exponential_args)
    elif config_args.config_name == 'exponential':
        args = SimpleNamespace(**SV_search_config.exponential_args)
    elif config_args.config_name == 'inverse_exponential':
        args = SimpleNamespace(**SV_search_config.inverse_exponential_args)

    args.trajectory_length = config_args.trajectory_length
    args.results_out_dir = f'results/SV_search_{config_args.config_name}_nosave_T{config_args.trajectory_length}/'
    args.data_out_dir = f'data/SV_search_{config_args.config_name}_nosave_T{config_args.trajectory_length}/'
    
    data_path = Path(args.data_out_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        args_ = utils.load_dict(data_path / 'args.pkl')
        if utils._hash(args) != utils._hash(args_):
            raise Exception('SV search with different args already exists in this directory. Choose another.')
    except FileNotFoundError:
        pass

    utils.save_dict(args, data_path / 'args.pkl')

    try:
        SV_search_tape = utils.load_dict(data_path / 'SV_search_tape.pkl')
    except FileNotFoundError:
        SV_search_tape = utils.get_SV_search_tape(args)
    

    for i, lambda_0 in enumerate(args.lambda_0_range):
        for j, S_T in enumerate(args.S_T_range):
            args.name = f'lambda_0_{lambda_0}_S_T_{S_T}'
            args.lambda_0 = [lambda_0]*args.num_SV
            args.S_T = [S_T]*args.num_SV
            args.S = SV_configs.create_SV_trajectories(args)
            print(f"Running {args.name}.")
            tape, data_tape = simulate.run_simulation_multiseed(args)
            SV_search_tape.final_loss[i,j] = tape.loss[:,-1]
            SV_search_tape.input_output_modes[i,j] = tape.input_output_modes[:,-1,:]
            SV_search_tape.input_output_modes_mean[i,j] = np.mean(tape.input_output_modes[:,-1,:])
            SV_search_tape.input_output_modes_sem[i,j] = stats.sem(tape.input_output_modes[:,-1,:], axis=None)
            SV_search_tape.recurrent_modes[i,j] = tape.recurrent_modes[:,-1,:]
            SV_search_tape.recurrent_modes_mean[i,j] = np.mean(tape.recurrent_modes[:,-1,:])
            SV_search_tape.recurrent_modes_sem[i,j] = stats.sem(tape.recurrent_modes[:,-1,:], axis=None)
            SV_search_tape.dd_term[i,j] = utils.compute_data_driven_term(args, args.S[:,0,0], SV_search_tape.input_output_modes_mean[i,j], SV_search_tape.recurrent_modes_mean[i,j])
            SV_search_tape.eff_reg_term[i,j] = utils.compute_effective_regularization_term(args, SV_search_tape.input_output_modes_mean[i,j], SV_search_tape.recurrent_modes_mean[i,j])
        

            utils.save_dict(SV_search_tape, data_path / 'SV_search_tape.pkl') # continuously saving progress