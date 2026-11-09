from src import search_argparser
from src import simulate
from src import SV_configs
from src import utils
from configs import rich_lazy_search_config
from pathlib import Path
import numpy as np
from types import SimpleNamespace
from scipy import stats

# TODO: create proper configs
# make for aligned and unaligned case
# make for different task dynamics and rotations


if __name__ == "__main__":
    config_parser = search_argparser.get_parser() 
    config_args = config_parser.parse_args()

    if config_args.config_name == 'constant':
        args = SimpleNamespace(**rich_lazy_search_config.constant_args)
    elif config_args.config_name == 'rotation':
        args = SimpleNamespace(**rich_lazy_search_config.rotation_args)
        args.num_SV = config_args.num_SV
        args.input_size = args.num_SV
        args.hidden_size = args.num_SV
        args.output_size = args.num_SV
    elif config_args.config_name == 'rotation_v2':
        args = SimpleNamespace(**rich_lazy_search_config.rotation_v2_args)
    elif config_args.config_name == 'infinite_width':
        args = SimpleNamespace(**rich_lazy_search_config.infinite_width_args)
    elif config_args.config_name == 'independent_modes_init':
        args = SimpleNamespace(**rich_lazy_search_config.independent_modes_init_args)

    if config_args.aligned_student_weights:
        args.results_out_dir = f'results/richlazy_search_{config_args.config_name}_aligned/'
        args.data_out_dir = f'data/richlazy_search_{config_args.config_name}_aligned/'
        args.aligned_student_weights = True
        args.initialization_scale_range = args.initialization_scale_range/np.sqrt(args.num_SV)
    else: 
        args.results_out_dir = f'results/richlazy_search_{config_args.config_name}_unaligned/'
        args.data_out_dir = f'data/richlazy_search_{config_args.config_name}_unaligned/'
        args.aligned_student_weights = False
    
    data_path = Path(args.data_out_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    try:
        args_ = utils.load_dict(data_path / 'args.pkl')
        if utils._hash(args) != utils._hash(args_):
            raise Exception('Search with different args already exists in this directory. Choose another.')
    except FileNotFoundError:
        pass

    utils.save_dict(args, data_path / 'args.pkl')

    try:
        rich_lazy_search_tape = utils.load_dict(data_path / 'richlazy_search_tape.pkl')
    except FileNotFoundError:
        if config_args.config_name == 'dirac_delta':
            rich_lazy_search_tape = utils.get_rich_lazy_search_tape(args, len(args.trajectory_length_range), len(args.initialization_scale_range), len(args.beta_range))
        elif config_args.config_name == 'independent_modes_init':
            rich_lazy_search_tape = utils.get_rich_lazy_search_tape(args, len(args.trajectory_length_range), len(args.inout_scale_range), len(args.rec_scale_range))
        else:
            rich_lazy_search_tape = utils.get_rich_lazy_search_tape(args, len(args.trajectory_length_range), len(args.initialization_scale_range), None)
      

    if config_args.config_name == 'independent_modes_init':
        for i, traj_length in enumerate(args.trajectory_length_range):
            for j, inout_scale in enumerate(args.inout_scale_range):
                for k, rec_scale in enumerate(args.rec_scale_range):
                    args.name = f'traj_length_{traj_length}_inout_scale_{inout_scale}_rec_{rec_scale}'
                    args.inout_mode_init = [inout_scale]*args.num_SV
                    args.rec_mode_init = [rec_scale]*args.num_SV
                    args.trajectory_length = traj_length
                    args.S = SV_configs.create_SV_trajectories(args)
                    print(f"Running {args.name}.")
                    tape, data_tape = simulate.run_simulation_multiseed(args)
                    rich_lazy_search_tape.final_loss[i,j,k] = tape.loss[:,-1]
                    rich_lazy_search_tape.input_output_modes[i,j,k] = tape.input_output_modes[:,-1,:]
                    rich_lazy_search_tape.input_output_modes_mean[i,j,k] = np.mean(tape.input_output_modes[:,-1,:])
                    rich_lazy_search_tape.input_output_modes_sem[i,j,k] = stats.sem(tape.input_output_modes[:,-1,:], axis=None)
                    rich_lazy_search_tape.recurrent_modes[i,j,k] = tape.recurrent_modes[:,-1,:]
                    rich_lazy_search_tape.recurrent_modes_mean[i,j,k] = np.mean(tape.recurrent_modes[:,-1,:])
                    rich_lazy_search_tape.recurrent_modes_sem[i,j,k] = stats.sem(tape.recurrent_modes[:,-1,:], axis=None)
                    rich_lazy_search_tape.NTK_distance[i,j,k] = tape.NTK_distance
                    rich_lazy_search_tape.NTK_distance_mean[i,j,k] = np.mean(tape.NTK_distance)
                    rich_lazy_search_tape.NTK_distance_sem[i,j,k] = stats.sem(tape.NTK_distance, axis=None)
                    utils.save_dict(rich_lazy_search_tape, data_path / 'rich_lazy_search_tape.pkl') # continuously saving progress
    else:
        for i, traj_length in enumerate(args.trajectory_length_range):
            for j, init_scale in enumerate(args.initialization_scale_range):
                args.name = f'traj_length_{traj_length}_init_scale_{init_scale}'
                if args.aligned_student_weights: 
                    args.inout_mode_init = [init_scale**2]*args.num_SV 
                    args.rec_mode_init = [init_scale]*args.num_SV
                else:
                    args.initialization_scale = init_scale
                args.trajectory_length = traj_length
                if args.task_dynamics == 'diagonalized':
                    if args.eigenspace_dynamics:
                        args.D = SV_configs.create_SV_trajectories(args)
                    else:
                        args.S = SV_configs.create_SV_trajectories(args)
                print(f"Running {args.name}.")
                tape, data_tape = simulate.run_simulation_multiseed(args)
                rich_lazy_search_tape.final_loss[i,j] = tape.loss[:,-1]
                rich_lazy_search_tape.input_output_modes[i,j] = tape.input_output_modes[:,-1,:]
                rich_lazy_search_tape.input_output_modes_mean[i,j] = np.mean(tape.input_output_modes[:,-1,:])
                rich_lazy_search_tape.input_output_modes_sem[i,j] = stats.sem(tape.input_output_modes[:,-1,:], axis=None)
                rich_lazy_search_tape.recurrent_modes[i,j] = tape.recurrent_modes[:,-1,:]
                rich_lazy_search_tape.recurrent_modes_mean[i,j] = np.mean(tape.recurrent_modes[:,-1,:])
                rich_lazy_search_tape.recurrent_modes_sem[i,j] = stats.sem(tape.recurrent_modes[:,-1,:], axis=None)
                rich_lazy_search_tape.NTK_distance[i,j] = tape.NTK_distance
                rich_lazy_search_tape.NTK_distance_mean[i,j] = np.mean(tape.NTK_distance)
                rich_lazy_search_tape.NTK_distance_sem[i,j] = stats.sem(tape.NTK_distance, axis=None)
                utils.save_dict(rich_lazy_search_tape, data_path / 'rich_lazy_search_tape.pkl') # continuously saving progress