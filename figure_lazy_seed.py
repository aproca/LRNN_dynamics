from src import SV_configs
from src import utils
from src import linear_RNN
import numpy as np
import pandas as pd
from types import SimpleNamespace
import matplotlib.pyplot as plt


def generate_unaligned_teacher_data(args, variance):  # (one seed)
    X = np.random.normal(0, 0.01, size=(args.num_samples, args.trajectory_length * args.input_size)).astype(float)
    X = utils.whiten(X).reshape(args.num_samples, args.trajectory_length, args.input_size)
    teacher_W_x = np.random.normal(loc=0.0, scale=variance, size=(args.hidden_size, args.input_size))
    teacher_W_h = np.random.normal(loc=0.0, scale=variance, size=(args.hidden_size, args.hidden_size))
    teacher_W_y = np.random.normal(loc=0.0, scale=variance, size=(args.output_size, args.hidden_size))
    h_t = np.zeros((args.num_samples, args.hidden_size))
    h_trajectory = np.zeros((args.num_samples, args.trajectory_length, args.hidden_size))
    for t in range(args.trajectory_length):
        h_t = np.einsum("ij,bj->bi", teacher_W_h, h_t) + np.einsum("ij,bj->bi", teacher_W_x, X[:, t])
    y_t = np.einsum("ij,bj->bi", teacher_W_y, h_t)
    y = y_t.reshape(y_t.shape[0], 1, -1)

    return X, y, teacher_W_x, teacher_W_h, teacher_W_y

def kernel_distance(A, B):
    # Extract matrices K_t1 and K_t2
    K_t1 = A
    K_t2 = B
    # Compute the Frobenius inner product
    inner_product = np.sum(K_t1 * K_t2)
    # Compute the Frobenius norms
    norm_t1 = np.linalg.norm(K_t1, 'fro')
    norm_t2 = np.linalg.norm(K_t2, 'fro')
    # Compute the similarity measure
    S_t1_t2 = 1 - inner_product / (norm_t1 * norm_t2)
    return S_t1_t2



lenght =[1,2]
seeds = [0, 1, 2,]
scales = [0.01, 0.1,0.3, 0.5, 0.7]
# TO: for differnet time steps
NTK_a = {}
losses = {}
kernel_distances = {}
kernel_distances_seeds = {}
reshaped_NTK_3D_traj_1_seed = {}
for s in scales:
    for l in lenght:
        for seed in seeds:
            traj_1_args = {
                'data_out_dir': 'data/',
                'results_out_dir': 'results/',
                'name': 'unaligned',
                'num_seeds': 1,
                # to run over more seeds which need to be specified as arguments (see simulate.py for example)
                'override_load': False,
                'trajectory_length': l,
                'num_samples': 50,
                'num_epochs': 30000,
                'lr': 0.1,
                'lambda': 1,
                'initialization_scale': 0.01,
                # (variance of the 0-mean initialized weights when the network is unaligned)
                'input_size': 4,
                'hidden_size': 4,
                'output_size': 4,
                'S_config': ['constant', 'constant', 'constant', 'constant'],
                # data dynamics: constant (lambda_t=1-- overrides the lambda_0 argument), exponential (lambda_t=lambda^t), inverse-exponential (lambda_t=lambda^{T-t})
                'delta': [1, 1, 1, 1],  # data dynamics scaling-term (i.e., s_t = delta*lambda_t) for each SV dimension
                'lambda_0': [1, 1, 1, 1],  # data dynamics temporal-term
                'S_T': [1, 1, 1, 1],  # SV at last timestep (feedforward)
                'S_TeqS_t': True,
                'relative_initialization': True,
                # whether SV at last timestep follows the data dynamics (if true, ignores S_T argument)
                'zero_balanced': False,  # if true, sets a=c at initialization (for aligned weights)
                'frozen_hidden_layer': False,  # freezes learning on recurrent layer
                'frozen_input': False,
                'frozen_output': False,# freezes learning on input-output layers
                'single_output_y': True,
                # only computes loss on y_T (ignores y_1 to y_{T-1}); code not compatible with the multi-output case at the moment
                'inout_mode_init': [0.7, 0.7, 0.7, 0.7],
                # input-output mode initialization for each dimension (when weights are aligned and input-output are zero-balanced)
                'in_mode_init': [s,s, s, s],
                # input mode initialization for each dimension (when input-output weights are not zero-balanced)
                'out_mode_init': [0.7, 0.7, 0.7, 0.7],  # output ""
                'rec_mode_init': [0.1, 0.1, 0.1, 0.1],  # recurrent ""
                'num_SV': 4,  # num of SV dimensions (should be equal to input=hidden=output sizes)
                'aligned_student_weights': True,
                'log_H': False,
                'log_y_hat': False,
                'compute_NTK': True,
                'nonlinear_RNN': False,
                'log_every_e_epoch': 100,  # how many gradient steps until logging metrics
                'subsample_NTK': False,  # whether to use a subsample of data when computing NTK
                'single_output_rotation': False,
            }
            traj_1_args = SimpleNamespace(**traj_1_args)
            traj_1_tape = utils.get_tape_module(traj_1_args)
            traj_1_data_tape = utils.get_data_tape(traj_1_args)
            utils.set_seeds(seed)
            seed_id = 0
            ## --- generating tasks ----
            ## example 1: generating data from SV data dynamics specified in arguments
            traj_1_args.S = SV_configs.create_SV_trajectories(traj_1_args)
            traj_1_data_tape = utils.generate_data_multiseed(traj_1_args, traj_1_data_tape, seed_id)

            ## example 2: generating data from a random unaligned teacher (or could be anything else)
            # traj_1_X, traj_1_y, traj_1_t_W_x, traj_1_t_W_h, traj_1_t_W_y = generate_unaligned_teacher_data(traj_1_args, variance=0.1)
            # traj_1_data_tape.X[seed] = traj_1_X
            # traj_1_data_tape.y[seed] = traj_1_y

            ## --- generating LRNN ----
            ## example 1: aligned RNN (NOTE: not set up to align to any given task; has to be generated from SV data dynamics specified in arguments)
            ## so don't combine aligned RNN + unaligned teacher unless new implementation
            traj_1_tape = utils.initialize_student_multiseed(traj_1_args, traj_1_data_tape, seed_id, traj_1_tape)

            # example 2: unaligned RNN
            # traj_1_tape = utils.initialize_unaligned_student(traj_1_args, seed, traj_1_tape)
            # --- model initialization, training, modes, NTK ---
            traj_1_model = linear_RNN.LinearRNN_multiseed(traj_1_args, traj_1_tape, seed_id)
            traj_1_model.train(traj_1_data_tape.X[seed_id], traj_1_data_tape.y[seed_id])
            traj_1_model.project_SVD_space(
                traj_1_data_tape)  # computes modes for aligned network; computes SV for unaligned network
            traj_1_model.compute_NTK(traj_1_data_tape.X[seed_id], traj_1_data_tape.y[seed_id], seed_id)
            traj_1_tape = traj_1_model.tape

            # ntks_empirical[(lmda, scale)] = np.array(
            #     [get_ntk(w1w1, w2w2, X, out_dim) for (w1w1, w2w2) in zip(w1w1s_empirical[(lmda, scale)], w2w2s_empirical[(lmda, scale)])])
            reshaped_NTK_3D_traj_1 = traj_1_tape.NTK[seed_id].reshape(
                int(traj_1_args.num_epochs / traj_1_args.log_every_e_epoch),
                traj_1_args.num_samples * traj_1_args.output_size,
                traj_1_args.num_samples * traj_1_args.output_size
            )
            reshaped_NTK_3D_traj_1_seed[(seed)] = reshaped_NTK_3D_traj_1
            kernel_distances_seeds[(seed)] = kernel_distance(reshaped_NTK_3D_traj_1[0, :, :],
                                                             reshaped_NTK_3D_traj_1[-1, :, :])

        average_kernel_distance_over_seeds = np.mean(list(kernel_distances_seeds.values()))
        average_reshaped_NTK_3D_traj_1 = np.mean(list(reshaped_NTK_3D_traj_1_seed.values()), axis=0)
        kernel_distances[(l, s)] = average_kernel_distance_over_seeds
        NTK_a[(l, s)] = average_reshaped_NTK_3D_traj_1
        #losses[(l, s)] = traj_1_tape.loss[seed]



l_values_a = sorted(set(key[0] for key in NTK_a))
s_values_a = sorted(set(key[1] for key in NTK_a))
steps_of_interest = [-1]  # Last step
kernel_heatmaps_a = {step: pd.DataFrame(np.nan, index=l_values_a, columns=s_values_a) for step in steps_of_interest}

# Populate the heatmaps
for step in steps_of_interest:
    for (param1, param2), distances in NTK_a.items():
        if step <= len(distances):  # Ensure the step exists in the data
            kernel_heatmaps_a[step].loc[param1, param2] = kernel_distance(distances[0, :, :], distances[step, :,
                                                                                              :])  # Adjust for 0-based indexing
            # kernel_distance(reshaped_NTK_3D_traj_1[0, :, :], reshaped_NTK_3D_traj_1[-1, :, :])

# Use the scale of the last kernel heatmap and the first loss heatmap
kernel_scale_a = kernel_heatmaps_a[steps_of_interest[-1]]  # Last kernel heatmap

vmin_kernel_a = kernel_scale_a.min().min()
vmax_kernel_a = kernel_scale_a.max().max()

fig, axes = plt.subplots(1, 2, figsize=(5, 5))  # Taller figure for better proportions

# Shared color maps for kernel distance and loss
cmap_kernel_na = 'coolwarm'
cmap_kernel_a = 'coolwarm'

# Plot kernel distance and loss heatmaps f
import matplotlib.pyplot as plt
import seaborn as sns

# Adjusted figure size for better readability
fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})

# Shared color range for both heatmaps
vmin = vmin_kernel_a
vmax = vmax_kernel_a

# Plot heatmap for not aligned (na)
sns.heatmap(kernel_heatmaps_a[step],
            ax=axes[0],
            xticklabels=True,
            yticklabels=True,
            cmap='coolwarm',  # Retain the original color map
            cbar=True,
            vmin=vmin, vmax=vmax,
            square=True)
axes[0].set_title('Kernel Distance (Not Aligned)', fontsize=14)
axes[0].set_xlabel('Initialisation Scale', fontsize=12)
axes[0].set_ylabel('Sequence Lengths', fontsize=12)
axes[0].tick_params(axis='both', which='major', labelsize=10)

# Plot heatmap for aligned (a)
sns.heatmap(kernel_heatmaps_a[step],
            ax=axes[1],
            xticklabels=True,
            yticklabels=True,
            cmap='coolwarm',  # Retain the original color map
            cbar=True,
            vmin=vmin, vmax=vmax,
            square=True)
axes[1].set_title('Kernel Distance (Aligned)', fontsize=14)
axes[1].set_xlabel('Initialisation Scale', fontsize=12)
axes[1].set_ylabel('Sequence Lengths', fontsize=12)
axes[1].tick_params(axis='both', which='major', labelsize=10)

# Adjust layout to improve spacing
plt.tight_layout(pad=3)
plt.show()

print('hello')
