from src import SV_configs
from src import utils
from src import linear_RNN
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import svd

def rotation_matrix(theta):
    """Returns a 2D rotation matrix for angle theta (in radians)."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def kernel_distance(A,B):
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

def initialize_unaligned_student(args, seed, tape): # TODO: check scaling
    tape.W_x[seed][0] = np.random.normal(loc=0.0, scale = args.initialization_scale/(args.input_size**0.5), size=(args.hidden_size, args.input_size))
    tape.W_h[seed][0] = np.random.normal(loc=0.0, scale =  args.initialization_scale/(args.hidden_size**0.5), size=(args.hidden_size, args.hidden_size))
    tape.W_y[seed][0] = np.random.normal(loc=0.0, scale = args.initialization_scale/(args.hidden_size**0.5), size=(args.output_size, args.hidden_size))
    return tape


lenght = [20]
scale = [1]
# TO: for differnet time steps
NTK_a = {}
losses = {}
kernel_distances = {}
for s in scale:
    for l in lenght:
        traj_1_args = {
            'data_out_dir': 'data/',
            'results_out_dir': 'results/',
            'name': 'unaligned',
            'num_seeds': 1,
            # to run over more seeds which need to be specified as arguments (see simulate.py for example)
            'override_load': False,
            'trajectory_length': l,
            'num_samples': 20,
            'num_epochs': 300,
            'lr': 0.1,
            'initialization_scale': 0.01,
            # (variance of the 0-mean initialized weights when the network is unaligned)
            'input_size': 2,
            'hidden_size': 2,
            'output_size': 2,
            'S_config': ['exponential', 'exponential', 'exponential', 'exponential'],
            # data dynamics: constant (lambda_t=1-- overrides the lambda_0 argument), exponential (lambda_t=lambda^t), inverse-exponential (lambda_t=lambda^{T-t})
            'delta': [1, 1],  # data dynamics scaling-term (i.e., s_t = delta*lambda_t) for each SV dimension
            'lambda_0': [1, 1],  # data dynamics temporal-term
            'S_T': [1, 1],  # SV at last timestep (feedforward)
            'S_TeqS_t': True,
            # whether SV at last timestep follows the data dynamics (if true, ignores S_T argument)
            'zero_balanced': True,  # if true, sets a=c at initialization (for aligned weights)
            'frozen_hidden_layer': False,  # freezes learning on recurrent layer
            'frozen_input_output': False,  # freezes learning on input-output layers
            'single_output_y': False,
            # only computes loss on y_T (ignores y_1 to y_{T-1}); code not compatible with the multi-output case at the moment
            'inout_mode_init': [s, s],
            # input-output mode initialization for each dimension (when weights are aligned and input-output are zero-balanced)
            'in_mode_init': [0.7, 0.7],
            # input mode initialization for each dimension (when input-output weights are not zero-balanced)
            'out_mode_init': [0.7, 0.7],  # output ""
            'rec_mode_init': [s, s],  # recurrent ""
            'num_SV': 2 , # num of SV dimensions (should be equal to input=hidden=output sizes)
            'aligned_student_weights': True,
            # if False, random weights ; if True, aligned to SV data dynamics specified above
            'log_H': False,
            'log_y_hat': False,
            'compute_NTK': True,
            'nonlinear_RNN': False,
            'log_every_e_epoch': 100,  # how many gradient steps until logging metrics
            'subsample_NTK': False,  # whether to use a subsample of data when computing NTK
            'single_output_rotation': True
        }
        traj_1_args = SimpleNamespace(**traj_1_args)
        traj_1_tape = utils.get_tape_module(traj_1_args)
        traj_1_data_tape = utils.get_data_tape(traj_1_args)

        seed = 0
        utils.set_seeds(seed)

        ## --- generating tasks ----
        traj_1_args.S = SV_configs.create_SV_trajectories(traj_1_args)
        traj_1_data_tape = utils.generate_data_multiseed(traj_1_args, traj_1_data_tape, seed)
        #traj_1_data_tape = utils.generate_rotation_data_multiseed(traj_1_args, traj_1_data_tape, seed)

        ## example 2: generating data from a random unaligned teacher (or could be anything else)
        # traj_1_X, traj_1_y, traj_1_t_W_x, traj_1_t_W_h, traj_1_t_W_y = generate_unaligned_teacher_data(traj_1_args, variance=0.1)
        # traj_1_data_tape.X[seed] = traj_1_X
        # traj_1_data_tape.y[seed] = traj_1_y

        ## --- generating LRNN ----
        ## example 1: aligned RNN (NOTE: not set up to align to any given task; has to be generated from SV data dynamics specified in arguments)
        ## so don't combine aligned RNN + unaligned teacher unless new implementation
        #traj_1_tape = utils.initialize_student_multiseed(traj_1_args, traj_1_data_tape, seed, traj_1_tape)

        # example 2: unaligned RNN
        traj_1_tape = utils.initialize_unaligned_student(traj_1_args, seed, traj_1_tape)

        # --- model initialization, training, modes, NTK ---
        traj_1_model = linear_RNN.LinearRNN_multiseed(traj_1_args, traj_1_tape, seed)
        traj_1_model.train(traj_1_data_tape.X[seed], traj_1_data_tape.y[seed])
        #        traj_1_model.project_SVD_space(
        #        traj_1_data_tape)  # computes modes for aligned network; computes SV for unaligned network
        #        traj_1_model.compute_NTK(traj_1_data_tape.X[seed], traj_1_data_tape.y[seed], seed)
        traj_1_tape = traj_1_model.tape
        #ntks_empirical[(lmda, scale)] = np.array(
        #  [get_ntk(w1w1, w2w2, X, out_dim) for (w1w1, w2w2) in zip(w1w1s_empirical[(lmda, scale)], w2w2s_empirical[(lmda, scale)])])
        # reshaped_NTK_3D_traj_1 = traj_1_tape.NTK[seed].reshape(
        #    int(traj_1_args.num_epochs / traj_1_args.log_every_e_epoch),
        #    traj_1_args.num_samples * traj_1_args.output_size,
        #    traj_1_args.num_samples * traj_1_args.output_size
        # )
        # kernel_distances[(l, s)] = kernel_distance(reshaped_NTK_3D_traj_1[0, :, :],
        # reshaped_NTK_3D_traj_1[-1, :, :])
        #  NTK_a[(l, s)] = reshaped_NTK_3D_traj_1
        losses[(l, s)] = traj_1_tape.loss[seed]

# Is the Y correct
# X should be fixed 8 not changing) Idendity matrix
#Need to plot this Y
traj_1_data_tape.y
traj_1_data_tape.X
X = traj_1_data_tape.X[0,1,:,:]
Y = traj_1_data_tape.y[0,4,:,:]

fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.grid(True)

# Plot vectors
vector_lines = [ax.plot([0, Y[i, 0]], [0, Y[i, 1]], 'ro-', lw=2)[0] for i in range(20)]  # Red vectors
circle = plt.Circle((0, 0), 1, color='b', fill=False)  # Unit circle
ax.add_patch(circle)
plt.show()


# Compute the angles (in radians) of the vectors with respect to the x-axis
angles = np.arctan2(Y[:, 1], Y[:, 0])

# Compute the sine and cosine of the angles
sine_values = np.sin(angles)
cosine_values = np.cos(angles)

# Create the plot for sine and cosine values
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot the sine values
ax[0].plot(range(20), sine_values, 'ro-', lw=2, label='sin(θ)')
ax[0].set_title('Sine of Angles')
ax[0].set_xlabel('Vector Index')
ax[0].set_ylabel('sin(θ)')
ax[0].grid(True)
ax[0].legend()

# Plot the cosine values
ax[1].plot(range(20), cosine_values, 'bo-', lw=2, label='cos(θ)')
ax[1].set_title('Cosine of Angles')
ax[1].set_xlabel('Vector Index')
ax[1].set_ylabel('cos(θ)')
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.show()
bc = utils.BlindColours()
blind_colors = bc.get_colours()
div_cmap = bc.get_div_cmap()

fig = plt.figure(figsize=(12, 10), layout='constrained')
mosaic = """
a
d
g
j
"""

axd = fig.subplot_mosaic(mosaic)
for ax in axd.values():
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_prop_cycle('color', blind_colors[:-1])

axd['a'].plot(traj_1_tape.loss[seed], color=blind_colors[0])
axd['a'].set_ylabel(r'loss ($\mathcal{L}$)')
axd['a'].set_title(r'trajectory length $T=1$')


axd['d'].plot(traj_1_tape.NTK[seed].reshape(int(traj_1_args.num_epochs / traj_1_args.log_every_e_epoch),
                                            traj_1_args.num_samples * traj_1_args.output_size * traj_1_args.num_samples * traj_1_args.output_size)[
              :, :200])
axd['d'].set_ylabel('NTK')
axd['g'].plot(traj_1_tape.recurrent_modes[seed])
axd['g'].set_ylabel(r'recurrent singular values')
# axd['h'].plot([1.0]*300, color='gray', alpha=0.6, label='global solution')


# axd['j'].plot([1.0]*300, color='gray', alpha=0.6, label='global solution')
axd['j'].plot(traj_1_tape.input_output_modes[seed])
axd['j'].set_ylabel(r'input-output singular values')
axd['j'].set_xlabel(r'training steps ($t_\theta$)')
# axd['k'].plot([1.0]*300, color='gray', alpha=0.6, label='global solution')
fig.tight_layout()



import numpy as np

# Define matrices and weights
N, D, P = 5, 3, 4  # Example dimensions
X = np.random.randn(N, P)  # Input matrix
W_out = np.random.randn(N, D)  # Weight matrix

# Compute Z
Z = W_out.T @ X  # Equation (5)

# Compute Frobenius norm
frobenius_norm = lambda A: np.sqrt(np.sum(A**2))

# Compute generalized correlation (Equation 6)
rho = frobenius_norm(W_out.T @ X) / (frobenius_norm(W_out) * frobenius_norm(X))

# Compute the norm of Z (Equation 7)
Z_norm = rho * frobenius_norm(W_out) * frobenius_norm(X)

# Print results
print("Z matrix:\n", Z)
print("Generalized correlation (ρ):", rho)
print("Norm of Z:", Z_norm)

# Little update on the things I have done :
# Created the task with a vector y = 2 by 1 that rotates over time. The input is fixed, but changes over each sample.
# I ploted the task it seems to be doing what it should See the plots bellow.
# I also wrote the corrolation matrix to measure the alignment,but the first thing I think is to check if the network can learn this task.
# ( I have share the code so you can see what i have done so far) I didn't do more since i took time to make the new figures for my figure (see atccahed ) I am going on vaction next week. HOpfully I will get back to this once I coem backk
# It seems like what I have is a plot of the sine and cosine values of the angles of the vectors with respect to the x-axis.
#  And plot of the Y
# X is fixed for now, for simplicity.
#Check the multiplicaiton
#I want to plot X as well to see if it is changing over time