import torch
import numpy as np
from types import SimpleNamespace
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import random
import os
import json
import matplotlib as mpl
from scipy.linalg import block_diag

def loss_function(y, y_hat):
    return (1/2)*torch.mean((y-y_hat)**2)

def compute_YXt_correlation(X, y, t):
    X_t = X[:,t-1]

    YX_t = np.einsum('pij,pjk->pik', y.reshape(y.shape[0], y.shape[2],1), X_t.reshape(X_t.shape[0], 1, -1))
    
    YX_t = np.sum(YX_t, axis=0)
    return YX_t

def generate_y_T_multiseed(args, data_tape, seed): 
    y = np.einsum('ij,tjk,kl,ptl->pi', data_tape.U[seed], args.S, data_tape.V[seed].T, data_tape.X[seed])
    y = y.reshape(y.shape[0], 1, -1)
    return y

def generate_y_T_eigenspace_multiseed(args, data_tape, seed): 
    y = np.einsum('ij,tjk,kl,ptl->pi', data_tape.P[seed], args.D, data_tape.P_inv[seed], data_tape.X[seed])
    y = np.real_if_close(y.reshape(y.shape[0], 1, -1))
    return y

def initialize_LR_SVec_multiseed(args, data_tape, seed):
    teacher = np.random.randn(args.output_size, args.input_size)
    U, _, V = np.linalg.svd(teacher)
    data_tape.U[seed] = U
    data_tape.V[seed] = V
    return data_tape

def block_diagonal_matrix(eigenvalues, tol=1e-12):
    ev_list = list(eigenvalues)  
    blocks = []
    evecs = {}  

    total_dim = 0
    
    def block2x2(alpha, beta):
        return np.array([[alpha, -beta],
                         [beta,   alpha]], dtype=float)

    i = 0
    used = [False]*len(ev_list)
    
    for i in range(len(ev_list)):
        if used[i]:
            continue
        lam = ev_list[i]

        if abs(lam.imag) < tol:  
            lam_real = lam.real
            # 1x1 block
            blocks.append(np.array([[lam_real]], dtype=float))
            evecs.setdefault(lam_real, np.array([1.0]))  
            
            used[i] = True
            total_dim += 1
            
        else:
            alpha, beta = lam.real, lam.imag
            
            conj_val = alpha - 1j*beta
            found_conj = False
            for j in range(i+1, len(ev_list)):
                if not used[j] and abs(ev_list[j] - conj_val) < tol:
                    used[j] = True
                    found_conj = True
                    break
            
            if not found_conj:
                raise ValueError(f"Eigenvalue {lam} does not have a matching conjugate in the list.")
            
            # 2x2 real block
            B = block2x2(alpha, beta)
            blocks.append(B)
            total_dim += 2
            used[i] = True

            lam_plus  = alpha + 1j*beta
            lam_minus = alpha - 1j*beta

            evecs.setdefault(lam_plus,  np.array([1.0, 1.0j]))  
            evecs.setdefault(lam_minus, np.array([1.0,-1.0j]))

    A = block_diag(*blocks)
    return A, evecs

def initialize_eigenvectors_multiseed(args, data_tape, seed):
    assert args.output_size == args.input_size
    teacher, _ = block_diagonal_matrix(np.diag(args.D[0])) 
    data_tape.P[seed] = np.real_if_close(np.linalg.eig(teacher)[1])
    data_tape.P_inv[seed] = np.real_if_close(np.conj(np.linalg.eig(teacher)[1]).T) # using conjugate transpose instead of inverse
    return data_tape

def whiten(X): 
    scaler = StandardScaler()
    X_standardised = scaler.fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_standardised)
    X_whitened = X_pca / np.sqrt(pca.explained_variance_)
    X_whitened = np.sqrt(X.shape[0] / (X.shape[0] - 1)) * X_whitened
    assert X.shape == X_whitened.shape
    return X_whitened / np.sqrt(X.shape[0])

def generate_X(args):
    X = np.random.normal(0, 0.01, size=(args.num_samples, args.trajectory_length*args.input_size)).astype(float) 
    X = whiten(X).reshape(args.num_samples, args.trajectory_length, args.input_size)
    return X

def generate_eigenspace_data_multiseed(args, data_tape, seed): ## TODO: double check that constructing tasks this way makes sense
    data_tape = initialize_eigenvectors_multiseed(args, data_tape, seed)
    data_tape.X[seed] = generate_X(args)
    data_tape.y[seed] = generate_y_T_eigenspace_multiseed(args, data_tape, seed)
    return data_tape

def generate_data_multiseed(args, data_tape, seed):
    data_tape = initialize_LR_SVec_multiseed(args, data_tape, seed)
    data_tape.X[seed] = generate_X(args)
    data_tape.y[seed] = generate_y_T_multiseed(args, data_tape, seed)
    return data_tape

def rotation_matrix(num_dim, theta):
    """Returns a rotation matrix for angle theta (in radians)."""
    if num_dim == 2:
        return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    elif num_dim == 3:
        return np.array([[np.cos(theta)*np.cos(theta), np.cos(theta)*np.sin(theta)*np.sin(theta) - np.sin(theta)*np.cos(theta), np.cos(theta)*np.sin(theta)*np.cos(theta) + np.sin(theta)*np.sin(theta)],
                         [np.sin(theta)*np.cos(theta), np.sin(theta)*np.sin(theta)*np.sin(theta) + np.cos(theta)*np.cos(theta), np.sin(theta)*np.sin(theta)*np.cos(theta) - np.cos(theta)*np.sin(theta)],
                         [-np.sin(theta), np.cos(theta)*np.sin(theta), np.cos(theta)*np.cos(theta)]
                         ])

def generate_y_T_teacher(args, data_tape, seed): ### is it worth just using hidden layer and ignoring Wx,Wy ??
    if args.task_dynamics == 'random_teacher': 
        tW_x = np.random.randn(args.hidden_size, args.input_size)
        tW_h = np.random.randn(args.hidden_size, args.hidden_size)
        tW_y = np.random.randn(args.output_size, args.hidden_size)
    elif args.task_dynamics == 'rotation':
        tW_x = np.eye(args.num_SV) ## ignore rotation of Wx,Wy?
        tW_h = rotation_matrix(args.num_SV, args.theta) 
        data_tape.P[seed] = np.real_if_close(np.linalg.eig(tW_h)[1])
        data_tape.P_inv[seed] = np.real_if_close(np.conj(np.linalg.eig(tW_h)[1]).T) # using conjugate transpose instead of inverse
        # data_tape.P_inv[seed] = np.real_if_close(np.linalg.inv(np.linalg.eig(tW_h)[1]))
        data_tape.D[seed] = np.real_if_close(np.linalg.eig(tW_h)[0])
        tW_y = np.eye(args.num_SV)
    
    y = np.zeros((args.num_samples, 1, args.output_size))
    h_t = np.zeros((args.num_samples, args.hidden_size))
    for t in range(args.trajectory_length):
        h_t = np.real_if_close(np.einsum("ij,bj->bi", tW_h, h_t) + np.einsum("ij,bj->bi", tW_x, data_tape.X[seed,:,t]))
        y_t = np.real_if_close(np.einsum("ij,bj->bi", tW_y, h_t))
    data_tape.y[seed] = y_t.reshape(args.num_samples, 1, -1)
    return data_tape

def generate_teacher_data(args, data_tape, seed): 
    data_tape.X[seed] = generate_X(args)
    data_tape = generate_y_T_teacher(args, data_tape, seed)
    return data_tape

def initialize_student_eigenspace_multiseed(args, data_tape, seed, tape):
    dW_x = np.zeros((args.hidden_size, args.input_size))
    if args.zero_balanced:
        dW_x = np.diag(np.sqrt(args.inout_mode_init))
    else:
        dW_x = np.diag(args.in_mode_init)
    W_x = np.real_if_close(data_tape.P[seed] @ dW_x @ data_tape.P_inv[seed])

    dW_y = np.zeros((args.output_size, args.hidden_size))
    if args.zero_balanced:
        dW_y = np.diag(np.sqrt(args.inout_mode_init))
    else:
        dW_y = np.diag(args.out_mode_init)
    W_y = np.real_if_close(data_tape.P[seed] @ dW_y @ data_tape.P_inv[seed])

    dW_h = np.zeros((args.hidden_size, args.hidden_size))
    dW_h = np.diag(args.rec_mode_init)
    W_h = np.real_if_close(data_tape.P[seed] @ dW_h @ data_tape.P_inv[seed])

    tape.W_x[seed][0] = np.real_if_close(W_x) 
    tape.W_h[seed][0] = np.real_if_close(W_h) 
    tape.W_y[seed][0] = np.real_if_close(W_y)
    tape.dW_x[seed][0] = dW_x
    tape.dW_h[seed][0] = dW_h
    tape.dW_y[seed][0] = dW_y
    return tape

def initialize_student_multiseed(args, data_tape, seed, tape):
    r = np.random.randn(args.hidden_size, args.hidden_size) # SHOULD BE GAUSSIAN NOT UNIFORM
    u,s,v = np.linalg.svd(r)
    R_y = v.T
    R_x = np.linalg.inv(v)

    dW_x = np.zeros((args.hidden_size, args.input_size))
    if args.zero_balanced:
        dW_x = np.diag(np.sqrt(args.inout_mode_init))
    else:
        dW_x = np.diag(args.in_mode_init)
    W_x = R_x @ dW_x @ data_tape.V[seed].T

    dW_y = np.zeros((args.output_size, args.hidden_size))
    if args.zero_balanced:
        dW_y = np.diag(np.sqrt(args.inout_mode_init))
    else:
        dW_y = np.diag(args.out_mode_init)
    W_y = data_tape.U[seed] @ dW_y @ R_y.T

    dW_h = np.zeros((args.hidden_size, args.hidden_size))
    dW_h = np.diag(args.rec_mode_init)
    W_h = R_y @ dW_h @ R_x.T

    tape.W_x[seed][0] = W_x 
    tape.W_h[seed][0] = W_h 
    tape.W_y[seed][0] = W_y
    tape.dW_x[seed][0] = dW_x
    tape.dW_h[seed][0] = dW_h
    tape.dW_y[seed][0] = dW_y
    tape.R_y[seed][0] = R_y
    tape.R_x[seed][0] = R_x
    return tape

def initialize_unaligned_student(args, seed, tape): # TODO: check scaling
    if args.initialization_scale == None: # TODO: bad practice-- make it's own argument
        tape.W_x[seed][0] = np.random.normal(loc=0.0, scale = np.sqrt(args.inout_mode_init[0])/(args.input_size**0.5), size=(args.hidden_size, args.input_size))
        tape.W_h[seed][0] = np.random.normal(loc=0.0, scale = args.rec_mode_init[0]/(args.hidden_size**0.5), size=(args.hidden_size, args.hidden_size))
        tape.W_y[seed][0] = np.random.normal(loc=0.0, scale = np.sqrt(args.inout_mode_init[0])/(args.hidden_size**0.5), size=(args.output_size, args.hidden_size))
    else:
        tape.W_x[seed][0] = np.random.normal(loc=0.0, scale = args.initialization_scale/(args.input_size**0.5), size=(args.hidden_size, args.input_size))
        tape.W_h[seed][0] = np.random.normal(loc=0.0, scale = args.initialization_scale/(args.hidden_size**0.5), size=(args.hidden_size, args.hidden_size))
        tape.W_y[seed][0] = np.random.normal(loc=0.0, scale = args.initialization_scale/(args.hidden_size**0.5), size=(args.output_size, args.hidden_size))
    return tape

def get_tape_module(args):
    tape_elements = dict(
        loss = (),
        W_x = (args.hidden_size, args.input_size),
        W_h = (args.hidden_size, args.hidden_size),
        W_y = (args.output_size, args.hidden_size),
        W_x_grad = (args.hidden_size, args.input_size),
        W_h_grad = (args.hidden_size, args.hidden_size),
        W_y_grad = (args.output_size, args.hidden_size),
        input_modes = (args.num_SV,),
        recurrent_modes = (args.num_SV,),
        output_modes = (args.num_SV,),
        input_output_modes = (args.num_SV,),
    )
    if args.aligned_student_weights:
        tape_elements['dW_x'] = (args.hidden_size, args.input_size)
        tape_elements['dW_h'] = (args.hidden_size, args.hidden_size)
        tape_elements['dW_y'] = (args.output_size, args.hidden_size)
        tape_elements['R_x'] = (args.hidden_size, args.hidden_size)
        tape_elements['R_y'] = (args.hidden_size, args.hidden_size)
    if args.log_H:
        if args.subsample_H:
            tape_elements['H'] = (200, 1, args.hidden_size)
            tape_elements['H_grad'] = (200, 1, args.hidden_size)
        else:
            tape_elements['H'] = (args.num_samples, args.trajectory_length, args.hidden_size)
            tape_elements['H_grad'] = (args.num_samples, args.trajectory_length, args.hidden_size)
    if args.log_y_hat: 
        tape_elements['y_hat'] = (args.num_samples, 1, args.output_size)
        tape_elements['y_hat_grad'] = (args.num_samples, 1, args.output_size)
    if args.compute_NTK:
        num_samples = args.num_samples
        if args.subsample_NTK:
            num_samples = 50
        tape_elements['H_NTK'] = (num_samples*args.output_size, num_samples*args.output_size) 
        tape_elements['H_Tp1_grad'] = (num_samples, args.hidden_size) 
        tape_elements['NTK'] = (num_samples*args.output_size, num_samples*args.output_size)
        tape_elements['pred_y_grad'] = (num_samples, 1, args.output_size)
    tape_module = SimpleNamespace(**{k: np.zeros((args.num_seeds, int(args.num_epochs/args.log_every_e_epoch)) + v,dtype=complex) for k,v in tape_elements.items()})
    tape_module.NTK_distance = np.zeros((args.num_seeds))
    return tape_module

def get_data_tape(args):
    tape_elements = dict(
        U = (args.output_size, args.output_size),
        V = (args.input_size, args.input_size), 
        X = (args.num_samples, args.trajectory_length, args.input_size),
        P = (args.output_size, args.input_size),
        P_inv = (args.output_size, args.input_size),
        D = (args.output_size,), # find smart way to unify with the other D in args
    )
    tape_elements['y'] = (args.num_samples, 1, args.output_size)
    tape_module = SimpleNamespace(**{k: np.zeros((args.num_seeds,) + v,dtype=complex) for k,v in tape_elements.items()})
    return tape_module

def matrix_power(matrix, exponent):
    if exponent == 0: 
        return identity_matrix(matrix.shape)
    product = matrix 
    for i in range(exponent - 1):
        product = product @ matrix 
    return product 

def identity_matrix(shape): 
    matrix = np.zeros((shape))
    for i in range(matrix.shape[0]):
        matrix[i,i] = 1
    return matrix

def kernel_distance(A, B):
    K_t1 = A
    K_t2 = B
    inner_product = np.sum(K_t1 * K_t2)
    norm_t1 = np.linalg.norm(K_t1, 'fro')
    norm_t2 = np.linalg.norm(K_t2, 'fro')
    S_t1_t2 = 1 - inner_product / (norm_t1 * norm_t2)
    return S_t1_t2

def encode_json(data_dict):
    json_data = dict()
    for key, value in data_dict.__dict__.items():
        if type(value).__module__ == np.__name__:
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            else:
                json_data[key] = value.item()
        else:
            json_data[key] = value
    return json_data

def decode_json(json_data):
    data_dict = dict()
    for key, value in json_data.items():
        if isinstance(value, list) and (len(np.array(value).shape) > 1):
            data_dict[key] = np.array(value)
        else:
            data_dict[key] = value
    return SimpleNamespace(**data_dict)

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_SV_search_tape(args):
    tape_elements = dict(
        final_loss = np.zeros((len(args.lambda_0_range), len(args.S_T_range), args.num_seeds)),
        input_output_modes = np.zeros((len(args.lambda_0_range), len(args.S_T_range), args.num_seeds, args.num_SV)),
        recurrent_modes = np.zeros((len(args.lambda_0_range), len(args.S_T_range), args.num_seeds, args.num_SV)),
        input_output_modes_mean = np.zeros((len(args.lambda_0_range), len(args.S_T_range))), 
        recurrent_modes_mean = np.zeros((len(args.lambda_0_range), len(args.S_T_range))),
        input_output_modes_sem = np.zeros((len(args.lambda_0_range), len(args.S_T_range))), 
        recurrent_modes_sem = np.zeros((len(args.lambda_0_range), len(args.S_T_range))),
        dd_term = np.zeros((len(args.lambda_0_range), len(args.S_T_range))),
        eff_reg_term = np.zeros((len(args.lambda_0_range), len(args.S_T_range))),
    )
    tape_module = SimpleNamespace(**tape_elements)
    return tape_module

def get_rich_lazy_search_tape(args, param1_len, param2_len, param3_len):                                                                           
    if param3_len == None:
        tape_elements = dict(
            final_loss = np.zeros((param1_len, param2_len, args.num_seeds)),
            input_output_modes = np.zeros((param1_len, param2_len, args.num_seeds, args.num_SV)),
            recurrent_modes = np.zeros((param1_len, param2_len, args.num_seeds, args.num_SV)),
            input_output_modes_mean = np.zeros((param1_len, param2_len)), 
            recurrent_modes_mean = np.zeros((param1_len, param2_len)),
            input_output_modes_sem = np.zeros((param1_len, param2_len)), 
            recurrent_modes_sem = np.zeros((param1_len, param2_len)),
            NTK_distance = np.zeros((param1_len, param2_len, args.num_seeds)),
            NTK_distance_mean = np.zeros((param1_len, param2_len)),
            NTK_distance_sem = np.zeros((param1_len, param2_len))
        )
    else:
        tape_elements = dict(
            final_loss = np.zeros((param1_len, param2_len, param3_len, args.num_seeds)),
            input_output_modes = np.zeros((param1_len, param2_len, param3_len, args.num_seeds, args.num_SV)),
            recurrent_modes = np.zeros((param1_len, param2_len, param3_len, args.num_seeds, args.num_SV)),
            input_output_modes_mean = np.zeros((param1_len, param2_len, param3_len)), 
            recurrent_modes_mean = np.zeros((param1_len, param2_len, param3_len)),
            input_output_modes_sem = np.zeros((param1_len, param2_len, param3_len)), 
            recurrent_modes_sem = np.zeros((param1_len, param2_len, param3_len)),
            NTK_distance = np.zeros((param1_len, param2_len, param3_len, args.num_seeds)),
            NTK_distance_mean = np.zeros((param1_len, param2_len, param3_len)),
            NTK_distance_sem = np.zeros((param1_len, param2_len, param3_len))
        )
    tape_module = SimpleNamespace(**tape_elements)
    return tape_module

def _hash(obj):
    json_obj = encode_json(obj)
    return hash(json.dumps(json_obj, sort_keys=True))

def compute_data_driven_term(args, S, input_output_mode, recurrent_mode):
    tau = (args.num_samples*args.output_size)/args.lr
    dd_term = 0
    for i in range(1, args.trajectory_length+1):
        dd_term += S[i-1]**2 - 2*input_output_mode*(recurrent_mode**(args.trajectory_length-i))*S[i-1]
    return dd_term/(2*tau)

def compute_effective_regularization_term(args, input_output_mode, recurrent_mode):
    tau = (args.num_samples*args.output_size)/args.lr
    eff_reg_term = (1/(2*tau))*(input_output_mode**2)*((1-(recurrent_mode**(2*args.trajectory_length)))/(1-(recurrent_mode**2)))
    return eff_reg_term

class BlindColours:
    def __init__(self, reverse_cmap=True):

        # Define a set of colors that are accessible to individuals with color blindness
        hex_colours = ["#d65c00", "#0071b2", "#009e73", "#cc78a6", "#e59c00", "#55b2e8", "#efe440", "#000000"]
        self.blind_colours = [mpl.colors.to_rgb(h) for h in hex_colours]

        # Define a diverging colormap
        div = ['#6d0000', '#720400', '#770900', '#7c0d00', '#821200', '#871600', '#8b1b00', '#901f00', '#952300',
               '#9a2700', '#9f2c00', '#a33000', '#a83400', '#ad3800', '#b13c00', '#b64000', '#bb4500', '#bf4900',
               '#c44d00', '#c85100', '#cc5604', '#cf5b09', '#d3600e', '#d66513', '#d96a18', '#dd6f1d', '#e07422',
               '#e37927', '#e67e2c', '#ea8331', '#ed8836', '#f08d3b', '#f3923f', '#f69744', '#f99b49', '#fda04e',
               '#ffa555', '#feac62', '#fdb26e', '#fdb87a', '#fcbe87', '#fbc492', '#faca9e', '#f9d5b4', '#f8dabf',
               '#f8e0ca', '#f7e5d5', '#f6ebe0', '#f6f0ea', '#ecf2f6', '#e3eef7', '#d9ebf8', '#d0e7f8', '#c6e4f9',
               '#bde0fa', '#b3ddfb', '#a9d9fc', '#9fd6fd', '#95d2fe', '#8bceff', '#85cafc', '#80c6f9', '#7bc2f6',
               '#75bef2', '#70baef', '#6bb6ec', '#66b1e9', '#61ade5', '#5ba9e2', '#56a5df', '#51a1dc', '#4c9dd8',
               '#4799d5', '#4295d2', '#3c91cf', '#378dcb', '#3289c8', '#2d85c5', '#2881c2', '#237dbf', '#2079ba',
               '#1e75b6', '#1c71b1', '#1a6dad', '#1969a8', '#1765a4', '#15619f', '#135d9b', '#115996', '#105592',
               '#0e518e', '#0c4d89', '#0a4a85', '#094681', '#07427d', '#053e79', '#033b74', '#023770', '#00346c']
        if reverse_cmap:
            div.reverse()  # Reverse the colormap if specified
        self.div_cmap = mpl.colors.ListedColormap(div)

        # Define color steps for specific color ranges
        oranges = [mpl.colors.to_rgb(h) for h in ['#871500', '#a93700', '#cc5400', '#ef721c', '#ff9c4a']]
        blues = [mpl.colors.to_rgb(h) for h in ['#00356e', '#005492', '#0975b7', '#4895d9', '#70b6fd']]
        greens = [mpl.colors.to_rgb(h) for h in ['#003e1d', '#005e39', '#008057', '#09a378', '#46c698']]
        self.colour_steps = [oranges, blues, greens]

    def get_colours(self):
        return self.blind_colours

    def get_div_cmap(self):
        return self.div_cmap

    def get_colour_steps(self):
        return self.colour_steps