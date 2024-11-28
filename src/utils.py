import torch
import numpy as np
import copy
from types import SimpleNamespace
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import random
import os
from pathlib import Path
import json

def loss_function(y, y_hat):
    return (1/2)*torch.mean((y-y_hat)**2)

# For single-output y
def compute_YXt_correlation(X, y, t):
    X_t = X[:,t-1]
    # YX_t = 0
    # for p in range(1, y.shape[0]+1):
    #     YX_t += y[p-1].reshape(-1,1) @ X[p-1,t-1].reshape(1,-1)

    # parallelized
    # YX_t = torch.einsum('pij,pjk->pik', y.reshape(y.shape[0], y.shape[1],1), X_t.reshape(X_t.shape[0], 1, -1))
    # YX_t = torch.sum(YX_t, axis=0)
    YX_t = np.einsum('pij,pjk->pik', y.reshape(y.shape[0], y.shape[1],1), X_t.reshape(X_t.shape[0], 1, -1))
    YX_t = np.sum(YX_t, axis=0)
    return YX_t

# For multi-output y
def compute_YtXi_correlation(X, y, t, i):
    X_i = X[:,i-1]
    y_t = y[:,t-1]

    # YtXi = 0
    # for p in range(1, X.shape[0]+1):
    #     YtXi += y[p-1,t-1].reshape(1,-1) @ X[p-1,i-1].reshape(1,-1)

    # parallelized
    # YtXi = torch.einsum('pij,pjk->pik', y_t.reshape(y_t.shape[0], y_t.shape[1],1), X_i.reshape(X_i.shape[0], 1, -1))
    # YtXi = torch.sum(YtXi, axis=0)
    YtXi = np.einsum('pij,pjk->pik', y_t.reshape(y_t.shape[0], y_t.shape[1],1), X_i.reshape(X_i.shape[0], 1, -1))
    YtXi = np.sum(YtXi, axis=0)
    
    return YtXi

# For single-output y
def generate_y_T(args, X): 
    # y = torch.einsum('ij,tjk,kl,ptl->pi', args.U, args.S, args.V.T, X)
    y = np.einsum('ij,tjk,kl,ptl->pi', args.U, args.S, args.V.T, X)
    y = y.reshape(y.shape[0], 1, -1)
    return y

def generate_y_T_multiseed(args, data_tape, seed): 
    # y = torch.einsum('ij,tjk,kl,ptl->pi', data_tape.U[seed], args.S, data_tape.V[seed].T, data_tape.X[seed])
    y = np.einsum('ij,tjk,kl,ptl->pi', data_tape.U[seed], args.S, data_tape.V[seed].T, data_tape.X[seed])
    y = y.reshape(y.shape[0], 1, -1)
    return y

# For multi-output y
def generate_y(args, X): # TODO: parallelize
    # y = torch.zeros((args.num_samples, args.trajectory_length, args.output_size))
    y = np.zeros((args.num_samples, args.trajectory_length, args.output_size))
    for p in range(args.num_samples):
        for t_y in range(args.trajectory_length):
            y_pt = 0
            for t_x in range(args.trajectory_length):
                y_pt += args.U @ np.diag(args.S[t_y,t_x]) @ args.V.T @ X[p,t_x]
                # y_pt += args.U @ torch.diag(args.S[t_y,t_x]) @ args.V.T @ X[p,t_x]
            y[p,t_y] = y_pt
    # parallelized- check
    # y = torch.einsum('ij,Ttjk,kl,ptl->pTi', args.U, args.S, args.V.T, X)
    return y

def generate_y_multiseed(args, data_tape, seed): # TODO: parallelize
    y = torch.zeros((args.num_samples, args.trajectory_length, args.output_size))
    for p in range(args.num_samples):
        for t_y in range(args.trajectory_length):
            y_pt = 0
            for t_x in range(args.trajectory_length):
                # y_pt += data_tape.U[seed] @ torch.diag(args.S[t_y,t_x]) @ data_tape.V[seed].T @ data_tape.X[seed][p,t_x]
                y_pt += data_tape.U[seed] @ np.diag(args.S[t_y,t_x]) @ data_tape.V[seed].T @ data_tape.X[seed][p,t_x]
            y[p,t_y] = y_pt
    # parallelized- check
    # y = torch.einsum('ij,Ttjk,kl,ptl->pTi', args.U, args.S, args.V.T, X)
    return y

def check_x_uncorrelated_whitened(args, X):
    true_instances = 0
    total_instances = 0
    for t in range(1, args.trajectory_length+1):
        XtXt = 0
        X_t = X[:,t-1]
        # for p in range(1, args.num_samples+1):
        #     XtXt += X[p-1, t-1].reshape(-1,1) @ X[p-1,t-1].reshape(1,-1)

        # parallelized
        # XtXt = torch.einsum('pij,pjk->pik',X_t.reshape(X_t.shape[0], -1, 1), X_t.reshape(X_t.shape[0], 1, -1))
        # XtXt = torch.sum(XtXt, axis=0)
        XtXt = np.einsum('pij,pjk->pik',X_t.reshape(X_t.shape[0], -1, 1), X_t.reshape(X_t.shape[0], 1, -1))
        XtXt = np.sum(XtXt, axis=0)
        # print('Expected: I, Simulation: ', XtXt)
        # isclose = torch.isclose(torch.eye(X.shape[2]), XtXt, atol=1e-2)
        # true_instances += torch.count_nonzero(isclose)
        isclose = np.isclose(np.eye(X.shape[2]), XtXt, atol=1e-2)
        true_instances += np.count_nonzero(isclose)
        total_instances += isclose.flatten().shape[0]
        # print(isclose)

    for t in range(1,args.trajectory_length+1):
        for t_prime in range(1,args.trajectory_length+1):
            if t != t_prime:
                XtXtprime = 0
                for p in range(1,args.num_samples+1):
                    XtXtprime += X[p-1,t-1].reshape(-1,1) @ X[p-1, t_prime-1].reshape(1,-1)
                # print('Expected: 0, Simulation: ', XtXtprime)
                # isclose = torch.isclose(torch.zeros(X.shape[2], X.shape[2]), XtXtprime, atol=1e-2)
                # true_instances += torch.count_nonzero(isclose)
                isclose = np.isclose(np.zeros((X.shape[2], X.shape[2])), XtXtprime, atol=1e-2)
                true_instances += np.count_nonzero(isclose)
                total_instances += isclose.flatten().shape[0]
                # print(isclose)
    print(true_instances/total_instances)
    return true_instances/total_instances

def initialize_LR_SVec(args):
    # teacher = torch.rand((args.output_size, args.input_size), requires_grad = False) ## SHOULD BE A GAUSSIAN DISTRIBUTION NOT UNIFORM
    # U, _, V = torch.svd(teacher, some=False)
    teacher = np.random.randn(args.output_size, args.input_size)
    U, _, V = np.linalg.svd(teacher)
    args.U, args.V = U, V
    return args

def initialize_LR_SVec_multiseed(args, data_tape, seed):
    # teacher = torch.rand((args.output_size, args.input_size), requires_grad = False)
    # U, _, V = torch.svd(teacher, some=False)
    teacher = np.random.randn(args.output_size, args.input_size)
    U, _, V = np.linalg.svd(teacher)
    data_tape.U[seed] = U
    data_tape.V[seed] = V
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
    # X = torch.normal(0, 0.01, size=(args.num_samples, args.trajectory_length, args.input_size), dtype=torch.float32, requires_grad=False) * np.sqrt(10000/args.num_samples)
    X = np.random.normal(0, 0.01, size=(args.num_samples, args.trajectory_length*args.input_size)).astype(float) 
    X = whiten(X).reshape(args.num_samples, args.trajectory_length, args.input_size)
    return X

def generate_data(args):
    args = initialize_LR_SVec(args)
    X = generate_X(args)
    if args.single_output_y:
        y = generate_y_T(args, X)
    else:
        y = generate_y(args, X)
    return args, X, y

def generate_data_multiseed(args, data_tape, seed):
    data_tape = initialize_LR_SVec_multiseed(args, data_tape, seed)
    data_tape.X[seed] = generate_X(args)
    if args.single_output_y:
        data_tape.y[seed] = generate_y_T_multiseed(args, data_tape, seed)
    else:
        data_tape.y[seed] = generate_y_multiseed(args, data_tape, seed)
    return data_tape

def initialize_student(args):
    # r = torch.rand(args.hidden_size, args.hidden_size)
    # u,s,v = torch.svd(r, some=False)
    r = np.random.randn(args.hidden_size, args.hidden_size) ## GAUSSIAN NOT UNIFORM
    u,s,v = np.linalg.svd(r)
    R_y = v.T
    # R_x = torch.linalg.inv(v)
    R_x = np.linalg.inv(v)

    # dW_x = torch.zeros(args.hidden_size, args.input_size)
    dW_x = np.zeros((args.hidden_size, args.input_size))
    if args.zero_balanced:
        # torch.diag(torch.sqrt(args.inout_mode_init), 0, out=dW_x)
        dW_x = np.diag(np.sqrt(args.inout_mode_init))
    else:
        # torch.diag(args.in_mode_init, 0, out=dW_x)
        dW_x = np.diag(args.in_mode_init)
    W_x = R_x @ dW_x @ args.V.T
    # if not args.frozen_input_output:
    #     W_x.requires_grad_()

    # dW_y = torch.zeros(args.output_size, args.hidden_size)
    dW_y = np.zeros((args.output_size, args.hidden_size))
    if args.zero_balanced:
        # torch.diag(torch.sqrt(args.inout_mode_init), 0, out=dW_y)
        dW_y = np.diag(np.sqrt(args.inout_mode_init))
    else:
        # torch.diag(args.out_mode_init, 0, out=dW_y)
        dW_y = np.diag(args.out_mode_init)
    W_y = args.U @ dW_y @ R_y.T
    # if not args.frozen_input_output:
    #     W_y.requires_grad_()

    # dW_h = torch.zeros(args.hidden_size, args.hidden_size)
    # torch.diag(args.rec_mode_init, 0, out=dW_h)
    dW_h = np.zeros((args.hidden_size, args.hidden_size))
    dW_h = np.diag(args.rec_mode_init)
    W_h = R_y @ dW_h @ R_x.T
    # if not args.frozen_hidden_layer:
    #     W_h.requires_grad_()

    return W_x, W_h, W_y, dW_x, dW_h, dW_y, R_y, R_x

def initialize_student_multiseed(args, data_tape, seed, tape):
    # r = torch.rand(args.hidden_size, args.hidden_size)
    # u,s,v = torch.svd(r, some=False)
    r = np.random.randn(args.hidden_size, args.hidden_size) # SHOULD BE GAUSSIAN NOT UNIFORM
    u,s,v = np.linalg.svd(r)
    R_y = v.T
    # R_x = torch.linalg.inv(v)
    R_x = np.linalg.inv(v)

    # dW_x = torch.zeros(args.hidden_size, args.input_size)
    dW_x = np.zeros((args.hidden_size, args.input_size))
    if args.zero_balanced:
        # torch.diag(torch.sqrt(args.inout_mode_init), 0, out=dW_x)
        dW_x = np.diag(np.sqrt(args.inout_mode_init))
    else:
        # torch.diag(args.in_mode_init, 0, out=dW_x)
        dW_x = np.diag(args.in_mode_init)
    W_x = R_x @ dW_x @ data_tape.V[seed].T
    # if not args.frozen_input_output:
    #     W_x.requires_grad_()

    # dW_y = torch.zeros(args.output_size, args.hidden_size)
    dW_y = np.zeros((args.output_size, args.hidden_size))
    if args.zero_balanced:
        # torch.diag(torch.sqrt(args.inout_mode_init), 0, out=dW_y)
        dW_y = np.diag(np.sqrt(args.inout_mode_init))
    else:
        # torch.diag(args.out_mode_init, 0, out=dW_y)
        dW_y = np.diag(args.out_mode_init)
    W_y = data_tape.U[seed] @ dW_y @ R_y.T
    # if not args.frozen_input_output:
    #     W_y.requires_grad_()

    # dW_h = torch.zeros(args.hidden_size, args.hidden_size)
    # torch.diag(args.rec_mode_init, 0, out=dW_h)
    dW_h = np.zeros((args.hidden_size, args.hidden_size))
    dW_h = np.diag(args.rec_mode_init)
    W_h = R_y @ dW_h @ R_x.T
    # if not args.frozen_hidden_layer:
    #     W_h.requires_grad_()

    tape.W_x[seed][0] = W_x 
    tape.W_h[seed][0] = W_h 
    tape.W_y[seed][0] = W_y
    tape.dW_x[seed][0] = dW_x
    tape.dW_h[seed][0] = dW_h
    tape.dW_y[seed][0] = dW_y
    tape.R_y[seed][0] = R_y
    tape.R_x[seed][0] = R_x
    return tape

def get_tape_module(args):
    tape_elements = dict(
        loss = (),
        W_x = (args.hidden_size, args.input_size),
        W_h = (args.hidden_size, args.hidden_size),
        W_y = (args.output_size, args.hidden_size),
        dW_x = (args.hidden_size, args.input_size),
        dW_h = (args.hidden_size, args.hidden_size),
        dW_y = (args.output_size, args.hidden_size),
        R_x = (args.hidden_size, args.hidden_size),
        R_y = (args.hidden_size, args.hidden_size),
        H = (args.num_samples, args.trajectory_length, args.hidden_size),
        W_x_grad = (args.hidden_size, args.input_size),
        W_h_grad = (args.hidden_size, args.hidden_size),
        W_y_grad = (args.output_size, args.hidden_size),
        H_grad = (args.num_samples, args.trajectory_length, args.hidden_size),
        input_modes = (args.num_SV,),
        recurrent_modes = (args.num_SV,),
        output_modes = (args.num_SV,),
        input_output_modes = (args.num_SV,)
    )
    tape_module = SimpleNamespace(**{k: np.zeros((args.num_seeds, args.num_epochs) + v) for k,v in tape_elements.items()})
    return tape_module

def get_data_tape(args):
    tape_elements = dict(
        U = (args.output_size, args.output_size),
        V = (args.input_size, args.input_size), 
        X = (args.num_samples, args.trajectory_length, args.input_size),
    )
    if args.single_output_y:
        tape_elements['y'] = (args.num_samples, 1, args.output_size)
    else:
        tape_elements['y'] = (args.num_samples, args.trajectory_length, args.output_size)
    tape_module = SimpleNamespace(**{k: np.zeros((args.num_seeds,) + v) for k,v in tape_elements.items()})
    return tape_module

def matrix_power(matrix, exponent):
    if exponent == 0: #<= 0:
        return identity_matrix(matrix.shape)
    product = matrix #.detach().numpy().copy()
    for i in range(exponent - 1):
        product = product @ matrix #.detach().numpy().copy()
    return product #torch.tensor(product)

def identity_matrix(shape): 
    matrix = np.zeros((shape))
    for i in range(matrix.shape[0]):
        matrix[i,i] = 1
    return matrix

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
        recurrent_modes_sem = np.zeros((len(args.lambda_0_range), len(args.S_T_range)))
    )
    tape_module = SimpleNamespace(**tape_elements)
    return tape_module

def _hash(obj):
    json_obj = encode_json(obj)
    return hash(json.dumps(json_obj, sort_keys=True))