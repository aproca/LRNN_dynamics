import torch
import numpy as np
import copy
from types import SimpleNamespace
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# For single-output y
def compute_YXt_correlation(X, y, t):
    X_t = X[:,t-1]
    # YX_t = 0
    # for p in range(1, y.shape[0]+1):
    #     YX_t += y[p-1].reshape(-1,1) @ X[p-1,t-1].reshape(1,-1)

    # parallelized
    YX_t = torch.einsum('pij,pjk->pik', y.reshape(y.shape[0], y.shape[1],1), X_t.reshape(X_t.shape[0], 1, -1))
    YX_t = torch.sum(YX_t, axis=0)
    return YX_t

# For multi-output y
def compute_YtXi_correlation(X, y, t, i):
    X_i = X[:,i-1]
    y_t = y[:,t-1]

    # YtXi = 0
    # for p in range(1, X.shape[0]+1):
    #     YtXi += y[p-1,t-1].reshape(1,-1) @ X[p-1,i-1].reshape(1,-1)

    # parallelized
    YtXi = torch.einsum('pij,pjk->pik', y_t.reshape(y_t.shape[0], y_t.shape[1],1), X_i.reshape(X_i.shape[0], 1, -1))
    YtXi = torch.sum(YtXi, axis=0)
    
    return YtXi

# def loss_function(y, y_hat): # double check whether to use this loss
#     return torch.mean((y-y_hat)**2)

# def linear_RNN(W_x, W_h, W_y, h_t, x_t):
#     h_tp1 = torch.einsum("ij,bj->bi", W_h, h_t) + torch.einsum("ij,bj->bi", W_x, x_t)
#     y_t = torch.einsum("ij,bj->bi", W_y, h_tp1)
#     return h_tp1, y_t

# For single-output y
def generate_y_T(args, X): # TODO: parallelize
    # y = torch.zeros((args.num_samples, 1, args.output_size))
    # for p in range(args.num_samples):
    #     y_p = 0
    #     for t in range(args.trajectory_length):
    #         y_p += args.U @ torch.diag(args.S[t]) @ args.V.T @ X[p,t] 
    #         # U is outxout, S is time x out, V is inxin, X is samples x time x in x in, 
    #         # y_p is out
    #     y[p] = y_p
    # parallelized- check
    y = torch.einsum('ij,tjk,kl,ptl->pi', args.U, args.S, args.V.T, X)
    y = y.reshape(y.shape[0], 1, -1)
    return y

# For multi-output y
def generate_y(args, X): # TODO: parallelize
    y = torch.zeros((args.num_samples, args.trajectory_length, args.output_size))
    for p in range(args.num_samples):
        for t_y in range(args.trajectory_length):
            y_pt = 0
            for t_x in range(args.trajectory_length):
                y_pt += args.U @ torch.diag(args.S[t_y,t_x]) @ args.V.T @ X[p,t_x]
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
        XtXt = torch.einsum('pij,pjk->pik',X_t.reshape(X_t.shape[0], -1, 1), X_t.reshape(X_t.shape[0], 1, -1))
        XtXt = torch.sum(XtXt, axis=0)
        # print('Expected: I, Simulation: ', XtXt)
        isclose = torch.isclose(torch.eye(X.shape[2]), XtXt, atol=1e-2)
        true_instances += torch.count_nonzero(isclose)
        total_instances += isclose.flatten().shape[0]
        # print(isclose)

    for t in range(1,args.trajectory_length+1):
        for t_prime in range(1,args.trajectory_length+1):
            if t != t_prime:
                XtXtprime = 0
                for p in range(1,args.num_samples+1):
                    XtXtprime += X[p-1,t-1].reshape(-1,1) @ X[p-1, t_prime-1].reshape(1,-1)
                # print('Expected: 0, Simulation: ', XtXtprime)
                isclose = torch.isclose(torch.zeros(X.shape[2], X.shape[2]), XtXtprime, atol=1e-2)
                true_instances += torch.count_nonzero(isclose)
                total_instances += isclose.flatten().shape[0]
                # print(isclose)
    print(true_instances/total_instances)
    return true_instances/total_instances

def initialize_LR_SVec(args):
    teacher = torch.rand((args.output_size, args.input_size), requires_grad = False)
    U, _, V = torch.svd(teacher, some=False)
    args.U, args.V = U, V
    return args

# def whiten(X): # extend this to dimensions of X
#     scaler = StandardScaler()
#     X_standardised = scaler.fit_transform(X)
#     pca = PCA()
#     X_pca = pca.fit_transform(X_standardised)
#     # print('explained variance: ', pca.explained_variance_)
#     X_whitened = X_pca / np.sqrt(pca.explained_variance_)
#     X_whitened = np.sqrt(X.shape[0] / (X.shape[0] - 1)) * X_whitened
#     return X_whitened

def generate_X(args):
    X = torch.normal(0, 0.01, size=(args.num_samples, args.trajectory_length, args.input_size), dtype=torch.float32, requires_grad=False) * np.sqrt(10000/args.num_samples)
    # X = whiten(X)
    while check_x_uncorrelated_whitened(args, X) < 0.7:
        X = torch.normal(0, 0.01, size=(args.num_samples, args.trajectory_length, args.input_size), dtype=torch.float32, requires_grad=False) * np.sqrt(10000/args.num_samples)
    #     X = whiten(X)
    return X

def generate_data(args):
    args = initialize_LR_SVec(args)
    X = generate_X(args)
    if args.single_output_y:
        y = generate_y_T(args, X)
    else:
        y = generate_y(args, X)
    return args, X, y

def initialize_student(args):
    r = torch.rand(args.hidden_size, args.hidden_size)
    u,s,v = torch.svd(r, some=False)
    R_y = v.T
    R_x = torch.linalg.inv(v)

    dW_x = torch.zeros(args.hidden_size, args.input_size)
    if args.zerobalanced:
        torch.diag(torch.sqrt(args.inout_mode_init), 0, out=dW_x)
    else:
        torch.diag(args.in_mode_init, 0, out=dW_x)
    W_x = R_x @ dW_x @ args.V.T
    if not args.frozen_input_output:
        W_x.requires_grad_()

    dW_y = torch.zeros(args.output_size, args.hidden_size)
    if args.zerobalanced:
        torch.diag(torch.sqrt(args.inout_mode_init), 0, out=dW_y)
    else:
        torch.diag(args.out_mode_init, 0, out=dW_y)
    W_y = args.U @ dW_y @ R_y.T
    if not args.frozen_input_output:
        W_y.requires_grad_()

    dW_h = torch.zeros(args.hidden_size, args.hidden_size)
    torch.diag(args.rec_mode_init, 0, out=dW_h)
    W_h = R_y @ dW_h @ R_x.T
    if not args.frozen_hidden_layer:
        W_h.requires_grad_()

    return W_x, W_h, W_y, dW_x, dW_h, dW_y, R_y, R_x

# def run_iteration(args, W_x, W_h, W_y, X):
#     y = torch.zeros((args.num_samples, args.trajectory_length, args.output_size))
#     if args.single_output_y:
#         y = torch.zeros((args.num_samples, 1, args.output_size))
#     h_t = torch.zeros((args.num_samples,args.hidden_size), requires_grad=False)
#     for t in range(args.trajectory_length):
#         h_t, y_t = linear_RNN(W_x, W_h, W_y, h_t, X[:,t])
#         y[:,t] = y_t.reshape(y_t.shape[0], 1, -1)
    
#     if args.single_output_y:
#         y = y_t.reshape(y_t.shape[0], 1, -1)
#     return y

# def run_training(args, tape, W_x, W_h, W_y, X, y):
#     W_x_ar = [copy.deepcopy(W_x.detach().numpy())]
#     W_h_ar = [copy.deepcopy(W_h.detach().numpy())]
#     W_y_ar = [copy.deepcopy(W_y.detach().numpy())]
#     for e in range(args.num_epochs):
#         y_hat = run_iteration(args, W_x, W_h, W_y, X)
#         loss = loss_function(y, y_hat)
#         tape = eqx.tree_at(lambda tape: getattr(tape, f"loss"), tape,  getattr(tape, f"loss").at[ti_tape].set(loss))
#         loss.backward()
#         with torch.no_grad():
#             W_x -= args.lr*W_x.grad
#             W_y -= args.lr*W_y.grad
#             if not args.frozen_hidden_layer:
#                 W_h -= args.lr*W_h.grad
#                 W_h.grad = None
#                 W_h_ar.append(copy.deepcopy(W_h.detach().numpy()))
#             W_x.grad = None
#             W_y.grad = None
#             W_x_ar.append(copy.deepcopy(W_x.detach().numpy()))
#             W_y_ar.append(copy.deepcopy(W_y.detach().numpy()))
#     if not args.frozen_hidden_layer:
#         return W_x_ar, W_h_ar, W_y_ar
#     return W_x_ar, W_y_ar

# def get_tape_module(args):
#     tape_var_specs = dict(
#         loss = (), 
#         t = (),
#     )
#     tape_var_specs['U'] = (args.output_size, args.output_size)
#     tape_var_specs['V'] = (args.input_size, args.input_size)
#     # tape_var_specs['S'] = (args.output_size, args.output_size) # correct sizing
#     tape_var_specs['W_x'] = (args.hidden_size, args.input_size)
#     tape_var_specs['W_h'] = (args.hidden_size, args.hidden_size)
#     tape_var_specs['W_y'] = (args.output_size, args.hidden_size)
#     tape_var_specs['grad_W_x'] = (args.hidden_size, args.input_size)
#     tape_var_specs['grad_W_h'] = (args.hidden_size, args.hidden_size)
#     tape_var_specs['grad_W_y'] = (args.output_size, args.hidden_size)

#     Tape = type('Tape', (eqx.Module,), {k: None for k in tape_var_specs.keys()} | {"__annotations__": {k: v for k, v in tape_var_specs.items()}} | {"__getitem__": lambda self, k: getattr(self, k)})

#     ts_out = np.linspace(0, args.num_epochs, args.num_epochs, dtype=int)  # log 1000 times
#     tape_module = Tape(**{k: np.empty((args.num_seeds, len(ts_out)) + v) for k, v in tape_var_specs.items()})
#     return tape_module

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
        R_y = (args.hidden_size, args.hidden_size)
    )
    tape_module = SimpleNamespace(**{k: np.zeros((args.num_seeds, args.num_epochs) + v) for k,v in tape_elements.items()})
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