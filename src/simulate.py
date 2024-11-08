import torch
from src.utils import generate_data, initialize_student, get_tape_module
from src.linear_RNN import LinearRNN
from types import SimpleNamespace
import numpy as np

def loss_function(y, y_hat):
    return (1/2)*torch.mean((y-y_hat)**2)

def simulate(args):
    args, X, y = generate_data(args) # TODO: generate the data 10 times
    tape = get_tape_module(args)
    W_x, W_h, W_y, dW_x, dW_h, dW_y, R_y, R_x = initialize_student(args) # TODO: 10 times in parallel; load into the tape
    tape.W_x[0,0] = W_x.detach().numpy() # TODO: don't set the tape here
    tape.W_h[0,0] = W_h.detach().numpy()
    tape.W_y[0,0] = W_y.detach().numpy()
    tape.dW_x[0,0] = dW_x.detach().numpy()
    tape.dW_h[0,0] = dW_h.detach().numpy()
    tape.dW_y[0,0] = dW_y.detach().numpy()
    tape.R_y[0,0] = R_y.detach().numpy()
    tape.R_x[0,0] = R_x.detach().numpy()
    # TODO: have arg setting where RNN is not aligned

    model = LinearRNN(args, init_W_x = W_x, init_W_h = W_h, init_W_y = W_y) # TODO: do for 10 seeds
    losses, W_x_ar, W_h_ar, W_y_ar, H_ar, W_x_grad_ar, W_h_grad_ar = model.train(X,y) # TODO: for 10 seeds, load tape inside the code
    a_sim, b_sim, c_sim, u_sim = model.project_SVD_space(W_x_ar, W_h_ar, W_y_ar, tape) # TODO: load onto tape
