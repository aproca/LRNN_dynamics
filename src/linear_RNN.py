import numpy as np
import copy
import torch
from src.simulate import loss_function

# using autograd for now, switch to using just the gradient equations later on
class LinearRNN:
    def __init__(self, args, init_W_x = None, init_W_h = None, init_W_y = None):

        self.args = args
        if init_W_x is not None and init_W_h is not None and init_W_y is not None:
            self.W_x = init_W_x
            self.W_h = init_W_h
            self.W_y = init_W_y
        else:
            self.W_x = np.random.randn(args.hidden_size, args.input_size)
            self.W_h = np.random.randn(args.hidden_size, args.hidden_size)
            self.W_y = np.random.randn(args.output_size, args.hidden_size)

    def forward(self, X): 
        if self.args.single_output_y:
            y_hat = torch.zeros((self.args.num_samples, 1, self.args.output_size))
        else:
            y_hat = torch.zeros((self.args.num_samples, self.args.trajectory_length, self.args.output_size))
        h_t = torch.zeros((self.args.num_samples, self.args.hidden_size), requires_grad=False)
        h_trajectory = torch.zeros((self.args.num_samples, self.args.trajectory_length, self.args.hidden_size))
        for t in range(self.args.trajectory_length):
            h_t = torch.einsum("ij,bj->bi", self.W_h, h_t) + torch.einsum("ij,bj->bi", self.W_x, X[:,t])
            y_t = torch.einsum("ij,bj->bi", self.W_y, h_t)
            h_trajectory[:,t] = h_t
            if not self.args.single_output_y:
                y_hat[:,t] = y_t
        if self.args.single_output_y:
            y_hat = y_t.reshape(y_t.shape[0], 1, -1)

        return y_hat, h_trajectory

    def train(self, X, y):
        W_x_ar = [copy.deepcopy(self.W_x.detach().numpy())]
        W_h_ar = [copy.deepcopy(self.W_h.detach().numpy())]
        W_y_ar = [copy.deepcopy(self.W_y.detach().numpy())]
        W_h_grad_ar = []
        W_x_grad_ar = []
        losses = []
        H_ar = []
        for e in range(self.args.num_epochs):
            y_hat, h_trajectory = self.forward(X)
            loss = loss_function(y, y_hat)
            losses.append(loss.detach().numpy())
            loss.backward()
            with torch.no_grad():
                if not self.args.frozen_input_output:
                    self.W_x -= self.args.lr*self.W_x.grad
                    self.W_y -= self.args.lr*self.W_y.grad
                    W_x_grad_ar.append(self.W_x.grad.detach().numpy())
                    # W_y_grad_ar.append(self.W_y.grad.detach().numpy())
                    self.W_x.grad = None
                    self.W_y.grad = None
                if not self.args.frozen_hidden_layer:
                    self.W_h -= self.args.lr*self.W_h.grad
                    W_h_grad_ar.append(self.W_h.grad.detach().numpy())
                    self.W_h.grad = None
            W_x_ar.append(copy.deepcopy(self.W_x.detach().numpy()))
            W_h_ar.append(copy.deepcopy(self.W_h.detach().numpy()))
            W_y_ar.append(copy.deepcopy(self.W_y.detach().numpy()))
            H_ar.append(copy.deepcopy(h_trajectory.detach().numpy()))

        return losses, W_x_ar, W_h_ar, W_y_ar, H_ar, W_x_grad_ar, W_h_grad_ar
    
    def project_SVD_space(self, W_x_ar, W_h_ar, W_y_ar, tape):
        a = np.zeros((self.args.num_epochs, self.args.num_SV))
        b = np.zeros((self.args.num_epochs, self.args.num_SV))
        c = np.zeros((self.args.num_epochs, self.args.num_SV))
        u = np.zeros((self.args.num_epochs, self.args.num_SV))
        for e in range(self.args.num_epochs):
            for alpha in range(self.args.num_SV):
                a[e, alpha] = (tape.R_x[0,0].T @ W_x_ar[e] @ self.args.V.detach().numpy())[alpha,alpha]
                b[e, alpha] = (tape.R_y[0,0].T @ W_h_ar[e] @ tape.R_x[0,0])[alpha,alpha]
                c[e, alpha] = (self.args.U.detach().numpy().T @ W_y_ar[e] @ tape.R_y[0,0])[alpha,alpha]
                u[e, alpha] = a[e, alpha]*c[e, alpha]
        return a,b,c,u
    
    def compute_gradient_H(self, H_ar):
        H_grad_ar = []
        for e in range(self.args.num_epochs-1):
            H_grad_ar.append(H_ar[e+1] - H_ar[e])
        return H_grad_ar