import numpy as np
import copy
import torch
import torch.nn.functional as F
from src.utils import loss_function
from src import utils

class LinearRNN_multiseed:
    def __init__(self, args, tape, seed):
        self.args = args
        self.tape = tape
        self.seed = seed
        self.W_x = torch.tensor(copy.deepcopy(tape.W_x[seed][0]), dtype=torch.float64)
        self.W_h = torch.tensor(copy.deepcopy(tape.W_h[seed][0]), dtype=torch.float64)
        self.W_y = torch.tensor(copy.deepcopy(tape.W_y[seed][0]), dtype=torch.float64)
        if not args.frozen_input_output:
            self.W_x.requires_grad_()
            self.W_y.requires_grad_()
        if not args.frozen_hidden_layer:
            self.W_h.requires_grad_()

    def forward(self, X): 
        y_hat = torch.zeros((self.args.num_samples, 1, self.args.output_size))
        h_t = torch.zeros((self.args.num_samples, self.args.hidden_size), requires_grad=False, dtype=torch.float64)
        h_trajectory = torch.zeros((self.args.num_samples, self.args.trajectory_length, self.args.hidden_size))
        for t in range(self.args.trajectory_length):
            h_t = torch.einsum("ij,bj->bi", self.W_h, h_t) + torch.einsum("ij,bj->bi", self.W_x, X[:,t])
            if self.args.nonlinear_RNN:
                h_t = F.relu(h_t)
            y_t = torch.einsum("ij,bj->bi", self.W_y, h_t)
            h_trajectory[:,t] = h_t
        y_hat = y_t.reshape(y_t.shape[0], 1, -1)

        return y_hat, h_trajectory

    def train(self, X, y):
        X = torch.tensor(X, dtype=torch.float64, requires_grad=False)
        y = torch.tensor(y, dtype=torch.float64, requires_grad=False)
        for e in range(self.args.num_epochs):
            y_hat, h_trajectory = self.forward(X)
            loss = loss_function(y, y_hat)
            if e % self.args.log_every_e_epoch == 0:
                self.tape.loss[self.seed][int(e/self.args.log_every_e_epoch)] = loss.detach().numpy()
            loss.backward()
            with torch.no_grad():
                if not self.args.frozen_input_output:
                    self.W_x -= self.args.lr*self.W_x.grad
                    self.W_y -= self.args.lr*self.W_y.grad
                    if e % self.args.log_every_e_epoch == 0:
                        self.tape.W_x_grad[self.seed][int(e/self.args.log_every_e_epoch)] = np.real_if_close(self.W_x.grad.detach().numpy())
                        self.tape.W_y_grad[self.seed][int(e/self.args.log_every_e_epoch)] = np.real_if_close(self.W_y.grad.detach().numpy())
                    self.W_x.grad = None
                    self.W_y.grad = None
                if not self.args.frozen_hidden_layer:
                    self.W_h -= self.args.lr*self.W_h.grad
                    if e % self.args.log_every_e_epoch == 0:
                        self.tape.W_h_grad[self.seed][int(e/self.args.log_every_e_epoch)] = np.real_if_close(self.W_h.grad.detach().numpy())
                    self.W_h.grad = None
            if e % self.args.log_every_e_epoch == 0 and self.args.log_H:
                if self.args.subsample_H:
                    self.tape.H[self.seed][int(e/self.args.log_every_e_epoch)] = copy.deepcopy(h_trajectory.detach().numpy())[:200,-1].reshape(200, 1, self.args.hidden_size)
                else:
                    self.tape.H[self.seed][int(e/self.args.log_every_e_epoch)] = copy.deepcopy(h_trajectory.detach().numpy())
            if e % self.args.log_every_e_epoch == 0 and self.args.log_y_hat:
                self.tape.y_hat[self.seed][int(e/self.args.log_every_e_epoch)] = copy.deepcopy(y_hat.detach().numpy())
            if (e < self.args.num_epochs - 1) :
                if (e+1) % self.args.log_every_e_epoch == 0: 
                    self.tape.W_x[self.seed][int((e+1)/self.args.log_every_e_epoch)] = np.real_if_close(copy.deepcopy(self.W_x.detach().numpy()))
                    self.tape.W_h[self.seed][int((e+1)/self.args.log_every_e_epoch)] = np.real_if_close(copy.deepcopy(self.W_h.detach().numpy()))
                    self.tape.W_y[self.seed][int((e+1)/self.args.log_every_e_epoch)] = np.real_if_close(copy.deepcopy(self.W_y.detach().numpy()))
    
    def project_SVD_space(self, data_tape): ## rename to include eigenspace
        if self.args.aligned_student_weights:
            if self.args.eigenspace_dynamics:
                for e in range(int(self.args.num_epochs/self.args.log_every_e_epoch)):
                    for alpha in range(self.args.num_SV):
                        self.tape.input_modes[self.seed][e][alpha] = (data_tape.P_inv[self.seed] @ self.tape.W_x[self.seed][e] @ data_tape.P[self.seed])[alpha,alpha]
                        self.tape.recurrent_modes[self.seed][e][alpha] = (data_tape.P_inv[self.seed] @ self.tape.W_h[self.seed][e] @ data_tape.P[self.seed])[alpha,alpha]
                        self.tape.output_modes[self.seed][e][alpha] = (data_tape.P_inv[self.seed] @ self.tape.W_y[self.seed][e] @ data_tape.P[self.seed])[alpha,alpha]
                        self.tape.input_output_modes[self.seed][e][alpha] = self.tape.input_modes[self.seed][e, alpha]*self.tape.output_modes[self.seed][e, alpha]
            else:
                for e in range(int(self.args.num_epochs/self.args.log_every_e_epoch)):
                    for alpha in range(self.args.num_SV):
                        self.tape.input_modes[self.seed][e][alpha] = (self.tape.R_x[self.seed][0].T @ self.tape.W_x[self.seed][e] @ data_tape.V[self.seed])[alpha,alpha]
                        self.tape.recurrent_modes[self.seed][e][alpha] = (self.tape.R_y[self.seed][0].T @ self.tape.W_h[self.seed][e] @ self.tape.R_x[self.seed][0])[alpha,alpha]
                        self.tape.output_modes[self.seed][e][alpha] = (data_tape.U[self.seed].T @ self.tape.W_y[self.seed][e] @ self.tape.R_y[self.seed][0])[alpha,alpha]
                        self.tape.input_output_modes[self.seed][e][alpha] = self.tape.input_modes[self.seed][e, alpha]*self.tape.output_modes[self.seed][e, alpha]
        elif self.args.hidden_size == self.args.input_size and self.args.hidden_size == self.args.output_size:
            for e in range(int(self.args.num_epochs/self.args.log_every_e_epoch)):
                try:
                    self.tape.input_modes[self.seed][e] = np.linalg.svd(self.tape.W_x[self.seed,e])[1]
                except: 
                    self.tape.input_modes[self.seed][e] = 0
                try:
                    self.tape.recurrent_modes[self.seed][e] = np.linalg.svd(self.tape.W_h[self.seed,e])[1]
                except:
                    self.tape.recurrent_modes[self.seed][e] = 0
                try:
                    self.tape.output_modes[self.seed][e] = np.linalg.svd(self.tape.W_y[self.seed,e])[1]
                except: 
                    self.tape.output_modes[self.seed][e] = 0
                try:
                    self.tape.input_output_modes[self.seed][e] = np.linalg.svd(self.tape.W_y[self.seed,e] @ self.tape.W_x[self.seed,e])[1]
                except:
                    self.tape.input_output_modes[self.seed][e] = 0
    
    def compute_gradient_H(self):
        if self.args.log_H:
            for e in range(int(self.args.num_epochs/self.args.log_every_e_epoch)-1):
                self.tape.H_grad[self.seed][e+1] = self.tape.H[self.seed][e+1] - self.tape.H[self.seed][e]

    def compute_gradient_y_hat(self):
        if self.args.log_y_hat:
            for e in range(int(self.args.num_epochs/self.args.log_every_e_epoch)-1):
                self.tape.y_hat_grad[self.seed][e+1] = self.tape.y_hat[self.seed][e+1] - self.tape.y_hat[self.seed][e]

    def compute_hidden_layer_NTK(self, X, y): 
        if self.args.compute_NTK:
            num_samples = self.args.num_samples
            if self.args.subsample_NTK:
                X = X[:50]
                y = y[:50]
                num_samples = 50
            tau = (num_samples*self.args.output_size)/self.args.lr
            T = self.args.trajectory_length
            t = self.args.trajectory_length ## arbitrary choice for the point in hidden layer to compute NTK over
            for e in range(self.args.num_epochs-1):
                first_half = 0
                second_half = 0
                for i in range(1, t+1):
                    for j in range(1, T+1):
                        if j < T:
                            for r in range(0, T-j):
                                first_term = (t-i)*np.einsum('bj,jj,jp->bp', X[:,i-1], self.tape.W_x[self.seed][e].T @ utils.matrix_power(self.tape.W_h[self.seed][e], T-j-1-r) @ self.tape.W_x[self.seed][e], X[:,j-1].T)
                                second_term = utils.matrix_power(self.tape.W_h[self.seed][e], t-i-1) @ utils.matrix_power(self.tape.W_h[self.seed][e], r).T @ self.tape.W_y[self.seed][e].T
                                first_half += np.kron(first_term, second_term)
                        first_term = np.einsum('bj,jp->bp', X[:,i-1], X[:,j-1].T)
                        second_term = utils.matrix_power(self.tape.W_h[self.seed][e],t-i) @ utils.matrix_power(self.tape.W_h[self.seed][e], T-j).T @ self.tape.W_y[self.seed][e].T
                        second_half += np.kron(first_term, second_term)
                self.tape.H_NTK[self.seed][e] = (first_half + second_half)/tau # P x P
                y_hat = 0
                for k in range(1,T+1):
                    y_hat += np.einsum('ij,bj->bi', utils.matrix_power(self.tape.W_h[self.seed][e], T-k) @ self.tape.W_x[self.seed][e], X[:,k-1])
                y_hat = np.einsum('ij,bj->bi', self.tape.W_y[self.seed][e], y_hat)
                error = (y.reshape(y_hat.shape)-y_hat).T        # N_y x P
                self.tape.H_Tp1_grad[self.seed][e] = (self.tape.H_NTK[self.seed][e] @ error.reshape(self.args.output_size*num_samples).reshape(self.args.output_size,num_samples)).T 

    def compute_NTK(self, X, y, seed): 
        if self.args.compute_NTK:
            num_samples = self.args.num_samples
            if self.args.subsample_NTK:
                X = X[:50]
                y = y[:50]
                num_samples = 50
            tau = (num_samples*self.args.output_size)/self.args.lr 
            T = self.args.trajectory_length
            t = self.args.trajectory_length
            for e in range(int(self.args.num_epochs/self.args.log_every_e_epoch)):
                Wh_term, Wx_term, Wy_term = 0,0,0
                for i in range(1, t+1):
                    for j in range(1, T+1): 
                        if j < T:
                            for r in range(0, T-j):
                                first_term = (t-i)*np.einsum('bj,jj,jp->bp',X[:,i-1],self.tape.W_x[seed][e].T @ utils.matrix_power(self.tape.W_h[seed][e], T-j-1-r) @ self.tape.W_x[seed][e], X[:,j-1].T) 
                                second_term = self.tape.W_y[seed][e] @ utils.matrix_power(self.tape.W_h[seed][e], t-i-1) @ utils.matrix_power(self.tape.W_h[seed][e], r).T @ self.tape.W_y[seed][e].T
                                Wh_term += np.kron(first_term, second_term)
                        first_term = np.einsum('bj,jp->bp', X[:,i-1], X[:,j-1].T)
                        second_term = self.tape.W_y[seed][e] @ utils.matrix_power(self.tape.W_h[seed][e], t-i) @ utils.matrix_power(self.tape.W_h[seed][e], T-j).T @ self.tape.W_y[seed][e].T
                        Wx_term += np.kron(first_term, second_term)
                        first_term = np.eye(self.args.output_size)
                        second_term = np.einsum('bj,jj,jp->bp',X[:,i-1], self.tape.W_x[seed][e].T @ utils.matrix_power(self.tape.W_h[seed][e], t-i).T @ utils.matrix_power(self.tape.W_h[seed][e], T-j) @ self.tape.W_x[seed][e], X[:,j-1].T)
                        Wy_term += np.kron(first_term, second_term)
                self.tape.NTK[seed][e] = (Wh_term + Wx_term + Wy_term)/tau
                y_hat = 0
                for k in range(1,T+1):
                    y_hat += np.einsum('ij,bj->bi', utils.matrix_power(self.tape.W_h[seed][e], T-k) @ self.tape.W_x[seed][e], X[:,k-1])
                y_hat = np.einsum('ij,bj->bi', self.tape.W_y[seed][e], y_hat)
                error = (y.reshape(y_hat.shape) - y_hat).T
                # non-vectorized
                self.tape.pred_y_grad[seed][e] = (self.tape.NTK[seed][e] @ error.reshape(self.args.output_size*num_samples)).reshape(self.args.output_size,1, num_samples).T 
                NTK_reshape = self.tape.NTK[seed].reshape(int(self.args.num_epochs/self.args.log_every_e_epoch), num_samples*self.args.output_size, num_samples*self.args.output_size)
                self.tape.NTK_distance[seed] = utils.kernel_distance(NTK_reshape[0], NTK_reshape[-1])