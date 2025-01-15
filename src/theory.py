import numpy as np
import math
from src.utils import matrix_power

class TheoryRNN:
    def __init__(self, args):
        self.args = args

    def input_output_solution(self): 
        tau = (self.args.num_samples*self.args.output_size)/self.args.lr
        T = self.args.trajectory_length
        u_pred = np.zeros((self.args.num_SV, self.args.num_epochs))
        for alpha in range(self.args.num_SV):
            b = self.args.rec_mode_init[alpha]
            u = self.args.inout_mode_init[alpha]
            beta_s = 0
            beta_b = 0
            if self.args.single_output_y:
                for i in range(1, T+1):
                    beta_s += (b**(T-i))*self.args.S[i-1, alpha, alpha]
                    beta_b += (b**(2*(T-i)))
            else:
                for t in range(1, T+1):
                    for i in range(1, t+1):
                        beta_s += (b**(t-i))*self.args.S[i-1, alpha, alpha]
                        beta_b += (b**(2*(t-i)))
            for e in range(1, self.args.num_epochs+1):
                numer = np.exp(2*e*beta_s/tau)*beta_s
                denom = (beta_s/u) - beta_b + np.exp(2*e*beta_s/tau)*beta_b
                u_pred[alpha,e-1] = numer/denom
        return u_pred

    # def recurrent_solution(self, omega): ## have to setup the sv so that a single omega solves equation
    #     tau = (self.args.num_samples*self.args.output_size)/self.args.lr
    #     T = self.args.trajectory_length
    #     b_pred = np.zeros((self.args.num_SV, self.args.num_epochs))
    #     for alpha in range(self.args.num_SV):
    #         b = self.args.rec_mode_init[alpha]
    #         u = self.args.inout_mode_init[alpha]
    #         sum_term = 0
    #         for i in range(1, T):
    #             sum_term += (T-i)*(T-i-1)*self.args.S[i-1, alpha, alpha]*(omega[alpha]**(T-i-2))
    #             sum_term += (T-i)*(2*(T-i-1)+1)*u*(omega[alpha]**(2*(T-i-1)))
    #         for e in range(1,self.args.num_epochs+1):
    #             b_pred[alpha,e-1] = omega[alpha] + (b - omega[alpha])*np.exp((e*u*sum_term)/tau)
    #     return b_pred
    ## THIS IS THE WRONG VERSION !!
    def recurrent_solution(self): ## have to setup the sv so that a single omega solves equation
        tau = (self.args.num_samples*self.args.output_size)/self.args.lr
        T = self.args.trajectory_length
        b_pred = np.zeros((self.args.num_SV, self.args.num_epochs))
        for e in range(1, self.args.num_epochs+1):
            for alpha in range(self.args.num_SV):
                b = self.args.rec_mode_init[alpha]
                u = self.args.inout_mode_init[alpha]
                sum_term = 0
                for n in range(1, T):
                    inner_sum_term = 0
                    for i in range(1, T-n+1):
                        inner_sum_term += (math.factorial(T-i) / math.factorial(T-i-n))*u*self.args.S[i-1, alpha, alpha]*(b**(T-i-n))
                        inner_sum_term += (((T-i)*math.factorial(2*T-2*i-1))/(math.factorial(2*T-2*i-n)))*(u**2)*(b**(2*T-2*i-n))
                    sum_term += (inner_sum_term * (e**n)/(math.factorial(n)))
                sum_term = sum_term / tau
                b_pred[alpha,e-1] = b + sum_term
        return b_pred
    
    def hidden_layer_ode(self, W_x_ar, W_h_ar, W_x_grad_ar, W_h_grad_ar, X, T): 
        grad_h_Tp1 = []
        for e in range(self.args.num_epochs-1):
            first_term = 0
            second_term = 0
            for i in range(1, T):
                first_term += -(T - i)*np.einsum('ij,bj->bi', matrix_power(W_h_ar[e],T-i-1) @ W_h_grad_ar[e] @ W_x_ar[e], X[:,i-1].detach().numpy())
            for i in range(1,T+1):    
                second_term += -np.einsum('ij,bj->bi', matrix_power(W_h_ar[e], T-i) @ W_x_grad_ar[e], X[:,i-1].detach().numpy()) 
            grad_h_Tp1.append(first_term + second_term)
        return np.array(grad_h_Tp1)*self.args.lr