import numpy as np


def create_SV_trajectories(args):
    S = np.zeros((args.trajectory_length, args.output_size, args.input_size)) #torch.zeros((args.trajectory_length, args.output_size, args.input_size))
    for alpha in range(args.num_SV):
        S_alpha = get_SV_trajectory(args, args.S_config[alpha], args.delta[alpha], args.lambda_0[alpha], args.S_T[alpha])
        S[:,alpha,alpha] = S_alpha
    return S


def get_SV_trajectory(args, S_config, delta, lambda_0, S_T): 
    ## NOTE: for more custom-made SV trajectories, need to hard-code somewhere
    S_alpha = np.zeros(args.trajectory_length) #torch.zeros(args.trajectory_length)
    if S_config == 'constant':
        for t in range(1, args.trajectory_length+1):
            S_t = delta*lambda_0 #torch.tensor([delta*lambda_0])
            if (t == args.trajectory_length and S_T == S_t) or (t < args.trajectory_length) or (args.S_TeqS_t):
                S_alpha[t-1] = S_t
            else:
                S_alpha[t-1] = S_T
    elif S_config == 'exponential':
        for t in range(1, args.trajectory_length+1):
            S_t = delta*(lambda_0**t) #torch.tensor([delta*lambda_0**t])
            if (t == args.trajectory_length and S_T == S_t) or (t < args.trajectory_length) or (args.S_TeqS_t):
                S_alpha[t-1] = S_t
            else:
                S_alpha[t-1] = S_T
    elif S_config == 'inverse_exponential':
        for t in range(1, args.trajectory_length+1):
            S_t = delta*(lambda_0**(args.trajectory_length-t)) #torch.tensor([delta*lambda_0**(args.trajectory_length-t)])
            if (t == args.trajectory_length and S_T == S_t) or (t < args.trajectory_length) or (args.S_TeqS_t):
                S_alpha[t-1] = S_t
            else:
                S_alpha[t-1] = S_T
    return S_alpha

    