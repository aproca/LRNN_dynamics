# from src.utils import generate_data, initialize_student, get_tape_module
from src import utils
from src import linear_RNN
# from src.linear_RNN import LinearRNN

def run_simulation_unaligned_RNN(args): ## TODO: integrate this better with the below function
    data_tape = utils.get_data_tape(args)
    tape = utils.get_tape_module(args)

    for seed in range(args.num_seeds):
        utils.set_seeds(seed)
        data_tape = utils.generate_data_multiseed(args, data_tape, seed) 
        tape = utils.initialize_unaligned_student(args, seed, tape)
        model = linear_RNN.LinearRNN_multiseed(args, tape, seed)
        model.train(data_tape.X[seed], data_tape.y[seed]) 
        model.compute_gradient_H()
        model.compute_hidden_layer_NTK(data_tape.X[seed], data_tape.y[seed])
        tape = model.tape

    return tape, data_tape

def run_simulation_multiseed(args):
    data_tape = utils.get_data_tape(args)
    tape = utils.get_tape_module(args)

    for seed in range(args.num_seeds):
        utils.set_seeds(seed)
        data_tape = utils.generate_data_multiseed(args, data_tape, seed) 
        tape = utils.initialize_student_multiseed(args, data_tape, seed, tape)
        model = linear_RNN.LinearRNN_multiseed(args, tape, seed)
        model.train(data_tape.X[seed],data_tape.y[seed]) 
        model.project_SVD_space(data_tape)
        model.compute_gradient_H()
        tape = model.tape

    return tape, data_tape

def run_simulation_oneseed(args):
    args, X, y = utils.generate_data(args) 
    tape = utils.get_tape_module(args)
    W_x, W_h, W_y, dW_x, dW_h, dW_y, R_y, R_x = utils.initialize_student(args) # TODO: 10 times in parallel; load into the tape
    tape.W_x[0,0] = W_x.detach().numpy() # TODO: don't set the tape here
    tape.W_h[0,0] = W_h.detach().numpy()
    tape.W_y[0,0] = W_y.detach().numpy()
    tape.dW_x[0,0] = dW_x.detach().numpy()
    tape.dW_h[0,0] = dW_h.detach().numpy()
    tape.dW_y[0,0] = dW_y.detach().numpy()
    tape.R_y[0,0] = R_y.detach().numpy()
    tape.R_x[0,0] = R_x.detach().numpy()
    # TODO: have arg setting where RNN is not aligned

    model = linear_RNN.LinearRNN(args, init_W_x = W_x, init_W_h = W_h, init_W_y = W_y) # TODO: do for 10 seeds
    losses, W_x_ar, W_h_ar, W_y_ar, H_ar, W_x_grad_ar, W_h_grad_ar = model.train(X,y) # TODO: for 10 seeds, load tape inside the code
    a_sim, b_sim, c_sim, u_sim = model.project_SVD_space(W_x_ar, W_h_ar, W_y_ar, tape) # TODO: load onto tape
