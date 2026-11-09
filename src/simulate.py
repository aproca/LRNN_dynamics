from src import utils
from src import linear_RNN

def run_simulation_unaligned_RNN(args): ## TODO: integrate this better with the below function; TODO: specify flag for other possible tasks
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
        if args.task_dynamics == 'diagonalized':
            if args.eigenspace_dynamics:
                data_tape = utils.generate_eigenspace_data_multiseed(args, data_tape, seed)
            else:
                data_tape = utils.generate_data_multiseed(args, data_tape, seed)
        else:
            data_tape = utils.generate_teacher_data(args, data_tape, seed)

        if args.aligned_student_weights:
            if args.eigenspace_dynamics:
                tape = utils.initialize_student_eigenspace_multiseed(args, data_tape, seed, tape)
            else:
                tape = utils.initialize_student_multiseed(args, data_tape, seed, tape)
        else:
            tape = utils.initialize_unaligned_student(args, seed, tape)
        model = linear_RNN.LinearRNN_multiseed(args, tape, seed)
        model.train(data_tape.X[seed],data_tape.y[seed]) 
        model.project_SVD_space(data_tape)
        model.compute_gradient_H()
        model.compute_NTK(data_tape.X[seed], data_tape.y[seed], seed)
        tape = model.tape

    return tape, data_tape
