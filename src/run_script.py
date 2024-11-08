from src import argparser
from src.simulate import run_simulation

if __name__ == "__main__":
    parser = argparser.get_parser() 
    args = parser.parse_args()

    run_simulation(args)