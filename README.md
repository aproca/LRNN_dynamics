# Learning dynamics in linear recurrent neural networks
by Alexandra M. Proca, Clémentine C.J. Dominé, Murray Shanahan, Pedro A.M. Mediano<br/>

Code for [Learning dynamics in linear recurrent neural networks](https://openreview.net/forum?id=KGOcrIWYnx) (ICML 2025 Oral).

## Setup
To set up the conda environment, run:
```
conda env create -f environment.yml
conda activate environment
```

## Code
For simulations, configurations of hyperparameters can be created using `src/argparser.py` and be run using `run_script.py`. <br/> <br/>

The folders `configs/` and `data/` contain configurations and training logs for the sweeps used in the phase transition figure (Fig 4) and the feature learning experiments (Fig. 5, A4-7). To run the simulations and generate the data yourself, you can use `run_SV_search_nosave.py`,`configs/SV_search_config.py` and `run_rich_lazy_search.py`,`configs/rich_lazy_search_config.py`, respectively.

## Figures
Notebooks to reproduce all simulations and figures in the manuscript (labeled by the respective figure) can be found in ```figure_notebooks/```.

## Citation
Please cite our paper if you use this code in your research.

```
@inproceedings{proca2025learning,
  title={Learning dynamics in linear recurrent neural networks},
  author={Alexandra Maria Proca and Cl{\'e}mentine Carla Juliette Domin{\'e} and Murray Shanahan and Pedro A. M. Mediano},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=KGOcrIWYnx}
}
```