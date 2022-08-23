# Sparse Federated Learning
This is the official MATLAB implementation of the ICML 2022 paper "**Fast Composite Optimization and Statistical Recovery in Federated Learning**" (https://arxiv.org/abs/2207.08204). 


The 3 experiments (linear regression, matrix estimation, logistic regression) can be reproduced with the three scripts `LassoRun.m`, `NuclearRun.m`, and `EmnistRun.m`. These scripts contain functions that can be run in batch mode from the command line or called from an interactive Matlab session.

The Lasso and Nuclear scripts do not have any dependencies, because the data is synthetic and generated within the scripts. To run the EMNIST experiment, you have to set up the dataset from the Leaf repository at https://github.com/TalwalkarLab/leaf.  Follow the instructions to from the Leaf repository to preprocess the FEMNIST data, then copy the directories `data/femnist/data/train` and `data/femnist/data/test` from the Leaf repository into the subdirectory `data/FederatedEMNIST` of this directory. The data can then be processed with the script `process_femnist.py` by running `python3 process_femnist.py`. This will produce the .mat files needed for the EMNIST-10 experiment. To prepare the EMNIST-62 dataset, change `DIGITS_ONLY` to `False` in `process_femnist.py`, and run `process_femnist.py` again. To run the EMNIST-62 experiments, change `digits_only` in `EmnistRun.m` to `false`.

The `LassoRun.m` should run in about 3 minutes, and `NuclearRun.m` should run in about 15 minutes. `EmnistRun.m` with `digits_only` set to `true` (EMNIST-10 experiment) should take around 10 hours, and the same script with `digits_only` set to `false` (EMNIST-62 experiment) should take about 24 hours (depending on your hardware).

Citation
---------
If you find this repo helpful, please cite the following paper:

```
@InProceedings{pmlr-v162-bao22b,
  title = 	 {Fast Composite Optimization and Statistical Recovery in Federated Learning},
  author =       {Bao, Yajie and Crawshaw, Michael and Luo, Shan and Liu, Mingrui},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {1508--1536},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR}
}

```
