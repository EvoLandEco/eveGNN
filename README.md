# eveGNN
eveGNN contains functions and scripts to apply Neural Network approaches on phylogenetic trees, for tree-level parameter estimation.

## Dependency
- Optuna >= 3.4.0
- R >= 4.2.1
- Python = 3.8.16
- PyTorch Geometric = 2.4.0
- eveGNN (in R, run `devtools::install_github("EvoLandEco/eveGNN")`)

## How to use
ADAM (Automated DAta Manager) is a simple shell program to help manage simulation data, train GNN model and more. The program and the scripts it calls are designed for a cluster computer in the SLURM environment.

- First, clone the repo:
```bash
git clone https://github.com/EvoLandEco/eveGNN
```
- Second, locate to bash, create a folder for new project:
```bash
cd Bash
mkdir myproject
```
- Finally, run ADAM:
```bash
bash ADAM.sh myproject
```
![624461](https://github.com/EvoLandEco/eveGNN/assets/57348932/f4bc8341-68d6-4c7b-86e4-36ee44391c09)
