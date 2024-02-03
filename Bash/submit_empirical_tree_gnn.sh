#!/bin/bash
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=gnn_emp_gnn
#SBATCH --output=logs/gnn_emp_gnn-%j.log
#SBATCH --mem=16GB
#SBATCH --partition=gpu

ml R
ml Python/3.8.16-GCCcore-11.2.0
source $HOME/venvs/eve/bin/activate

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <name> <task_type> <gnn_depth>"
    exit 1
fi

# Get the arguments
name=$1
gnn_depth=$2

# Firstly we export the empirical tree to GNN representation
Rsript ../Script/empirical_tree_gnn_export.R

# Call the Python script with the arguments
python ../Script/app_bd_pars_est_DiffPool.py "$name" "$gnn_depth"
python ../Script/app_ddd_pars_est_DiffPool.py "$name" "$gnn_depth"
python ../Script/app_pbd_pars_est_DiffPool_full.py "$name" "$gnn_depth"

deactivate