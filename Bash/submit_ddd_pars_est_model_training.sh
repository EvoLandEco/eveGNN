#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gnn_ddd_pars_est_model_train
#SBATCH --output=logs/gnn_ddd_pars_est_model_train-%j.log
#SBATCH --mem=3GB
#SBATCH --partition=regular

ml Python/3.8.16-GCCcore-11.2.0
source $HOME/venvs/eve/bin/activate

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <name> <task_type>"
    exit 1
fi

# Get the arguments
name=$1
task_type=$2

# Call the Python script with the arguments
python ../Script/train_ddd_pars_est.py "$name" "$task_type"

deactivate