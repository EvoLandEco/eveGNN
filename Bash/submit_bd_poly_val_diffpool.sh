#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=gnn_bd_poly_val_diffpool
#SBATCH --output=logs/gnn_bd_poly_val_diffpool-%j.log
#SBATCH --mem=64GB
#SBATCH --partition=gpu

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
python ../Script/val_bd_poly_DiffPool.py "$name" "$task_type"

deactivate