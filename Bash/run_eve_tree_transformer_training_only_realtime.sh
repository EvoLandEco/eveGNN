#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <name> [task_type] [run_id]"
    exit 1
fi

name=$1
task_type=${2:-EVE_FREE_TES}
run_id=${3:-1}

bash_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "$bash_dir/.." && pwd)

train_script=${TRAIN_SCRIPT:-$project_root/Script/train_eve_pars_est_TreeTransformer_realtime.py}
train_config=${TRAIN_CONFIG:-$project_root/Config/eve_train_tree_transformer.yaml}
data_dir="$bash_dir/$name/$task_type/GNN"

mkdir -p "$bash_dir/logs"

if [ ! -f "$train_script" ]; then
    echo "Missing training script: $train_script"
    exit 1
fi
if [ ! -f "$train_config" ]; then
    echo "Missing training config: $train_config"
    exit 1
fi
if [ ! -d "$data_dir/tree" ] || [ ! -d "$data_dir/tree/EL" ] || [ ! -d "$data_dir/tree/BT" ]; then
    echo "Missing exported data folders under: $data_dir"
    exit 1
fi

job_id=$(sbatch --parsable --chdir="$bash_dir" <<EOF_SLURM
#!/bin/bash
#SBATCH --time=6-22:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=gnn_eve_pars_est_tree_transformer
#SBATCH --output=logs/gnn_eve_pars_est_tree_transformer-%j.log
#SBATCH --error=logs/gnn_eve_pars_est_tree_transformer-%j.log
#SBATCH --mem=48GB
#SBATCH --partition=gpu
#SBATCH --signal=B:USR1@300

set -euo pipefail

name="$name"
task_type="$task_type"
run_id="$run_id"
train_script="$train_script"
train_config="$train_config"

export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0

ml Python/3.8.16-GCCcore-11.2.0
source \$HOME/venvs/eve/bin/activate

which python
python --version
python - <<'PY_CHECK'
import importlib
for pkg in ["numpy", "pandas", "torch", "yaml", "pyreadr"]:
    importlib.import_module(pkg)
print("Python packages visible", flush=True)
PY_CHECK

date
stdbuf -oL -eL python -u "\$train_script" "\$name" "\$task_type" "\$run_id" --config "\$train_config"
date

deactivate
EOF_SLURM
)

echo "Submitted training job: ${job_id%%;*}"
echo "Log: $bash_dir/logs/gnn_eve_pars_est_tree_transformer-${job_id%%;*}.log"
