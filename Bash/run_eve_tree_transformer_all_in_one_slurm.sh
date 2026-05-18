#!/bin/bash
set -e

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <name> [task_type] [run_id]"
    exit 1
fi

name=$1
task_type=${2:-EVE_FREE_TES}
run_id=${3:-1}
poll_seconds=${POLL_SECONDS:-60}

bash_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "$bash_dir/.." && pwd)

sim_script=${SIM_SCRIPT:-$project_root/Script/eve_pars_est_bd_ed_nnd_data.R}
sim_config=${SIM_CONFIG:-$project_root/Config/eve_sim.yaml}
train_script=${TRAIN_SCRIPT:-$project_root/Script/train_eve_pars_est_TreeTransformer.py}
train_config=${TRAIN_CONFIG:-$project_root/Config/eve_train_tree_transformer.yaml}

mkdir -p "$bash_dir/logs"

if [ ! -f "$sim_script" ]; then
    echo "Missing simulation script: $sim_script"
    exit 1
fi
if [ ! -f "$sim_config" ]; then
    echo "Missing simulation config: $sim_config"
    exit 1
fi
if [ ! -f "$train_script" ]; then
    echo "Missing training script: $train_script"
    exit 1
fi
if [ ! -f "$train_config" ]; then
    echo "Missing training config: $train_config"
    exit 1
fi

wait_job_gone() {
    local jid=$1
    while squeue -h -j "$jid" 2>/dev/null | grep -q .; do
        sleep "$poll_seconds"
    done
}

job_state() {
    local jid=$1
    local state=""
    for _ in $(seq 1 20); do
        state=$(sacct -j "$jid" --format=State -n -X 2>/dev/null | awk 'NF {print $1; exit}')
        if [ -n "$state" ]; then
            echo "$state"
            return 0
        fi
        sleep 3
    done
    echo "UNKNOWN"
}

echo "Submitting simulation job"
sim_job_submit=$(sbatch --parsable --chdir="$bash_dir" <<EOF_SIM
#!/bin/bash
set -e
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=gnn_eve_pars_free
#SBATCH --output=logs/gnn_eve_pars_free-%j.log
#SBATCH --mem=48GB
#SBATCH --partition=regular

name="$name"
task_type="$task_type"
sim_script="$sim_script"
sim_config="$sim_config"

ml R
Rscript -e 'install.packages("devtools", repos="http://cran.us.r-project.org")'
Rscript -e 'devtools::install_github("EvoLandEco/eveGNN@multimodal-stacking-boosting")'
Rscript -e 'devtools::install_github("HHildenbrandt/evesim@tianjian")'
Rscript "\$sim_script" "\$name" "\$sim_config" "\$task_type"
EOF_SIM
)
sim_job_id=${sim_job_submit%%;*}
echo "Simulation job: $sim_job_id"

echo "Submitting training job with dependency afterok:$sim_job_id"
train_job_submit=$(sbatch --parsable --dependency=afterok:${sim_job_id} --chdir="$bash_dir" <<EOF_TRAIN
#!/bin/bash
set -e
#SBATCH --time=2-22:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=gnn_eve_pars_est_tree_transformer
#SBATCH --output=logs/gnn_eve_pars_est_tree_transformer-%j.log
#SBATCH --mem=64GB
#SBATCH --partition=gpu

name="$name"
task_type="$task_type"
run_id="$run_id"
train_script="$train_script"
train_config="$train_config"

ml Python/3.8.16-GCCcore-11.2.0
source \$HOME/venvs/eve/bin/activate
python "\$train_script" "\$name" "\$task_type" "\$run_id" --config "\$train_config"
deactivate
EOF_TRAIN
)
train_job_id=${train_job_submit%%;*}
echo "Training job: $train_job_id"

wait_job_gone "$sim_job_id"
sim_state=$(job_state "$sim_job_id")
if [ "$sim_state" != "COMPLETED" ]; then
    echo "Simulation job $sim_job_id ended with state: $sim_state"
    scancel "$train_job_id" >/dev/null 2>&1 || true
    exit 1
fi

wait_job_gone "$train_job_id"
train_state=$(job_state "$train_job_id")
if [ "$train_state" != "COMPLETED" ]; then
    echo "Training job $train_job_id ended with state: $train_state"
    exit 1
fi

echo "Done: $name/$task_type/STBO"