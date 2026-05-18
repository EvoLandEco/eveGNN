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

sim_script=${SIM_SCRIPT:-$project_root/Script/eve_pars_est_bd_ed_nnd_data.R}
sim_config=${SIM_CONFIG:-$project_root/Config/eve_sim.yaml}
train_script=${TRAIN_SCRIPT:-$project_root/Script/train_eve_pars_est_TreeTransformer.py}
train_config=${TRAIN_CONFIG:-$project_root/Config/eve_train_tree_transformer.yaml}
r_lib_user=${R_LIBS_USER:-$HOME/R/eve_tt_r_libs}

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

sim_submit=$(sbatch --parsable --chdir="$bash_dir" <<EOF_SIM
#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=gnn_eve_pars_free
#SBATCH --output=logs/gnn_eve_pars_free-%j.log
#SBATCH --mem=48GB
#SBATCH --partition=regular

set -euo pipefail

name="$name"
task_type="$task_type"
sim_script="$sim_script"
sim_config="$sim_config"
export R_LIBS_USER="$r_lib_user"
mkdir -p "\$R_LIBS_USER"

ml R
Rscript -e '.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths())); pkgs <- c("devtools", "yaml", "ape", "RcppParallel"); miss <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]; if (length(miss)) install.packages(miss, repos = "http://cran.us.r-project.org", lib = Sys.getenv("R_LIBS_USER"))'
Rscript -e '.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths())); devtools::install_github("EvoLandEco/eveGNN@multimodal-stacking-boosting", lib = Sys.getenv("R_LIBS_USER"), upgrade = "never")'
Rscript -e '.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths())); devtools::install_github("HHildenbrandt/evesim@tianjian", lib = Sys.getenv("R_LIBS_USER"), upgrade = "never")'
Rscript "\$sim_script" "\$name" "\$sim_config" "\$task_type"
EOF_SIM
)
sim_job_id=${sim_submit%%;*}

train_submit=$(sbatch --parsable --dependency=afterok:${sim_job_id} --chdir="$bash_dir" <<EOF_TRAIN
#!/bin/bash
#SBATCH --time=2-22:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=gnn_eve_pars_est_tree_transformer
#SBATCH --output=logs/gnn_eve_pars_est_tree_transformer-%j.log
#SBATCH --mem=64GB
#SBATCH --partition=gpu

set -euo pipefail

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
train_job_id=${train_submit%%;*}

echo "simulation_job_id=$sim_job_id"
echo "training_job_id=$train_job_id"
echo "training_dependency=afterok:$sim_job_id"
echo "simulation_log=$bash_dir/logs/gnn_eve_pars_free-${sim_job_id}.log"
echo "training_log=$bash_dir/logs/gnn_eve_pars_est_tree_transformer-${train_job_id}.log"
echo "output=$name/$task_type/STBO"
