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
train_script=${TRAIN_SCRIPT:-$project_root/Script/train_eve_pars_est_TreeTransformer.py}

mkdir -p "$bash_dir/logs"

if [ ! -f "$train_script" ]; then
    echo "Missing training script: $train_script"
    exit 1
fi

if [[ "$name" = /* ]]; then
    data_root="$name"
else
    data_root="$bash_dir/$name"
fi

if [ ! -d "$data_root/$task_type/GNN/tree" ]; then
    echo "Missing data directory: $data_root/$task_type/GNN/tree"
    exit 1
fi
if [ ! -d "$data_root/$task_type/GNN/tree/EL" ]; then
    echo "Missing data directory: $data_root/$task_type/GNN/tree/EL"
    exit 1
fi
if [ ! -d "$data_root/$task_type/GNN/tree/BT" ]; then
    echo "Missing data directory: $data_root/$task_type/GNN/tree/BT"
    exit 1
fi

safe_name=$(printf '%s_%s_%s' "$name" "$task_type" "$run_id" | tr '/ .' '___')
train_config="$bash_dir/logs/eve_train_tree_transformer_old_exports_${safe_name}.yaml"

cat > "$train_config" <<EOF_CFG
alpha: ${ALPHA:-0.5}
beta: ${BETA:-0.5}
huber_delta: ${HUBER_DELTA:-1.0}
n_predicted_values: ${N_PREDICTED_VALUES:-2}
n_classes: 3
class_names: [pd, ed, nnd]
metric_aliases:
  pd: pd
  PD: pd
  bd: pd
  BD: pd
  ed: ed
  ED: ed
  nnd: nnd
  NND: nnd
class_probability_columns: [pd_prob, ed_prob, nnd_prob]
seed: ${SEED:-12345}
train_fraction: ${TRAIN_FRACTION:-0.9}
shuffle_data: true
min_nodes: ${MIN_NODES:-4}
max_nodes_limit: ${MAX_NODES_LIMIT:-2500}
normalize_edge_length: true
target_standardization: true
epoch_number_transformer: ${EPOCH_NUMBER_TRANSFORMER:-51}
train_batch_size: ${TRAIN_BATCH_SIZE:-8}
test_batch_size: ${TEST_BATCH_SIZE:-8}
learning_rate: ${LEARNING_RATE:-0.0005}
weight_decay: ${WEIGHT_DECAY:-0.01}
d_model: ${D_MODEL:-128}
num_heads: ${NUM_HEADS:-4}
num_layers: ${NUM_LAYERS:-4}
dim_feedforward: ${DIM_FEEDFORWARD:-256}
dropout_ratio: ${DROPOUT_RATIO:-0.15}
attention_radius_edges: ${ATTENTION_RADIUS_EDGES:-0}
gradient_clip_norm: ${GRADIENT_CLIP_NORM:-1.0}
num_workers: ${NUM_WORKERS:-0}
early_stopping_patience: ${EARLY_STOPPING_PATIENCE:-0}
output_tag: tree_transformer
write_csv_fallback: true
save_every_epoch_predictions: false
EOF_CFG

job_id=$(sbatch --parsable --chdir="$bash_dir" <<EOF_JOB
#!/bin/bash
#SBATCH --time=${TRAIN_TIME:-2-22:59:00}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=${TRAIN_GPUS:-1}
#SBATCH --job-name=gnn_eve_pars_est_tree_transformer
#SBATCH --output=logs/gnn_eve_pars_est_tree_transformer-%j.log
#SBATCH --mem=${TRAIN_MEM:-64GB}
#SBATCH --partition=${TRAIN_PARTITION:-gpu}

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
EOF_JOB
)

echo "Submitted training job: $job_id"
echo "Data: $data_root/$task_type/GNN"
echo "Log: $bash_dir/logs/gnn_eve_pars_est_tree_transformer-${job_id}.log"
echo "Output: $data_root/$task_type/STBO"
