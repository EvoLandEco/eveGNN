#!/usr/bin/env bash
# All-in-one SLURM launcher for the eve BD/ED/NND TreeTransformer workflow.
# Version: 2026-05-18-hotfix3
# Fixes:
#   - preserves project paths across SLURM spool execution;
#   - installs/checks common CRAN R dependencies when INSTALL_R_PKGS=1;
#   - exports R_LIBS_USER consistently;
#   - passes task_type to the simulation script;
#   - supports EVE_PARALLEL_BACKEND=mclapply|future|serial;
#   - fixes R preflight getRversion() printing for package_version objects.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash run_eve_tree_transformer_all_in_one_slurm.sh <name> [task_type=EVE_FREE_TES] [run_id=1]

Examples:
  bash run_eve_tree_transformer_all_in_one_slurm.sh eve_tt_run
  INSTALL_R_PKGS=1 bash run_eve_tree_transformer_all_in_one_slurm.sh eve_tt_run
  EVE_PARALLEL_BACKEND=serial SKIP_TRAIN=1 bash run_eve_tree_transformer_all_in_one_slurm.sh debug_run

Outputs are expected under:
  <name>/<task_type>/STBO/
USAGE
}

abs_path_of_this_script() {
  local src="${BASH_SOURCE[0]}"
  while [[ -L "$src" ]]; do
    local dir
    dir="$(cd -P "$(dirname "$src")" >/dev/null 2>&1 && pwd)"
    src="$(readlink "$src")"
    [[ "$src" != /* ]] && src="$dir/$src"
  done
  local dir
  dir="$(cd -P "$(dirname "$src")" >/dev/null 2>&1 && pwd)"
  printf '%s/%s\n' "$dir" "$(basename "$src")"
}

load_hpc_module() {
  local module_name="${1:-}"
  [[ -z "$module_name" ]] && return 0
  if command -v ml >/dev/null 2>&1; then
    ml "$module_name"
  elif command -v module >/dev/null 2>&1; then
    module load "$module_name"
  else
    echo "No 'ml' or 'module' command found; assuming required software is already available." >&2
  fi
}

safe_job_component() {
  local value="${1:-run}"
  value="$(basename "$value")"
  value="$(printf '%s' "$value" | tr -c '[:alnum:]_-' '_' | cut -c1-48)"
  [[ -z "$value" ]] && value="run"
  printf '%s' "$value"
}

sacct_state() {
  local job_id="$1"
  if ! command -v sacct >/dev/null 2>&1; then
    return 1
  fi
  sacct -X -n -j "$job_id" -o State%40 2>/dev/null | awk 'NF {print $1; exit}'
}

wait_for_slurm_job() {
  local job_id="$1"
  local label="$2"
  local poll_interval="${POLL_INTERVAL:-60}"
  local state=""

  echo "Waiting for ${label} job ${job_id} ..."
  while true; do
    if squeue -h -j "$job_id" >/dev/null 2>&1; then
      if [[ -n "$(squeue -h -j "$job_id" 2>/dev/null || true)" ]]; then
        sleep "$poll_interval"
        continue
      fi
    fi

    state="$(sacct_state "$job_id" || true)"
    case "$state" in
      COMPLETED*)
        echo "${label} job ${job_id} completed."
        return 0
        ;;
      FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|BOOT_FAIL*|DEADLINE*|PREEMPTED*|REVOKED*|SPECIAL_EXIT*|DEPENDENCY*)
        echo "${label} job ${job_id} ended with state: ${state}" >&2
        return 1
        ;;
      "")
        sleep "$poll_interval"
        ;;
      *)
        sleep "$poll_interval"
        ;;
    esac
  done
}

print_effective_settings() {
  cat <<SETTINGS
Effective pipeline settings:
  name:                 ${RUN_NAME}
  task_type:            ${TASK_TYPE}
  run_id:               ${RUN_ID}
  bash_dir:             ${BASH_DIR}
  project_root:         ${PROJECT_ROOT}
  sim_script:           ${SIM_SCRIPT}
  sim_config:           ${SIM_CONFIG}
  train_script:         ${TRAIN_SCRIPT}
  train_config:         ${TRAIN_CONFIG}
  log_dir:              ${LOG_DIR}
  wait:                 ${WAIT}
  skip_sim:             ${SKIP_SIM}
  skip_train:           ${SKIP_TRAIN}
  install_r_pkgs:       ${INSTALL_R_PKGS}
  R module:             ${R_MODULE}
  R_LIBS_USER:          ${R_LIBS_USER}
  EVE_PARALLEL_BACKEND: ${EVE_PARALLEL_BACKEND}
  Python module:        ${PY_MODULE}
  Python venv:          ${VENV_PATH}
  simulation job:       time=${SIM_TIME}, cpus=${SIM_CPUS}, mem=${SIM_MEM}, partition=${SIM_PARTITION}
  training job:         time=${TRAIN_TIME}, gpus=${TRAIN_GPUS}, mem=${TRAIN_MEM}, partition=${TRAIN_PARTITION}
SETTINGS
}

install_or_check_r_packages() {
  # Assumes R module has already been loaded and R_LIBS_USER exported.
  mkdir -p "$R_LIBS_USER"

  echo "[simulation] R package library path preflight"
  Rscript - <<'RSCRIPT'
cat("R version:", as.character(getRversion()), "\n")
cat(".libPaths():\n")
cat(paste0("  ", .libPaths(), collapse = "\n"), "\n")
required <- c("yaml", "ape", "RcppParallel")
missing <- required[!vapply(required, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing) > 0) {
  cat("Missing CRAN packages:", paste(missing, collapse = ", "), "\n")
} else {
  cat("Required CRAN packages visible.\n")
}
RSCRIPT

  if [[ "$INSTALL_R_PKGS" == "1" ]]; then
    echo "[simulation] Installing/updating R dependencies because INSTALL_R_PKGS=1"
    Rscript - <<'RSCRIPT'
options(repos = c(CRAN = "https://cloud.r-project.org"))
if (!dir.exists(Sys.getenv("R_LIBS_USER"))) dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE)
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))

cran <- c("yaml", "ape", "RcppParallel", "future", "future.apply", "devtools")
missing <- cran[!vapply(cran, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing) > 0) {
  message("Installing CRAN packages: ", paste(missing, collapse = ", "))
  install.packages(missing, dependencies = TRUE)
}

if (!requireNamespace("devtools", quietly = TRUE)) {
  stop("devtools is still unavailable after attempted installation")
}
devtools::install_github("EvoLandEco/eveGNN@multimodal-stacking-boosting", upgrade = "never")
devtools::install_github("HHildenbrandt/evesim@tianjian", upgrade = "never")
RSCRIPT
  fi
}

run_simulation_worker() {
  local run_name="$1"
  local task_type="$2"
  local run_id="$3"

  cd "$BASH_DIR"
  echo "[simulation] started at $(date) on ${HOSTNAME:-unknown-host}"
  echo "[simulation] SLURM_JOB_ID=${SLURM_JOB_ID:-not-set}"
  echo "[simulation] run_name=${run_name}; task_type=${task_type}; run_id=${run_id}"

  load_hpc_module "$R_MODULE"
  export R_LIBS_USER
  export EVE_PARALLEL_BACKEND
  mkdir -p "$R_LIBS_USER"

  install_or_check_r_packages

  if [[ ! -f "$SIM_SCRIPT" ]]; then
    echo "Simulation script not found: $SIM_SCRIPT" >&2
    exit 2
  fi
  if [[ ! -f "$SIM_CONFIG" ]]; then
    echo "Simulation config not found: $SIM_CONFIG" >&2
    exit 2
  fi

  Rscript "$SIM_SCRIPT" "$run_name" "$SIM_CONFIG" "$task_type"

  local tree_dir="${run_name}/${task_type}/GNN/tree"
  if [[ ! -d "$tree_dir" ]]; then
    echo "Expected exported tree directory not found: $tree_dir" >&2
    echo "Check task_type or SIM_SCRIPT." >&2
    exit 3
  fi

  local n_trees
  n_trees="$(find "$tree_dir" -maxdepth 1 -type f -name 'tree_*.rds' | wc -l | tr -d ' ')"
  echo "[simulation] exported ${n_trees} tree RDS files under ${tree_dir}"
  if [[ "$n_trees" -eq 0 ]]; then
    echo "No exported tree_*.rds files found; stopping before training." >&2
    exit 4
  fi

  echo "[simulation] finished at $(date)"
}

run_training_worker() {
  local run_name="$1"
  local task_type="$2"
  local run_id="$3"

  cd "$BASH_DIR"
  echo "[training] started at $(date) on ${HOSTNAME:-unknown-host}"
  echo "[training] SLURM_JOB_ID=${SLURM_JOB_ID:-not-set}"
  echo "[training] run_name=${run_name}; task_type=${task_type}; run_id=${run_id}"

  load_hpc_module "$PY_MODULE"

  if [[ -n "$VENV_PATH" ]]; then
    if [[ -f "${VENV_PATH}/bin/activate" ]]; then
      # shellcheck source=/dev/null
      source "${VENV_PATH}/bin/activate"
    elif [[ -f "$VENV_PATH" ]]; then
      # shellcheck source=/dev/null
      source "$VENV_PATH"
    else
      echo "Python virtual environment activate script not found at: ${VENV_PATH}/bin/activate" >&2
      exit 5
    fi
  fi

  if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Training script not found: $TRAIN_SCRIPT" >&2
    exit 2
  fi
  if [[ ! -f "$TRAIN_CONFIG" ]]; then
    echo "Training config not found: $TRAIN_CONFIG" >&2
    exit 2
  fi

  local tree_dir="${run_name}/${task_type}/GNN/tree"
  if [[ ! -d "$tree_dir" ]]; then
    echo "Training input directory not found: $tree_dir" >&2
    exit 6
  fi

  export PYTHONUNBUFFERED=1
  local cmd=(python -u "$TRAIN_SCRIPT" "$run_name" "$task_type" "$run_id" --config "$TRAIN_CONFIG")
  if [[ -n "${TRAIN_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<< "$TRAIN_EXTRA_ARGS"
    cmd+=("${extra_args[@]}")
  fi

  echo "[training] command: ${cmd[*]}"
  "${cmd[@]}"

  local out_dir="${run_name}/${task_type}/STBO"
  if [[ ! -d "$out_dir" ]]; then
    echo "Expected output directory not found after training: $out_dir" >&2
    exit 7
  fi

  echo "[training] output directory contents:"
  ls -lah "$out_dir"
  echo "[training] finished at $(date)"
}

# Resolve paths before launcher/worker mode.
DISCOVERED_SELF_PATH="$(abs_path_of_this_script)"
if [[ -n "${EVE_TT_SELF_PATH:-}" && -f "${EVE_TT_SELF_PATH}" ]]; then
  SELF_PATH="${EVE_TT_SELF_PATH}"
else
  SELF_PATH="$DISCOVERED_SELF_PATH"
fi

if [[ -n "${EVE_TT_BASH_DIR:-}" && -d "${EVE_TT_BASH_DIR}" ]]; then
  BASH_DIR="${EVE_TT_BASH_DIR}"
else
  BASH_DIR="$(dirname "$SELF_PATH")"
fi

if [[ -n "${EVE_TT_PROJECT_ROOT:-}" && -d "${EVE_TT_PROJECT_ROOT}" ]]; then
  PROJECT_ROOT="${EVE_TT_PROJECT_ROOT}"
else
  PROJECT_ROOT="$(cd "$BASH_DIR/.." >/dev/null 2>&1 && pwd)"
fi

LOG_DIR="${LOG_DIR:-${EVE_TT_LOG_DIR:-${BASH_DIR}/logs}}"
mkdir -p "$LOG_DIR"

SIM_SCRIPT="${SIM_SCRIPT:-${PROJECT_ROOT}/Script/eve_pars_est_bd_ed_nnd_data.R}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${PROJECT_ROOT}/Script/train_eve_pars_est_TreeTransformer.py}"
SIM_CONFIG="${SIM_CONFIG:-${PROJECT_ROOT}/Config/eve_sim.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${PROJECT_ROOT}/Config/eve_train_tree_transformer.yaml}"

R_MODULE="${R_MODULE:-R}"
PY_MODULE="${PY_MODULE:-Python/3.8.16-GCCcore-11.2.0}"
VENV_PATH="${VENV_PATH:-${HOME}/venvs/eve}"
INSTALL_R_PKGS="${INSTALL_R_PKGS:-0}"
R_LIBS_USER="${R_LIBS_USER:-${HOME}/R/eve_tt_r_libs}"
EVE_PARALLEL_BACKEND="${EVE_PARALLEL_BACKEND:-mclapply}"

WAIT="${WAIT:-1}"
SKIP_SIM="${SKIP_SIM:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

SIM_TIME="${SIM_TIME:-23:59:00}"
SIM_CPUS="${SIM_CPUS:-24}"
SIM_MEM="${SIM_MEM:-64GB}"
SIM_PARTITION="${SIM_PARTITION:-regular}"

TRAIN_TIME="${TRAIN_TIME:-71:59:00}"
TRAIN_GPUS="${TRAIN_GPUS:-1}"
TRAIN_GPU_OPTION="${TRAIN_GPU_OPTION:---gpus-per-node}"
TRAIN_MEM="${TRAIN_MEM:-64GB}"
TRAIN_PARTITION="${TRAIN_PARTITION:-gpu}"

ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
MAIL_USER="${MAIL_USER:-}"
MAIL_TYPE="${MAIL_TYPE:-}"

export EVE_TT_SELF_PATH="$SELF_PATH"
export EVE_TT_BASH_DIR="$BASH_DIR"
export EVE_TT_PROJECT_ROOT="$PROJECT_ROOT"
export EVE_TT_LOG_DIR="$LOG_DIR"
export SIM_SCRIPT TRAIN_SCRIPT SIM_CONFIG TRAIN_CONFIG
export R_MODULE PY_MODULE VENV_PATH INSTALL_R_PKGS R_LIBS_USER EVE_PARALLEL_BACKEND
export WAIT SKIP_SIM SKIP_TRAIN
export SIM_TIME SIM_CPUS SIM_MEM SIM_PARTITION
export TRAIN_TIME TRAIN_GPUS TRAIN_GPU_OPTION TRAIN_MEM TRAIN_PARTITION
export ACCOUNT QOS MAIL_USER MAIL_TYPE

mode="${1:-}"
if [[ "$mode" == "__simulate" ]]; then
  shift
  if [[ "$#" -ne 3 ]]; then
    echo "Internal error: __simulate requires <name> <task_type> <run_id>" >&2
    exit 64
  fi
  RUN_NAME="$1" TASK_TYPE="$2" RUN_ID="$3"
  run_simulation_worker "$RUN_NAME" "$TASK_TYPE" "$RUN_ID"
  exit 0
elif [[ "$mode" == "__train" ]]; then
  shift
  if [[ "$#" -ne 3 ]]; then
    echo "Internal error: __train requires <name> <task_type> <run_id>" >&2
    exit 64
  fi
  RUN_NAME="$1" TASK_TYPE="$2" RUN_ID="$3"
  run_training_worker "$RUN_NAME" "$TASK_TYPE" "$RUN_ID"
  exit 0
fi

if [[ "$#" -lt 1 || "$#" -gt 3 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 64
fi

RUN_NAME="$1"
TASK_TYPE="${2:-EVE_FREE_TES}"
RUN_ID="${3:-1}"

print_effective_settings

if [[ "$SKIP_SIM" != "1" ]]; then
  [[ -f "$SIM_SCRIPT" ]] || { echo "Simulation script not found: $SIM_SCRIPT" >&2; exit 2; }
  [[ -f "$SIM_CONFIG" ]] || { echo "Simulation config not found: $SIM_CONFIG" >&2; exit 2; }
fi
if [[ "$SKIP_TRAIN" != "1" ]]; then
  [[ -f "$TRAIN_SCRIPT" ]] || { echo "Training script not found: $TRAIN_SCRIPT" >&2; exit 2; }
  [[ -f "$TRAIN_CONFIG" ]] || { echo "Training config not found: $TRAIN_CONFIG" >&2; exit 2; }
fi

common_sbatch_args=(--parsable --export=ALL --chdir="$BASH_DIR")
if [[ -n "$ACCOUNT" ]]; then common_sbatch_args+=(--account="$ACCOUNT"); fi
if [[ -n "$QOS" ]]; then common_sbatch_args+=(--qos="$QOS"); fi
if [[ -n "$MAIL_USER" ]]; then common_sbatch_args+=(--mail-user="$MAIL_USER"); fi
if [[ -n "$MAIL_TYPE" ]]; then common_sbatch_args+=(--mail-type="$MAIL_TYPE"); fi

safe_run="$(safe_job_component "$RUN_NAME")"
SIM_JOB_ID=""
TRAIN_JOB_ID=""

if [[ "$SKIP_SIM" != "1" ]]; then
  echo "Submitting simulation/export job ..."
  sim_submit_output="$({
    sbatch "${common_sbatch_args[@]}" \
      --job-name="eve_tt_data_${safe_run}" \
      --output="${LOG_DIR}/%x-%j.log" \
      --time="$SIM_TIME" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="$SIM_CPUS" \
      --mem="$SIM_MEM" \
      --partition="$SIM_PARTITION" \
      "$SELF_PATH" __simulate "$RUN_NAME" "$TASK_TYPE" "$RUN_ID"
  })"
  SIM_JOB_ID="${sim_submit_output%%;*}"
  echo "Simulation job id: ${SIM_JOB_ID}"
else
  echo "SKIP_SIM=1, not submitting simulation job."
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo "Submitting training job ..."
  train_sbatch_args=("${common_sbatch_args[@]}"
    --job-name="eve_tt_train_${safe_run}"
    --output="${LOG_DIR}/%x-%j.log"
    --time="$TRAIN_TIME"
    --nodes=1
    --ntasks=1
    --mem="$TRAIN_MEM"
    --partition="$TRAIN_PARTITION")

  if [[ "$TRAIN_GPUS" != "0" ]]; then
    train_sbatch_args+=("${TRAIN_GPU_OPTION}=${TRAIN_GPUS}")
  fi
  if [[ -n "$SIM_JOB_ID" ]]; then
    train_sbatch_args+=(--dependency="afterok:${SIM_JOB_ID}")
    if [[ "${KILL_INVALID_DEP_OPT:-0}" == "1" ]]; then
      train_sbatch_args+=(--kill-on-invalid-dep=yes)
    fi
  fi

  train_submit_output="$({
    sbatch "${train_sbatch_args[@]}" \
      "$SELF_PATH" __train "$RUN_NAME" "$TASK_TYPE" "$RUN_ID"
  })"
  TRAIN_JOB_ID="${train_submit_output%%;*}"
  echo "Training job id: ${TRAIN_JOB_ID}"
else
  echo "SKIP_TRAIN=1, not submitting training job."
fi

cat <<SUBMITTED
Submitted pipeline:
  simulation job: ${SIM_JOB_ID:-skipped}
  training job:   ${TRAIN_JOB_ID:-skipped}
  logs:           ${LOG_DIR}
SUBMITTED

if [[ "$WAIT" != "1" ]]; then
  echo "WAIT=${WAIT}; returning after submission."
  exit 0
fi

if [[ -n "$SIM_JOB_ID" ]]; then
  if ! wait_for_slurm_job "$SIM_JOB_ID" "simulation"; then
    if [[ -n "$TRAIN_JOB_ID" ]]; then
      echo "Cancelling dependent training job ${TRAIN_JOB_ID}." >&2
      scancel "$TRAIN_JOB_ID" >/dev/null 2>&1 || true
    fi
    echo "Simulation failed. Check log: ${LOG_DIR}/eve_tt_data_${safe_run}-${SIM_JOB_ID}.log" >&2
    exit 10
  fi
fi

if [[ -n "$TRAIN_JOB_ID" ]]; then
  if ! wait_for_slurm_job "$TRAIN_JOB_ID" "training"; then
    echo "Training failed. Check log: ${LOG_DIR}/eve_tt_train_${safe_run}-${TRAIN_JOB_ID}.log" >&2
    exit 11
  fi
fi

OUT_DIR="${RUN_NAME}/${TASK_TYPE}/STBO"
echo "Pipeline finished at $(date)."
echo "Expected output directory: ${OUT_DIR}"
if [[ -d "$OUT_DIR" ]]; then
  echo "Output files:"
  find "$OUT_DIR" -maxdepth 1 -type f | sort
else
  echo "Output directory not found from launcher context. Check training log and path: ${OUT_DIR}" >&2
fi
