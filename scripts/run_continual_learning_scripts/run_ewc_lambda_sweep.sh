#!/usr/bin/env bash
# =============================================================================
# EWC λ hyperparameter sweep on LIBERO-Goal.
#
# Launches one training run per λ value, each using the production EWC YAML
# (qwengr00t_cl_lora_ewc_libero.yaml).  Runs are serial by default — pass
# PARALLEL=1 to launch pairs concurrently on two GPU-pairs.
#
# Usage:
#   # Sequential on GPUs 3,4:
#   bash scripts/run_continual_learning_scripts/run_ewc_lambda_sweep.sh
#
#   # Parallel pairs on GPUs 3,4 + 5,6:
#   PARALLEL=1 bash scripts/run_continual_learning_scripts/run_ewc_lambda_sweep.sh
#
#   # Custom sweep values / step budget:
#   LAMBDAS="1e4 1e5" STEPS_PER_TASK=2000 bash .../run_ewc_lambda_sweep.sh
#
# Environment variables:
#   LAMBDAS           Space-separated λ values. Default: "1e3 1e4 1e5 1e6".
#   STEPS_PER_TASK    Steps per task. Default: 10000.
#   FISHER_BATCHES    Minibatches for Fisher estimation. Default: 100.
#   GPUS_A            GPU pair for worker A. Default: "3,4".
#   GPUS_B            GPU pair for worker B (parallel mode). Default: "5,6".
#   PARALLEL          0 or 1. Default: 0 (sequential).
#   LOG_DIR           Where to write per-run stdout logs. Default:
#                     /tmp/alphabrain_ewc_sweep.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

LAMBDAS="${LAMBDAS:-1e3 1e4 1e5 1e6}"
STEPS_PER_TASK="${STEPS_PER_TASK:-10000}"
FISHER_BATCHES="${FISHER_BATCHES:-100}"
GPUS_A="${GPUS_A:-3,4}"
GPUS_B="${GPUS_B:-5,6}"
PARALLEL="${PARALLEL:-0}"
LOG_DIR="${LOG_DIR:-/tmp/alphabrain_ewc_sweep}"

YAML="configs/continual_learning/qwengr00t_cl_lora_ewc_libero.yaml"

mkdir -p "$LOG_DIR"

# ---------- helpers ----------
launch_one() {
    # $1=lambda  $2=gpus  $3=logfile
    local LAM="$1"
    local GPUS="$2"
    local LOGFILE="$3"
    # Normalise run_id (1e4 → 1e04 to sort lex-ascending)
    local RUN_ID="ewc_lambda_${LAM}_stepsPerTask_${STEPS_PER_TASK}"
    echo "  [launch] λ=${LAM} gpus=${GPUS} run_id=${RUN_ID}"
    echo "           → log=${LOGFILE}"
    bash scripts/run_continual_learning_scripts/run_cl_train.sh \
        --yaml "$YAML" \
        --run-id "$RUN_ID" \
        --gpus "$GPUS" -- \
        --continual_learning.algorithm.lambda=${LAM} \
        --continual_learning.steps_per_task=${STEPS_PER_TASK} \
        --continual_learning.algorithm.fisher_num_batches=${FISHER_BATCHES} \
        > "$LOGFILE" 2>&1
}

banner() {
    printf '\n%s\n' "================================================================"
    printf '%s\n' "  EWC λ sweep"
    printf '%s\n' "================================================================"
    printf '  %-16s %s\n' "Lambdas"        "${LAMBDAS}"
    printf '  %-16s %s\n' "Steps/task"     "${STEPS_PER_TASK}"
    printf '  %-16s %s\n' "Fisher batches" "${FISHER_BATCHES}"
    printf '  %-16s %s\n' "Mode"           "$([[ $PARALLEL = 1 ]] && echo "parallel pairs" || echo "sequential")"
    printf '  %-16s %s\n' "GPUs A"         "${GPUS_A}"
    [[ $PARALLEL = 1 ]] && printf '  %-16s %s\n' "GPUs B" "${GPUS_B}"
    printf '  %-16s %s\n' "Logs"           "${LOG_DIR}"
    printf '%s\n\n' "================================================================"
}

banner

# ---------- main sweep ----------
if [[ "$PARALLEL" = "1" ]]; then
    # Pair up λs, launch two in parallel per round, wait, then next pair.
    LAM_ARR=($LAMBDAS)
    N=${#LAM_ARR[@]}
    i=0
    while (( i < N )); do
        LAM1="${LAM_ARR[$i]}"
        LOG1="${LOG_DIR}/ewc_lambda_${LAM1}.log"
        if (( i+1 < N )); then
            LAM2="${LAM_ARR[$((i+1))]}"
            LOG2="${LOG_DIR}/ewc_lambda_${LAM2}.log"
            echo ">>> Round $((i/2+1)): launching λ=${LAM1} on ${GPUS_A}, λ=${LAM2} on ${GPUS_B}"
            launch_one "$LAM1" "$GPUS_A" "$LOG1" &
            PID1=$!
            launch_one "$LAM2" "$GPUS_B" "$LOG2" &
            PID2=$!
            wait $PID1 $PID2
            i=$((i+2))
        else
            echo ">>> Final odd λ=${LAM1} on ${GPUS_A}"
            launch_one "$LAM1" "$GPUS_A" "$LOG1"
            i=$((i+1))
        fi
    done
else
    for LAM in $LAMBDAS; do
        LOGFILE="${LOG_DIR}/ewc_lambda_${LAM}.log"
        echo ">>> Sequential: launching λ=${LAM} on ${GPUS_A}"
        launch_one "$LAM" "$GPUS_A" "$LOGFILE"
    done
fi

echo
echo "================================================================"
echo "  EWC λ sweep — all runs complete"
echo "================================================================"
echo "Checkpoints : results/Checkpoints/ewc_lambda_*_stepsPerTask_${STEPS_PER_TASK}/"
echo "Logs        : ${LOG_DIR}/"
echo
echo "Next: evaluate each run via run_cl_eval.sh, then compare Avg SR / NBT."
