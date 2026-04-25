#!/usr/bin/env bash
# =============================================================================
# Continual Learning Matrix Evaluation — one-command, self-contained.
#
# Evaluates every task_*_... checkpoint in a CL run directory on all 10 LIBERO
# tasks, building the 10×10 success-rate matrix. Auto-detects LoRA (runs
# adapter+action_model merge first). Parallelizes across --gpus.
#
# Usage (from repo root):
#   # Full-param run (no LoRA merge needed)
#   bash scripts/run_continual_learning_scripts/run_cl_eval.sh \
#       --run-id neurovla_er_libero_goal_v1 --gpus 0
#
#   # LoRA run (supply --base-config)
#   bash scripts/run_continual_learning_scripts/run_cl_eval.sh \
#       --run-id alphabrain_er_lora_libero_goal_v1 \
#       --base-config configs/continual_learning/qwengr00t_er_lora_libero.yaml \
#       --gpus 0,1
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

# ---------- defaults ----------
RUN_ID=""
BASE_CONFIG=""
GPUS="0"
SUITE="libero_goal"
TRIALS=10
PORT_BASE=5694
OUTPUT_BASE=""
LAST_ONLY=0
EXTRA=()

# ---------- parse CLI ----------
usage() {
    cat <<EOF
Usage: bash $0 --run-id <RUN_ID> [options]

Required:
  --run-id ID           Run directory name under results/Checkpoints/

Common:
  --base-config PATH    LoRA merge base config (required when ckpts contain *_lora_adapter)
                        e.g. configs/continual_learning/qwengr00t_er_lora_libero.yaml
  --gpus LIST           Comma-separated GPU list (default "0"; "0,1" parallel, "1,2,3" etc.)
  --suite NAME          libero_goal|libero_spatial|libero_object|libero_10 (default libero_goal)
  --trials N            Trials per task (default 10)
  --port-base N         Starting port (default 5694)
  --output-base PATH    Eval results root (default results/eval_cl/<RUN_ID>)
  --last-only           Only evaluate the final task_* checkpoint (quick sanity)
  --                    Pass-through args (appended to server_policy.py)
  -h, --help            Show this help

Typical modes (run-id → base-config mapping for LoRA runs):
  RUN_ID                                      | BASE_CONFIG
  alphabrain_er_lora_libero_goal_v1 (5d)      | configs/continual_learning/qwengr00t_er_lora_libero.yaml
  neurovla_er_lora_libero_goal_5k (5h)                 | configs/continual_learning/neurovla_er_lora_libero.yaml
  llamaoft_er_lora_libero_goal_5k (5l)     | configs/continual_learning/llamaoft_er_lora_libero.yaml
  neurovla_er_libero_goal_v1 (5f, full-param) | (omit --base-config — no LoRA merge)

Examples:
  # 5d full matrix, 2 GPU
  bash $0 --run-id alphabrain_er_lora_libero_goal_v1 \\
          --base-config configs/continual_learning/qwengr00t_er_lora_libero.yaml \\
          --gpus 0,1

  # 5f (full-param) single GPU
  bash $0 --run-id neurovla_er_libero_goal_v1 --gpus 1

  # Custom trial count
  bash $0 --run-id alphabrain_er_lora_libero_goal_v1 \\
          --base-config configs/continual_learning/qwengr00t_er_lora_libero.yaml \\
          --gpus 0,1 --trials 50
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)        RUN_ID="$2"; shift 2 ;;
        --base-config)   BASE_CONFIG="$2"; shift 2 ;;
        --gpus)          GPUS="$2"; shift 2 ;;
        --suite)         SUITE="$2"; shift 2 ;;
        --trials)        TRIALS="$2"; shift 2 ;;
        --port-base)     PORT_BASE="$2"; shift 2 ;;
        --output-base)   OUTPUT_BASE="$2"; shift 2 ;;
        --last-only)     LAST_ONLY=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        --)              shift; EXTRA=("$@"); break ;;
        *)               echo "[error] Unknown arg: $1"; usage; exit 1 ;;
    esac
done

[ -n "$RUN_ID" ] || { echo "[error] --run-id is required"; usage; exit 1; }

CKPT_DIR="results/Checkpoints/${RUN_ID}"
CKPT_SUB="$CKPT_DIR/checkpoints"

# ---------- sanity-check: ckpt dir must exist AND contain task_*_(lora_adapter|pytorch_model.pt) ----------
list_available_runs() {
    echo ""
    echo "Available CL run directories with checkpoints:"
    local found=0
    for d in results/Checkpoints/*/checkpoints; do
        [ -d "$d" ] || continue
        local run="$(basename "$(dirname "$d")")"
        local n_lora=0 n_pt=0
        # use shopt nullglob-equivalent via command substitution
        if compgen -G "$d/task_*_lora_adapter" > /dev/null; then
            n_lora=$(ls -d "$d"/task_*_lora_adapter 2>/dev/null | wc -l)
        fi
        if compgen -G "$d/task_*_pytorch_model.pt" > /dev/null; then
            n_pt=$(ls "$d"/task_*_pytorch_model.pt 2>/dev/null | wc -l)
        fi
        if [ "$n_lora" -gt 0 ] || [ "$n_pt" -gt 0 ]; then
            found=$((found+1))
            local kind base_yaml
            if [ "$n_lora" -gt 0 ]; then
                kind="LoRA  (need --base-config)"
                # try to guess base config from run_id prefix
                case "$run" in
                    *qwen*|alphabrain*)         base_yaml="configs/continual_learning/qwengr00t_er_lora_libero.yaml" ;;
                    *neurovla*lora*|*neurovla_cl_lora*)          base_yaml="configs/continual_learning/neurovla_er_lora_libero.yaml" ;;
                    *llama*lora*|*llamaoft_cl_lora*)             base_yaml="configs/continual_learning/llamaoft_er_lora_libero.yaml" ;;
                    *) base_yaml="<pick matching configs/continual_learning/*.yaml>" ;;
                esac
                printf "  %-50s %-25s %2d ckpts   --base-config %s\n" "$run" "$kind" "$n_lora" "$base_yaml"
            else
                kind="Full-param"
                printf "  %-50s %-25s %2d ckpts\n" "$run" "$kind" "$n_pt"
            fi
        fi
    done
    [ "$found" -eq 0 ] && echo "  (none — train one first via run_cl_train.sh)"
}

if [ ! -d "$CKPT_SUB" ]; then
    echo "[error] $CKPT_SUB does not exist"
    list_available_runs
    exit 1
fi
if [ -z "$(ls -A "$CKPT_SUB" 2>/dev/null)" ]; then
    echo "[error] $CKPT_SUB exists but is EMPTY"
    echo "  Likely the train run for '$RUN_ID' aborted early before any ckpt was written."
    list_available_runs
    exit 1
fi
if ! compgen -G "$CKPT_SUB/task_*_lora_adapter" > /dev/null \
   && ! compgen -G "$CKPT_SUB/task_*_pytorch_model.pt" > /dev/null; then
    echo "[error] $CKPT_SUB has files but no expected pattern (task_*_lora_adapter/ or task_*_pytorch_model.pt)"
    echo "  Found instead:"; ls -1 "$CKPT_SUB" | head -5 | sed 's/^/    /'
    list_available_runs
    exit 1
fi

# ---------- load .env ----------
if [ -f "$REPO_ROOT/.env" ]; then
    set -a; # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"; set +a
fi

# ---------- pick Python interpreters ----------
# SERVER_PYTHON — alphabrain env (has torch + flash-attn, NO robosuite)
# EVAL_PYTHON   — robosuite env (has robosuite, MuJoCo, LIBERO client deps)
#
# .env's LIBERO_PYTHON is often misconfigured (points to alphabrain which lacks
# robosuite). We probe LIBERO_PYTHON for robosuite; if it's missing we fall
# back to the conventional vlacl_engine_eval env.
SERVER_PYTHON="${SERVER_PYTHON:-${ALPHABRAIN_PYTHON:-python}}"
EVAL_PYTHON="${EVAL_PYTHON:-${LIBERO_PYTHON:-python}}"

_has_mods() {
    # usage: _has_mods <python> <mod1> <mod2> ...
    local py="$1"; shift
    "$py" -c "$(printf 'import %s\n' "$@")" >/dev/null 2>&1
}

# SERVER_PYTHON needs torch (VLM inference) AND websockets (policy server IPC).
# Common failure: user's shell python is /opt/conda/bin/python (base env) which
# has torch but no websockets — passes a torch-only check but crashes at server
# startup. So we probe both.
if ! _has_mods "$SERVER_PYTHON" torch websockets; then
    FALLBACK="${ALPHABRAIN_PYTHON_FALLBACK:-/path/to/envs/alphabrain/bin/python}"
    if _has_mods "$FALLBACK" torch websockets; then
        echo "[warn] SERVER_PYTHON='$SERVER_PYTHON' missing torch/websockets → falling back to $FALLBACK"
        SERVER_PYTHON="$FALLBACK"
    else
        echo "[error] torch+websockets not found in '$SERVER_PYTHON' nor fallback '$FALLBACK'."
        echo "        Set ALPHABRAIN_PYTHON in .env, or 'conda activate alphabrain' first."
        exit 1
    fi
fi

# EVAL_PYTHON must have robosuite (LIBERO client)
if ! _has_mods "$EVAL_PYTHON" robosuite; then
    FALLBACK="${EVAL_PYTHON_FALLBACK:-/path/to/envs/vlacl_engine_eval/bin/python}"
    if _has_mods "$FALLBACK" robosuite; then
        echo "[warn] EVAL_PYTHON='$EVAL_PYTHON' lacks robosuite → falling back to $FALLBACK"
        EVAL_PYTHON="$FALLBACK"
    else
        echo "[error] No robosuite found in EVAL_PYTHON='$EVAL_PYTHON' nor fallback '$FALLBACK'."
        echo "        Set LIBERO_PYTHON in .env to an env with robosuite (e.g. vlacl_engine_eval)."
        exit 1
    fi
fi

# ---------- env for LIBERO client ----------
: "${LIBERO_HOME:=../LIBERO}"
export LIBERO_HOME
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# ---------- resolve checkpoints + LoRA detection ----------
IS_LORA=false
CHECKPOINTS=()
if compgen -G "$CKPT_SUB/task_*_lora_adapter" > /dev/null; then
    IS_LORA=true
    [ -n "$BASE_CONFIG" ] || { echo "[error] LoRA detected; --base-config is required"; exit 1; }
    while IFS= read -r p; do
        name=$(basename "$p")
        CHECKPOINTS+=("${name%_lora_adapter}")
    done < <(find "$CKPT_SUB" -maxdepth 1 -type d -name 'task_*_lora_adapter' | sort -V)
else
    while IFS= read -r p; do
        name=$(basename "$p")
        CHECKPOINTS+=("${name%_pytorch_model.pt}")
    done < <(find "$CKPT_SUB" -maxdepth 1 -type f -name 'task_*_pytorch_model.pt' ! -name '*_merged*' | sort -V)
fi
[ "${#CHECKPOINTS[@]}" -gt 0 ] || { echo "[error] No valid checkpoints under $CKPT_SUB"; exit 1; }

if [ "$LAST_ONLY" = "1" ]; then
    CHECKPOINTS=("${CHECKPOINTS[-1]}")
    echo "[info] --last-only: keeping only ${CHECKPOINTS[0]}"
fi

[ -n "$OUTPUT_BASE" ] || OUTPUT_BASE="$REPO_ROOT/results/eval_cl/${RUN_ID}"

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPU="${#GPU_ARRAY[@]}"

if [ -t 1 ]; then
    C0=$'\033[0m';       CB=$'\033[1m';       CD=$'\033[2m'
    CC=$'\033[38;5;51m';                     # bright cyan (rule)
    CT=$'\033[1;38;5;45m';                   # bold sky (title)
    CK=$'\033[38;5;244m';                    # gray (label)
    CV=$'\033[38;5;228m';                    # light yellow (path/value)
    CH=$'\033[1;38;5;214m';                  # bold orange (identifier)
    CG=$'\033[38;5;120m';                    # green (count/GPUs)
    CM=$'\033[38;5;213m';                    # magenta (framework/env)
    CY=$'\033[1;38;5;220m';                  # bold yellow (LoRA=true)
    COK=$'\033[1;38;5;46m';                  # bold green (success)
    CWARN=$'\033[1;38;5;214m';               # bold orange (warning)
    CERR=$'\033[1;38;5;196m';                # bold red    (error)
else
    C0=""; CB=""; CD=""; CC=""; CT=""; CK=""; CV=""; CH=""; CG=""; CM=""; CY=""; COK=""; CWARN=""; CERR=""
fi
_rule() { printf "${CC}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C0}\n"; }
_kv()   { printf "  ${CK}%-11s${C0} ${CD}│${C0}  %s\n" "$1" "$2"; }

if $IS_LORA; then LORA_STR="${CY}true${C0}  ${CD}(needs merge)${C0}"; else LORA_STR="${CD}false (full-param)${C0}"; fi

echo
_rule
printf "  ${CT}▶  Continual Learning Matrix Evaluation${C0}\n"
_rule
_kv "RunID"       "${CH}${RUN_ID}${C0}"
_kv "LoRA"        "${LORA_STR}"
_kv "Checkpoints" "${CG}${#CHECKPOINTS[@]}${C0}"
_kv "GPUs"        "${CG}${GPUS}${C0}  ${CD}(${NUM_GPU} parallel, port-base ${PORT_BASE})${C0}"
_kv "Suite"       "${CV}${SUITE}${C0}  ${CD}(${TRIALS} trials/task)${C0}"
_kv "Server py"   "${CM}${SERVER_PYTHON}${C0}"
_kv "Eval py"     "${CM}${EVAL_PYTHON}${C0}"
_kv "Output"      "${CV}${OUTPUT_BASE#$REPO_ROOT/}${C0}"
_kv "Start"       "${CD}$(date)${C0}"
_rule
echo

# ---------- Step 1 (LoRA only): merge adapter + action_model → *_merged.pt ----
merge_single() {
    local name=$1 gpu=$2
    local merged="$CKPT_SUB/${name}_merged.pt"
    local ts; ts="${CD}[$(date '+%H:%M:%S')]${C0}"
    if [ -f "$merged" ]; then
        printf "%b ${CD}SKIP merge (exists):${C0} ${CV}%s${C0}\n" "$ts" "$name"; return 0
    fi
    printf "%b ${CG}GPU%s${C0} ${CM}merging${C0}: ${CV}%s${C0}\n" "$ts" "$gpu" "$name"
    CUDA_VISIBLE_DEVICES=$gpu "$SERVER_PYTHON" -m \
        AlphaBrain.training.trainer_utils.peft.merge_lora_checkpoint \
        --base_config "$BASE_CONFIG" \
        --lora_adapter_dir "$CKPT_SUB/${name}_lora_adapter" \
        --action_model_pt  "$CKPT_SUB/${name}_action_model.pt" \
        --output_path      "$merged"
}

run_parallel() {
    local func=$1; shift
    local items=("$@")
    local pids=()
    for i in "${!items[@]}"; do
        local slot=$(( i % NUM_GPU ))
        local gpu="${GPU_ARRAY[$slot]}"
        local port=$(( PORT_BASE + slot ))
        "$func" "${items[$i]}" "$gpu" "$port" &
        pids+=($!)
        if [ $(( (i + 1) % NUM_GPU )) -eq 0 ]; then
            for pid in "${pids[@]}"; do wait "$pid"; done
            pids=()
        fi
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
}

if $IS_LORA; then
    printf "\n${CB}${CC}[Step 1]${C0} ${CT}Merge LoRA checkpoints${C0}\n"
    run_parallel merge_single "${CHECKPOINTS[@]}"
    printf "${CD}[$(date '+%H:%M:%S')]${C0} ${COK}✓${C0} All merges complete.\n\n"
fi

# ---------- Step 2: per-ckpt server + client ----------
run_eval_single() {
    local name=$1 gpu=$2 port=$3
    local out_dir="$OUTPUT_BASE/$name"
    local suffix; if $IS_LORA; then suffix="_merged.pt"; else suffix="_pytorch_model.pt"; fi
    local ckpt="$CKPT_SUB/${name}${suffix}"
    mkdir -p "$out_dir/videos"

    # one-line-per-event log helper: [HH:MM:SS] GPU<n> [step] <body>
    local ts_prefix="${CD}[$(date '+%H:%M:%S')]${C0} ${CG}GPU${gpu}${C0}"

    # Pre-flight: refuse to start if port is already in use. Otherwise our server
    # would fail silently to bind while the wrapper's TCP probe connects to the
    # stale occupant (→ fake "ready in 0s" + results from the wrong ckpt).
    if "$SERVER_PYTHON" -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', $port)); s.close()" 2>/dev/null; then
        printf "${CD}[$(date '+%H:%M:%S')]${C0} ${CERR}[error]${C0} ${CG}GPU%s${C0} port ${CH}%s${C0} ${CERR}already in use${C0} — kill the stale server or pick a new --port-base.\n" "$gpu" "$port"
        return 1
    fi

    printf "%b ${CC}[1/3]${C0} launching server: ${CV}%s${C0} ${CD}(port=%s)${C0}\n" "$ts_prefix" "$name" "$port"
    CUDA_VISIBLE_DEVICES=$gpu "$SERVER_PYTHON" \
        "$REPO_ROOT/deployment/model_server/server_policy.py" \
        --ckpt_path "$ckpt" --port "$port" --use_bf16 "${EXTRA[@]}" \
        > "$out_dir/server.log" 2>&1 &
    local server_pid=$!

    local waited=0
    while ! "$SERVER_PYTHON" -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', $port)); s.close()" 2>/dev/null; do
        if ! kill -0 $server_pid 2>/dev/null; then
            printf "${CD}[$(date '+%H:%M:%S')]${C0} ${CERR}[error]${C0} ${CG}GPU%s${C0} server ${CERR}died${C0}: ${CV}%s${C0}\n" "$gpu" "$name"
            tail -30 "$out_dir/server.log"; return 1
        fi
        if [ $waited -ge 600 ]; then
            printf "${CD}[$(date '+%H:%M:%S')]${C0} ${CERR}[error]${C0} ${CG}GPU%s${C0} server ${CERR}timeout (600s)${C0}: ${CV}%s${C0}\n" "$gpu" "$name"
            kill $server_pid 2>/dev/null; wait $server_pid 2>/dev/null || true; return 1
        fi
        sleep 10; waited=$((waited+10))
        if [ $((waited % 30)) -eq 0 ]; then
            local t2; t2="${CD}[$(date '+%H:%M:%S')]${C0} ${CG}GPU${gpu}${C0}"
            printf "%b ${CWARN}⋯${C0} still loading server ${CD}(%ss elapsed)${C0}: ${CV}%s${C0}\n" "$t2" "$waited" "$name"
        fi
    done
    local t3; t3="${CD}[$(date '+%H:%M:%S')]${C0} ${CG}GPU${gpu}${C0}"
    printf "%b ${COK}✓${C0} server ready in ${COK}%ss${C0}: ${CV}%s${C0}\n" "$t3" "$waited" "$name"

    local t4; t4="${CD}[$(date '+%H:%M:%S')]${C0} ${CG}GPU${gpu}${C0}"
    printf "%b ${CC}[2/3]${C0} running client: ${CV}%s${C0}\n" "$t4" "$name"
    # Full log → eval.log via tee; filter key lines (Task / Success / per-task SR /
    # total SR / Traceback) with a [GPU<n>] prefix to terminal so the user can
    # watch progress instead of staring at a silent shell for 20+ minutes.
    local filter_awk='
        /Task: / || /Starting episode/ || /Success: / ||
        /Current task success rate/ || /Current total success rate/ ||
        /Total success rate/ || /Traceback/ {
            printf "%s[GPU%s]%s %s\n", cg, gpu, c0, $0
            fflush()
        }'
    "$EVAL_PYTHON" "$REPO_ROOT/benchmarks/LIBERO/eval/eval_libero.py" \
        --args.pretrained-path "$ckpt" \
        --args.host 127.0.0.1 --args.port "$port" \
        --args.task-suite-name "$SUITE" \
        --args.num-trials-per-task "$TRIALS" \
        --args.video-out-path "$out_dir/videos" \
        2>&1 \
        | tee "$out_dir/eval.log" \
        | awk -v gpu="$gpu" -v cg="$CG" -v c0="$C0" "$filter_awk"

    local t5; t5="${CD}[$(date '+%H:%M:%S')]${C0} ${CG}GPU${gpu}${C0}"
    printf "%b ${CC}[3/3]${C0} killing server: ${CV}%s${C0}\n" "$t5" "$name"
    kill $server_pid 2>/dev/null; wait $server_pid 2>/dev/null || true

    # Robosuite/MuJoCo sometimes emits NULs to stdout, tagging the log as binary.
    # -a forces text mode; sed strips ANSI codes added by the client-side beautify.
    local total pct col
    # `head -N` closes the pipe early → upstream `grep -oE` can get SIGPIPE and
    # exit non-zero, which with `set -euo pipefail` silently kills the whole
    # script. `awk 'NR==1'` reads all input and still only prints line 1.
    total=$(grep -a "Total success rate" "$out_dir/eval.log" \
            | tail -1 \
            | sed 's/\x1b\[[0-9;]*m//g' \
            | grep -oE '[0-9]+(\.[0-9]+)?' | awk 'NR==1')
    # color SR: green ≥0.45, yellow 0.25–0.45, red <0.25, dim '?' if missing
    if [ -n "$total" ]; then
        pct=$(awk -v t="$total" 'BEGIN{printf "%.1f", t*100}')
        if   awk -v t="$total" 'BEGIN{exit !(t>=0.45)}'; then col="$COK"
        elif awk -v t="$total" 'BEGIN{exit !(t>=0.25)}'; then col="$CWARN"
        else                                                   col="$CERR"; fi
        local t6; t6="${CD}[$(date '+%H:%M:%S')]${C0} ${CG}GPU${gpu}${C0}"
        printf "%b ${COK}✓ done${C0}: ${CV}%s${C0}  ${CD}→${C0}  ${CB}Total SR: ${col}%s%% (%s)${C0}\n" "$t6" "$name" "$pct" "$total"
    else
        local t6; t6="${CD}[$(date '+%H:%M:%S')]${C0} ${CG}GPU${gpu}${C0}"
        printf "%b ${CWARN}⚠ done${C0}: ${CV}%s${C0}  ${CD}→${C0}  ${CWARN}Total SR: ?${C0} ${CD}(no line matched)${C0}\n" "$t6" "$name"
    fi
}

printf "\n${CB}${CC}[Step 2]${C0} ${CT}Matrix evaluation${C0} ${CD}(%d checkpoints)${C0}\n\n" "${#CHECKPOINTS[@]}"
run_parallel run_eval_single "${CHECKPOINTS[@]}"

echo
_rule
printf "  ${COK}✓  All evaluations complete${C0}  ${CD}$(date)${C0}\n"
_rule
_kv "Results" "${CV}${OUTPUT_BASE#$REPO_ROOT/}/${C0}"
_rule
echo
