# Continual Learning

Train a single VLA backbone sequentially over 10 LIBERO tasks with Experience Replay (ER) to mitigate catastrophic forgetting. Supports 4 architectures × LoRA / full-param.

---

## Prerequisites

```bash
conda activate alphabrain
cp .env.example .env
vim .env           # fill in paths below
```

Required env vars: `PRETRAINED_MODELS_DIR`, `LEROBOT_LIBERO_DATA_DIR`, `LIBERO_PYTHON`, `LIBERO_HOME`.

---

## Train

```bash
# Default: QwenGR00T LoRA + ER (~15 h on 2× A800)
bash scripts/run_continual_learning_scripts/run_cl_train.sh

# Smoke test (~3 min, pipeline verification)
bash scripts/run_continual_learning_scripts/run_cl_train.sh --smoke

# Switch architecture
bash scripts/run_continual_learning_scripts/run_cl_train.sh \
    --yaml configs/continual_learning/neurovla_er_libero.yaml \
    --run-id my_5f
```

Checkpoints: `results/Checkpoints/<run_id>/`.

## Evaluate

```bash
# LoRA run — base-config required for adapter merge
bash scripts/run_continual_learning_scripts/run_cl_eval.sh \
    --run-id alphabrain_er_lora_libero_goal_v1 \
    --base-config configs/continual_learning/qwengr00t_er_lora_libero.yaml \
    --gpus 0,1

# Full-param run
bash scripts/run_continual_learning_scripts/run_cl_eval.sh \
    --run-id neurovla_cl_libero_goal_v1 --gpus 1
```

Per-task SR + 10×10 matrix: `results/eval_cl/<run_id>/`.

---

Full yaml preset list, CLI flags, headline results (5a–5l), and author notes: [`scripts/run_continual_learning_scripts/README.md`](https://github.com/AlphaBrainGroup/AlphaBrain/blob/main/scripts/run_continual_learning_scripts/README.md).
