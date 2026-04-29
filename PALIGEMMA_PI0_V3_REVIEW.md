# PaliGemmaPi05 v3 (`paligemma_pi0_openpi_aligned_v3`) 实现回顾 & 待办

本文档整理 v3 训练 / 推理路径相对于 openpi `pi05_libero` 的工程层面差异和可改进点，供后续贡献者认领。算法层面 v3 与 openpi 等价、权重双向兼容，本文不再讨论；以下全部是工程实现层的优化。

**参考实现**：`/share/ziyangrao/openpi`（`models/pi0.py` JAX + `models_pytorch/pi0_pytorch.py` + `models_pytorch/gemma_pytorch.py` + `transformers_replace/`）。

**v3 涉及文件**：
- `AlphaBrain/model/framework/PaliGemmaPi.py`
- `AlphaBrain/model/modules/vlm/paligemma.py`
- `AlphaBrain/model/modules/action_model/pi0_flow_matching_head/`
  - `pi0_action_head.py`
  - `adarms_patch.py`
  - `weight_bridge.py`
  - `pi0_transforms.py`
- `configs/finetune_config.yaml`（`paligemma_pi0_openpi_aligned_v3` mode）

---

## 优先级总览

| Tier | 描述 |
|---|---|
| P0 | 影响正确性 / 性能的真问题，建议优先修 |
| P1 | 维护性 / 鲁棒性，长期有价值 |
| P2 | 性能优化 |
| P3 | 风格 / 文档清理 |

建议短期内挑 3 件做：**P0-1（统一 max_token_len）**、**P0-3 + P0-4（合并推理两份实现并删孤儿代码）**、**P1-2（norm stats 走 json）**。

---

## P0-1. 训练 / 推理 `max_token_len` 不一致

**位置**：
- 训练：`AlphaBrain/model/framework/PaliGemmaPi.py:397`
  ```python
  max_len = getattr(paligemma_cfg, 'max_token_len', 48)
  ```
- 推理：`AlphaBrain/model/framework/PaliGemmaPi.py:718`
  ```python
  _PREDICT_MAX_LEN = 200  # match openpi PaligemmaTokenizer default
  ```
- v3 配置 `configs/finetune_config.yaml:392-396` 没有显式设 `max_token_len`，所以训练实际用的是 **48**。

**对照**：openpi `pi05_libero` 训练 / 推理统一 200（`pi0_config.py:39` 当 `pi05=True` 时 `max_token_len=200`）。

**问题**：
1. 数学上不影响有效 token 输出（padding 被 mask 掉、`position_ids = cumsum(pad_masks) - 1` 不会撞），但推理 prefix 长度比训练长 ~4×，VLM 18 层和 SigLIP 都白算约 150 个 padding token，**单步推理慢 1.5–2×**。
2. 一旦未来某条 prompt 在训练时被截到 48 而推理时塞进 200，会出现训练没见过的尾部 token 分布。

**建议修法**：
- 推理路径读 `paligemma_cfg.max_token_len` 而不是硬编码 200。

---

## P0-2. `attn_implementation: flash_attention_2` 实际是死配置

**位置**：
- yaml `configs/finetune_config.yaml:395`：`attn_implementation: "flash_attention_2"`
- 加载：`AlphaBrain/model/modules/vlm/paligemma.py:76-94`

**问题**：v3 训练走 `flow_matching_head.compute_loss → _shared_forward`（`pi0_action_head.py:678-758`），里面是手算 Q/K/V joint attention 后调 `modeling_gemma.eager_attention_forward`，**完全不调用 VLM 的 `forward`**。所以这个 flash_attn 配置在 v3 训练里没生效。推理路径（`PaliGemmaPi.py:817`）又显式切回 `eager`。

**建议修法**：
- 删掉 yaml 里的 `attn_implementation` 字段，或注释清楚"仅 paligemma_oft 等单独走 VLM forward 的路径生效，v3 不读"。

---

## P0-3. 推理代码两份实现，需合并

**位置**：
- 抽象层：`AlphaBrain/model/modules/action_model/pi0_flow_matching_head/pi0_action_head.py:446-551` —— `Pi0FlowMatchingHead.sample_actions`
- 内联实现：`AlphaBrain/model/framework/PaliGemmaPi.py:707-887` —— `predict_action` 里完全重写一遍 prefix→KV cache→denoise 循环

**问题**：内联实现的注释写 `"Use openpi-style KV cache inference directly (proven to work in AB test)"`，应是历史上对齐时为了排错绕开抽象。现在两条路径必须同步维护，bug 修复风险翻倍。

**建议修法**：
- 让 `predict_action` 的 PaliGemma 分支调用 `flow_matching_head.sample_actions`。
- 把 inline 的 `_tokenize_openpi_style` / `_proc_img` 抽成独立的 inference-time preprocessing helper（可以放在 `pi0_transforms.py` 里）。
- 验证 LIBERO eval 数值与现版本一致后再合并。

---

## P0-4. 大量 dead code 待清理

**位置**：`AlphaBrain/model/modules/action_model/pi0_flow_matching_head/pi0_action_head.py`

| 函数 | 行 | 状态 |
|---|---|---|
| `compute_loss_prefix_cache` | 360–444 | Qwen 路径，v3 不走 |
| `sample_actions_prefix_cache` | 553–657 | 同上 |
| `compute_loss_llama` | 949–1005 | LlamaPi0，v3 不走 |
| `sample_actions_llama` | 1007–1076 | LlamaPi0 |
| `_shared_forward_llama` | 897–947 | LlamaPi0 |
| `_compute_prefix_cache` | 760–788 | **真孤儿**（grep 无调用方） |
| `_denoise_step` | 790–895 | **真孤儿**（grep 无调用方） |
| `_prepare_prefix_llama_raw` | `PaliGemmaPi.py:473-486` | 直接 return `_prepare_prefix_generic` |

**建议修法**：
- 立即删除 `_compute_prefix_cache` 和 `_denoise_step`（无调用方）。
- 把 Qwen / Llama 多 backend 代码拆到 `pi0_action_head_multimodal.py`，让核心文件从 1077 → ~250 行。
- 决定 LlamaPi0 / QwenPi05 是否还要维护，不维护就连同 `framework.LlamaPi0` 注册一起删。

---

## P1-1. monkey-patch 的全局副作用

**位置**：`AlphaBrain/model/modules/action_model/pi0_flow_matching_head/pi0_action_head.py:26-27`
```python
from .adarms_patch import patch_gemma_for_adarms
patch_gemma_for_adarms()
```

**问题**：import 时即执行，把整个 Python 进程的 `transformers.models.gemma.GemmaRMSNorm` 替换为 `AdaRMSNorm`。如果同进程还有其他 framework 用到 Gemma（间接经过 dataloader / 其他模型），它们会无差别拿到 patched 版本。`AdaRMSNorm.forward` 在 `cond=None` 时虽然走标准 RMSNorm 分支、参数名 `weight` 也对得上 HF 原版，但这种"权重加载靠巧合"的做法在升级 transformers 时容易踩雷。

**对照**：openpi 走 `transformers_replace/` 在 install 时 `cp` 到 site-packages，物理替换，没有进程内的 patch 时机问题。

**建议修法**：
- 把 patch 范围收窄到 `Pi0FlowMatchingHead.__init__` 内部，构造完 expert 后立即 unpatch。

---

## P1-2. norm 统计量硬编码进 yaml

**位置**：
- `configs/finetune_config.yaml:404-410`（v3 mode 下面 ~7 行 `action_mean/action_std/state_mean/state_std`）
- `results/training/paligemma_pi0_openpi_aligned_v3/final_model/framework_config.yaml:25-60`

**问题**：当前 norm stats 是几十个 float 字面量，换 dataset_mix（如 libero_goal → libero_all → 自有数据集）就要手算重算粘贴，容易和实际数据不一致；模型 final_model 里也固化了该次训练时的统计量，复现性差。

**对照**：openpi 走 `norm_stats.json`，仓库里 `pi0_transforms.py:78` 已经实现了 `load_norm_stats`，只是 v3 没接上。

**建议修法**：
1. dataloader 在首 epoch（或离线脚本）计算 `norm_stats.json`，落盘到 dataset 目录。
2. framework `__init__` 里读取该 json 写入 buffer，yaml 改成 `normalization.stats_path: data/datasets/.../norm_stats.json` 一行。

---

## P1-3. tokenizer 初始化的 try/except 黑洞

**位置**：`AlphaBrain/model/framework/PaliGemmaPi.py:229-237`
```python
for td in tokenizer_dirs:
    try:
        from transformers import AutoTokenizer
        self._hf_tokenizer = AutoTokenizer.from_pretrained(td)
        ...
        return
    except Exception:
        continue
```

**问题**：吞所有异常静默 fallback，调试 tokenizer 加载失败时只能看最后一条 `FileNotFoundError`，定位困难。

**建议修法**：
- 把 `except Exception` 收窄到 `(OSError, ImportError, ValueError)`。

---

## P1-4. `_shared_forward` 中的 magic number `1 * 8 * head_dim`

**位置**：`AlphaBrain/model/modules/action_model/pi0_flow_matching_head/pi0_action_head.py:718`
```python
att_output = att_output.reshape(q.shape[0], -1, 1 * 8 * head_dim)
```

**问题**：`8` 是 num_attention_heads 的硬编码，目前与 v3 配置 `num_heads=8` 一致是巧合。改了配置就会静默错形（reshape 不报错但语义错）。

**对照**：openpi PyTorch 那边 (`gemma_pytorch.py:210`) 同样有这个问题，是从那边继承下来的。

**建议修法**：用 `q.shape[1] * head_dim` 替换，让 head 数从 tensor 形状自动推导。

---

## P1-5. 调试日志留在 production

**位置**：`AlphaBrain/model/framework/PaliGemmaPi.py:781-801`（`_debug_count` + `print(...)` 段）

**问题**：每次 inference 启动头 3 次会无条件 `print` prefix 统计 + image meta，对齐期间留的痕迹。现在每次启 eval server 都会打。

**建议修法**：
- 改成 `logger.debug(...)`，或加 env flag `AB_PI0_DEBUG=1` 才打开。
- 注意里面用了 `sys.stdout.flush(); sys.stderr.flush()`，明显是当时排查 buffer 顺序时加的，连同 print 一起删。

---

## P1-6. 框架 init 里的"防御性 default"埋雷

**位置**：`AlphaBrain/model/framework/PaliGemmaPi.py:107-112`
```python
action_expert_num_heads=getattr(expert_cfg, 'num_heads', 8 if expert_type == "gemma" else 32),
action_expert_num_kv_heads=getattr(expert_cfg, 'num_kv_heads', 1 if expert_type == "gemma" else 8),
action_expert_head_dim=getattr(expert_cfg, 'head_dim', 256 if expert_type == "gemma" else 128),
```

**问题**：v3 yaml 显式声明了所有字段，default 永远不生效；但万一未来漏写一个，会用一个**和 pi05_base 权重不匹配的形状**静默初始化，权重加载只报 shape mismatch warning（`weight_bridge.py:140-143`），不是 error，模型会带着错误形状继续训练 / 推理。

**建议修法**：
- 去掉 default，缺字段直接 `raise ValueError`。

---

## P1-7. `gripper_remap` 默认值靠 framework 名隐式决定

**位置**：`AlphaBrain/model/framework/PaliGemmaPi.py:148-149`
```python
gripper_default = (config.framework.name == "PaliGemmaPi05")
self.gripper_remap = bool(getattr(config.framework, 'gripper_remap', gripper_default))
```

**问题**：dim-6 后处理映射 `[0,1] → [+1,-1]` 是 LIBERO eval client 专属，但默认值取决于 `framework.name == "PaliGemmaPi05"`。如果将来 LIBERO 以外的数据集也用 `PaliGemmaPi05` framework，会被默默打开 gripper remap 导致动作错误。

**建议修法**：
- 把 `gripper_remap` 改为显式必填字段，或让它默认 `False` 并在 LIBERO 数据集的 yaml 里显式设 `True`。

---

## P2-1. DynamicCache 重建 vs 直接保留 legacy tuple

**位置**：`AlphaBrain/model/framework/PaliGemmaPi.py:826-867`

**问题**：每个去噪步都 `to_legacy_cache() → DynamicCache.from_legacy_cache()` 重建一次，10 步推理 = 10 次拷贝 18 层 KV。当前实现是为了避免 `DynamicCache` 跨步累积 expert K/V（`transformers >= 4.45` 后 cache 是 stateful 的）。

**建议修法**：
- 第一次 prefix 推理后立即 `_vlm_kv_legacy = past_key_values.to_legacy_cache()`，后续步直接把 legacy tuple 当 `past_key_values` 传入（HF 4.45+ `forward` 同时接受 `DynamicCache` 和 legacy tuple，会内部转换）。
- 或者维护一个 read-only DynamicCache 的 wrapper，禁用 append。

---

## P2-2. 图像预处理 Python 循环

**位置**：`AlphaBrain/model/framework/PaliGemmaPi.py:335-372`

**问题**：每个样本 × 每个 view 单独跑 `TF.resize` + `TF.normalize`。LIBERO BS=64 × 3 views = 192 次串行调用。

**建议修法**：先 `torch.stack` 成 `[B*V, 3, H, W]` 再一次性 `TF.resize` / `TF.normalize`，PyTorch transforms 支持 batch tensor，可加速 2–4×。

---

## P2-3. tokenization 不缓存

**位置**：`AlphaBrain/model/framework/PaliGemmaPi.py:399-449`

**问题**：LIBERO 每个 task suite 任务指令只有几十种，但每次 forward 都重跑 sentencepiece encode。

**建议修法**：在 `_tokenize_openpi_style` 外层 `functools.lru_cache(maxsize=256)`（注意 cache 维度只能是字符串和数值，去掉 numpy state 之后再 cache）。或在数据集层面预 tokenize。

---

## P2-4. `_action_dim_mask` 每次 `.to(device)`

**位置**：`AlphaBrain/model/framework/PaliGemmaPi.py:213-214`（构造）+ `:614`（forward 里 `mask = self._action_dim_mask.to(loss.device)`）

**问题**：每个训练 step 都做一次 host→device 拷贝，虽然小。

**建议修法**：构造时改成 `self.register_buffer('_action_dim_mask', ...)`，自动跟 `model.to(device)` 走。代码里已有 `register_buffer('action_mean', ...)` 的先例，照抄即可。


---

## P3-1. RoPE 共用的隐式依赖未文档化

**位置**：
- `pi0_action_head.py:709`：`cos, sin = vlm_language_model.rotary_emb(dummy, position_ids)`
- `pi0_action_head.py:833`：`vlm_rotary = self._vlm_rotary_emb if hasattr(self, '_vlm_rotary_emb') else expert_model.rotary_emb`，注释写了 "CRITICAL: use VLM's rotary_emb to match training"

**问题**：joint forward 同样依赖 VLM 的 rotary_emb（理由是 VLM 和 expert 必须用同一份 RoPE），但 `_shared_forward` 那条路径没写 CRITICAL 注释。将来谁改 expert 的 RoPE 配置（max_position_embeddings、rope_scaling 等）会踩雷。

**建议修法**：在 `_shared_forward` 的 `compute_layer` 里加同样的注释，或封装一个 `_get_shared_rotary()` helper 让依赖显式化。

---

---

## 其他备注

- v3 在 `discrete_state_input: false` 模式下，state 既不进 prompt（因为 false）也不进 suffix（因为 `pi05=True` 时 `state_proj` 不构建）—— 这是和 openpi `pi05_libero` 完全对齐的有意行为，不是 bug。
- EMA 在 v3 关闭（`ema.enabled: false`），openpi `pi05_libero` 默认 `ema_decay=0.999`。两边唯一一个有意不对齐的训练超参，未来如果发现收敛后期波动可以试着打开。
- `pi0_transforms.py` 是为外部推理服务（LIBERO eval client）准备的独立 transforms，不被训练 / 推理主路径依赖。其中 `LiberoTransform` / `PaliGemmaTokenizer` 类与 `PaliGemmaPi.py` 内部的 inline 实现是**第三套** tokenization 代码，合并 P0-3 时可以一并考虑。

---


