# Context-to-Temporary-FFN 云服务器实验步骤

本文档用于在云服务器上从零开始复现 `Context-to-Temporary-FFN` 实验。目标是让实验者在 clone 仓库后，可以直接按照步骤完成环境配置、随机 QKV 模拟实验、真实 Transformer 激活拟合实验、layer/head sweep 和显存参数量分析。

## 1. 实验分支

本实验代码位于以下分支：

```bash
codex/context-to-temp-ffn-docs
```

建议在云服务器上直接 clone 该分支：

```bash
git clone -b codex/context-to-temp-ffn-docs git@github.com:lifeforzw/Model-Attribution.git
cd Model-Attribution
```

如果服务器没有配置 GitHub SSH key，可以改用 HTTPS：

```bash
git clone -b codex/context-to-temp-ffn-docs https://github.com/lifeforzw/Model-Attribution.git
cd Model-Attribution
```

确认当前分支：

```bash
git branch --show-current
```

期望输出：

```text
codex/context-to-temp-ffn-docs
```

## 2. 服务器环境要求

建议环境：

| 项目 | 建议 |
| --- | --- |
| OS | Ubuntu 20.04/22.04/24.04 |
| Python | 3.10 或更高 |
| GPU | NVIDIA GPU，建议 16GB 显存以上 |
| CUDA | 与当前 PyTorch 安装匹配即可 |
| 磁盘 | 至少 10GB，可按模型大小增加 |

随机 QKV 实验可以在 CPU 上运行，但真实模型激活实验建议使用 GPU。

## 3. 创建 Python 环境

使用 venv：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

安装项目和实验依赖：

```bash
pip install -e ".[dev,transformers]"
pip install accelerate
```

如果服务器已经有适配 CUDA 的 PyTorch，也可以先安装项目，再保留已有 torch：

```bash
pip install -e ".[dev]"
pip install transformers accelerate
```

检查核心依赖：

```bash
python - <<'PY'
import torch
import transformers
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
PY
```

如果 `cuda available: False`，实验仍可在 CPU 上跑小规模版本，但真实模型会较慢。

## 4. 基础代码检查

设置源码路径：

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

运行编译检查：

```bash
python -m compileall src tests examples
```

如果已安装 `pytest`，可以运行测试：

```bash
python -m pytest -q
```

## 5. 快速 Smoke Test

建议先跑一个很小的随机实验，确认训练链路和输出目录正常：

```bash
PYTHONPATH=src python examples/run_temp_ffn_compression.py \
  random \
  --context-length 64 \
  --head-dim 16 \
  --query-count 32 \
  --memory-units 4,8,16 \
  --steps 20 \
  --batch-size 16 \
  --output-dir runs/temp_ffn/smoke \
  --run-name random_smoke
```

成功后会在终端看到类似表格：

```text
M       MSE             RelErr          CosSim          Ratio
4       ...
8       ...
16      ...
Saved summary: runs/temp_ffn/smoke/random_smoke.summary.json
```

输出文件：

```bash
ls runs/temp_ffn/smoke
```

期望至少包含：

```text
random_smoke.summary.json
```

## 6. 实验一：随机 QKV 数学模拟

该实验对应实验文档中的“随机 $Q,K,V$ 的数学模拟实验”。

### 6.1 使用脚本运行默认实验

```bash
PYTHONPATH=src bash scripts/run_temp_ffn_random.sh
```

默认参数：

| 参数 | 默认值 |
| --- | --- |
| `CONTEXT_LENGTH` | `4096` |
| `HEAD_DIM` | `64` |
| `QUERY_COUNT` | `2048` |
| `MEMORY_UNITS` | `32,64,128,256,512,1024` |
| `STEPS` | `500` |
| `BATCH_SIZE` | `512` |
| `OUTPUT_DIR` | `runs/temp_ffn/random` |

### 6.2 自定义规模

较小 GPU 或 CPU 环境可以降低规模：

```bash
CONTEXT_LENGTH=1024 \
QUERY_COUNT=512 \
MEMORY_UNITS=32,64,128,256 \
STEPS=300 \
BATCH_SIZE=256 \
PYTHONPATH=src bash scripts/run_temp_ffn_random.sh
```

更接近正式实验的设置：

```bash
CONTEXT_LENGTH=4096 \
HEAD_DIM=64 \
QUERY_COUNT=2048 \
MEMORY_UNITS=32,64,128,256,512,1024 \
STEPS=1000 \
BATCH_SIZE=512 \
DEVICE=cuda \
PYTHONPATH=src bash scripts/run_temp_ffn_random.sh
```

### 6.3 直接调用 Python CLI

```bash
PYTHONPATH=src python examples/run_temp_ffn_compression.py \
  random \
  --context-length 4096 \
  --head-dim 64 \
  --query-count 2048 \
  --memory-units 32,64,128,256,512,1024 \
  --steps 500 \
  --batch-size 512 \
  --device auto \
  --output-dir runs/temp_ffn/random \
  --run-name random_qkv
```

## 7. 实验四：上下文长度与压缩难度 Sweep

该实验改变上下文长度 $N$，观察固定或不同 $M$ 下的重构误差变化。

运行默认 sweep：

```bash
PYTHONPATH=src bash scripts/run_temp_ffn_random_sweep.sh
```

默认会依次运行：

```text
N = 512, 1024, 2048, 4096, 8192
M = 32, 64, 128, 256, 512
```

自定义 sweep：

```bash
CONTEXT_LENGTHS="512 1024 2048" \
CONTEXT_LENGTHS_CSV="512,1024,2048" \
MEMORY_UNITS=32,64,128 \
STEPS=300 \
PYTHONPATH=src bash scripts/run_temp_ffn_random_sweep.sh
```

输出目录：

```text
runs/temp_ffn/random_sweep/
```

其中每个 `*.summary.json` 对应一个上下文长度设置。

## 8. 实验二：真实 Transformer 激活拟合

该实验从真实 HuggingFace causal LM 内部抽取某一层、某个 head 的 $Q,K,V$，然后训练临时 softmax-FFN 拟合完整 attention 输出。

### 8.1 GPT-2 small

GPT-2 使用 packed QKV layout：

```text
[q_all, k_all, v_all]
```

默认脚本已适配 GPT-2 的模块名：

```bash
MODEL=gpt2 \
LAYER=6 \
HEAD=0 \
CONTEXT_LENGTH=512 \
QUERY_COUNT=128 \
MEMORY_UNITS=32,64,128,256 \
STEPS=500 \
DEVICE=cuda \
PYTHONPATH=src bash scripts/run_temp_ffn_hf_activations.sh
```

如果没有 GPU：

```bash
MODEL=gpt2 \
LAYER=6 \
HEAD=0 \
CONTEXT_LENGTH=128 \
QUERY_COUNT=32 \
MEMORY_UNITS=8,16,32 \
STEPS=50 \
DEVICE=cpu \
PYTHONPATH=src bash scripts/run_temp_ffn_hf_activations.sh
```

### 8.2 Pythia-160M

Pythia 使用 GPT-NeoX 风格 packed QKV layout：

```text
[head0_q, head0_k, head0_v, head1_q, head1_k, head1_v, ...]
```

运行命令：

```bash
MODEL=EleutherAI/pythia-160m \
QKV_LAYOUT=gpt-neox \
QKV_MODULE_PATTERN='gpt_neox.layers.6.attention.query_key_value' \
LAYER=6 \
HEAD=0 \
CONTEXT_LENGTH=512 \
QUERY_COUNT=128 \
MEMORY_UNITS=32,64,128,256 \
STEPS=500 \
DEVICE=cuda \
PYTHONPATH=src bash scripts/run_temp_ffn_hf_activations.sh
```

如果模型下载需要访问 HuggingFace，请确认服务器网络、代理或镜像源配置正常。

### 8.3 使用自定义文本

直接传入文本：

```bash
TEXT="The capital of France is Paris. This document contains repeated facts and semantic context. " \
MODEL=gpt2 \
LAYER=6 \
HEAD=0 \
PYTHONPATH=src bash scripts/run_temp_ffn_hf_activations.sh
```

从文件读取长文本：

```bash
TEXT_FILE=/path/to/long_context.txt \
MODEL=gpt2 \
LAYER=6 \
HEAD=0 \
PYTHONPATH=src bash scripts/run_temp_ffn_hf_activations.sh
```

脚本默认会在文本 token 数不足时重复文本，直到满足：

```text
context_length + query_count
```

如需禁止重复，可以直接调用 Python CLI 并加：

```bash
--no-repeat-text
```

## 9. 实验三：不同 Layer 和 Head 可压缩性分析

该实验用于比较不同 layer/head 的压缩难度。

### 9.1 GPT-2 layer/head sweep

```bash
MODEL=gpt2 \
LAYERS="0 3 6 9 11" \
HEADS="0 1 2 3" \
CONTEXT_LENGTH=512 \
QUERY_COUNT=128 \
MEMORY_UNITS=32,64,128,256 \
STEPS=350 \
DEVICE=cuda \
PYTHONPATH=src bash scripts/run_temp_ffn_hf_layer_head_sweep.sh
```

### 9.2 Pythia layer/head sweep

当前 sweep 脚本会按 layer 自动使用默认 GPT-2 模块名。如果跑 Pythia，建议直接用循环调用 Python CLI，显式传入模块名：

```bash
for LAYER in 0 3 6 9 11; do
  for HEAD in 0 1 2 3; do
    PYTHONPATH=src python examples/run_temp_ffn_compression.py \
      hf-activations \
      --model EleutherAI/pythia-160m \
      --qkv-layout gpt-neox \
      --qkv-module-pattern "gpt_neox.layers.${LAYER}.attention.query_key_value" \
      --layer "$LAYER" \
      --head "$HEAD" \
      --context-length 512 \
      --query-count 128 \
      --memory-units 32,64,128,256 \
      --steps 350 \
      --batch-size 128 \
      --device cuda \
      --output-dir runs/temp_ffn/pythia_layer_head_sweep \
      --run-name "pythia160m_layer${LAYER}_head${HEAD}"
  done
done
```

输出目录：

```text
runs/temp_ffn/hf_layer_head_sweep/
runs/temp_ffn/pythia_layer_head_sweep/
```

后续可以根据每个 summary 中的 `relative_error` 和 `cosine_similarity` 对 layer/head 排序。

## 10. 显存与参数量分析

生成 KV cache 与临时 FFN 参数量对比表：

```bash
PYTHONPATH=src python examples/run_temp_ffn_compression.py \
  memory \
  --context-lengths 512,1024,2048,4096,8192,32768 \
  --memory-units 32,64,128,256,512 \
  --layers 12 \
  --heads 12 \
  --head-dim 64 \
  --output-dir runs/temp_ffn/memory \
  --run-name gpt2_memory_table
```

输出示例：

```text
N=   512 M=  32 ratio=  16.00 kv=...
N=  1024 M=  64 ratio=  16.00 kv=...
```

核心压缩比为：

$$
\text{Compression Ratio}=\frac{N}{M}
$$

## 11. 输出文件说明

所有脚本默认将结果写入 `runs/temp_ffn/`，该目录已被 `.gitignore` 忽略，不会误提交实验产物。

每次实验主要输出：

```text
*.summary.json
```

如果设置 `SAVE_STUDENTS=1` 或 `--save-students`，还会输出：

```text
*.students.pt
```

### 11.1 summary JSON 结构

随机实验和真实激活实验的 summary 大致包含：

```json
{
  "experiment": "random_qkv",
  "settings": {},
  "results": [
    {
      "memory_units": 64,
      "metrics": {
        "mse": 0.0,
        "relative_error": 0.0,
        "cosine_similarity": 1.0
      },
      "history": [],
      "memory_cost": {
        "context_length": 4096,
        "memory_units": 64,
        "kv_cache_parameters": 524288,
        "temp_ffn_parameters": 8192,
        "compression_ratio": 64.0
      }
    }
  ]
}
```

重点关注字段：

| 字段 | 含义 |
| --- | --- |
| `memory_units` | 临时 FFN memory units 数量 $M$ |
| `metrics.mse` | 输出重构误差 |
| `metrics.relative_error` | 相对重构误差 |
| `metrics.cosine_similarity` | 输出方向相似度 |
| `memory_cost.compression_ratio` | 理论压缩比 $N/M$ |
| `history` | 训练过程中的 loss 记录 |

## 12. 结果查看与汇总

查看某次实验结果：

```bash
python -m json.tool runs/temp_ffn/random/random_qkv.summary.json | less
```

如果服务器安装了 `jq`，可以快速抽取每个 $M$ 的指标：

```bash
jq '.results[] | {M: .memory_units, mse: .metrics.mse, rel: .metrics.relative_error, cos: .metrics.cosine_similarity, ratio: .memory_cost.compression_ratio}' \
  runs/temp_ffn/random/random_qkv.summary.json
```

汇总 sweep 目录下所有 summary：

```bash
find runs/temp_ffn -name "*.summary.json" -print
```

## 13. 推荐实验顺序

建议按以下顺序执行：

1. **Smoke test**：确认环境和代码链路正常；
2. **随机 QKV 小规模实验**：验证训练和指标输出；
3. **随机 QKV 正式规模实验**：观察 $M$ 增大时的误差下降；
4. **上下文长度 sweep**：观察 $N$ 增大时的压缩难度；
5. **GPT-2 单 layer/head 激活实验**：验证真实激活是否更容易压缩；
6. **GPT-2 layer/head sweep**：分析不同层和头的差异；
7. **Pythia-160M 实验**：验证 GPT-NeoX 风格模型；
8. **显存参数量分析**：将重构指标和理论压缩比放在一起解释。

## 14. 常见问题

### 14.1 `ModuleNotFoundError: No module named 'llm_hookkit'`

说明没有设置 `PYTHONPATH`，执行：

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

或者安装为 editable package：

```bash
pip install -e ".[dev,transformers]"
```

### 14.2 `Context-to-Temporary-FFN experiments require PyTorch.`

说明当前环境没有安装 PyTorch。执行：

```bash
pip install torch
```

如果需要指定 CUDA 版本，请按服务器 CUDA 环境选择对应 PyTorch 安装命令。

### 14.3 HuggingFace 模型下载失败

可能原因：

- 服务器无法访问 HuggingFace；
- 没有配置代理；
- 模型需要登录或许可；
- 磁盘空间不足。

可以先使用 `gpt2` 测试，也可以将模型提前下载到本地目录，然后：

```bash
MODEL=/path/to/local/model PYTHONPATH=src bash scripts/run_temp_ffn_hf_activations.sh
```

### 14.4 真实激活实验提示没有匹配到 QKV 模块

报错示例：

```text
No module output matched --qkv-module-pattern
```

说明 `--qkv-module-pattern` 与模型模块名不匹配。可以先打印模型模块名：

```bash
PYTHONPATH=src python - <<'PY'
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
for name, _ in model.named_modules():
    if "attn" in name or "query" in name or "key" in name or "value" in name:
        print(name)
PY
```

然后选择实际的 packed QKV projection 模块名作为 `QKV_MODULE_PATTERN`。

### 14.5 CUDA OOM

降低以下参数：

```bash
CONTEXT_LENGTH=256
QUERY_COUNT=64
MEMORY_UNITS=16,32,64
STEPS=200
BATCH_SIZE=64
```

也可以先用 CPU 小规模 smoke test 验证流程。

### 14.6 Pythia 结果异常

请确认同时设置：

```bash
QKV_LAYOUT=gpt-neox
QKV_MODULE_PATTERN='gpt_neox.layers.6.attention.query_key_value'
```

GPT-2 和 Pythia 的 packed QKV 排列方式不同，layout 配错会导致 head 切分错误。

## 15. 最小可执行命令合集

从空服务器开始，最小流程如下：

```bash
git clone -b codex/context-to-temp-ffn-docs https://github.com/lifeforzw/Model-Attribution.git
cd Model-Attribution

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -e ".[dev,transformers]"
pip install accelerate

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m compileall src tests examples

python examples/run_temp_ffn_compression.py \
  random \
  --context-length 64 \
  --head-dim 16 \
  --query-count 32 \
  --memory-units 4,8,16 \
  --steps 20 \
  --batch-size 16 \
  --output-dir runs/temp_ffn/smoke \
  --run-name random_smoke

PYTHONPATH=src bash scripts/run_temp_ffn_random.sh
```

如果有 GPU 并希望继续跑 GPT-2 真实激活：

```bash
MODEL=gpt2 \
LAYER=6 \
HEAD=0 \
CONTEXT_LENGTH=512 \
QUERY_COUNT=128 \
MEMORY_UNITS=32,64,128,256 \
STEPS=500 \
DEVICE=cuda \
PYTHONPATH=src bash scripts/run_temp_ffn_hf_activations.sh
```

## 16. 实验结论记录建议

每次正式实验建议记录：

| 项目 | 内容 |
| --- | --- |
| Git commit | `git rev-parse --short HEAD` |
| 模型 | 如 `gpt2`、`EleutherAI/pythia-160m` |
| Layer / Head | 单点实验或 sweep 范围 |
| $N$ | context length |
| $T$ | query count |
| $M$ | memory units 列表 |
| steps / batch size | 训练设置 |
| MSE / RelErr / CosSim | 核心重构指标 |
| compression ratio | 理论压缩比 |
| GPU 型号 | 便于对比速度和显存 |

建议将最终表格与 `docs/context_to_temporary_ffn_experiment.zh.md` 中的问题一一对应，重点判断：

- 随机 QKV 是否存在稳定压缩趋势；
- 真实模型激活是否比随机 QKV 更容易压缩；
- 哪些 layer/head 更容易压缩；
- 较小 $M$ 是否能在较高压缩比下保持较好的 cosine similarity；
- 该方法更适合作为语义压缩记忆，还是能够支持更精确的 token-level 记忆替代。
