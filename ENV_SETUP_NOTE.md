# Qwen3.5-9B + vLLM 环境配置笔记

> 服务器：4x RTX 4090 (24GB)，CUDA Driver 535.183.01 (最高支持 CUDA 12.2)
> 日期：2026-08-07 ~ 2026-08-09

## 最终可用配置

| 组件 | 版本 |
|------|------|
| Python | 3.10.20 |
| vLLM | 0.19.1 |
| PyTorch | 2.10.0+cu129 |
| transformers | 5.14.1 |
| CUDA Driver | 535.183.01 |

## 加载命令

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=2,3 python -c "
from vllm import LLM
llm = LLM(
    model='/data/lsl/cache/huggingface/models/Qwen3.5-9B',
    tensor_parallel_size=2,
    gpu_memory_utilization=0.85
)
print('model load success')
"
```

## 踩坑记录

### 坑 1：vLLM 版本太旧，不支持 Qwen3.5

**现象**：
```
ValueError: Model architectures ['Qwen3_5ForConditionalGeneration'] are not supported for now.
```

**原因**：原始安装的 vLLM 0.6.6.post1 发布于 2024 年底，Qwen3.5 是 2025 年中发布的，架构 `Qwen3_5ForConditionalGeneration` 不在支持列表中。

**解决**：vLLM 0.19.1 开始支持 Qwen3.5（`Qwen3_5ForCausalLM` 在 registry 中）。

### 坑 2：transformers 版本兼容性

**现象**：
- transformers 5.14.1（太新）：删除了 `all_special_tokens_extended` 属性，vLLM 0.8.5 报错 `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`
- transformers 4.57.6（太旧）：不认识 `qwen3_5` 模型类型，报错 `The checkpoint you are trying to load has model type 'qwen3_5' but Transformers does not recognize this architecture`

**原因**：vLLM 0.8.5~0.15.0 要求 `transformers<5`，但 `qwen3_5` 模型类型是 transformers 5.x 才加入的。vLLM 0.19.1 虽然也声明 `transformers<5`，但强制安装 transformers 5.14.1 后实际可用。

**解决**：强制安装 transformers 5.14.1，忽略 pip 的依赖冲突警告（vLLM 0.19.1 已经适配了 transformers 5.x 的 API 变化）。

### 坑 3：libstdc++ ABI 不兼容

**现象**：
```
OSError: libtorch_c_dlpack_addon_torch26-cuda.so: undefined symbol: _ZNSt15__exception_ptr13exception_ptr9_M_addrefEv
```

**原因**：系统是 Ubuntu 20.04，libstdc++ 版本 `GLIBCXX_3.4.28`（GCC 10.5.0），缺少 `_M_addref` 符号。vLLM 0.8.5 的 `torch_c_dlpack_ext` 依赖这个符号。

CUDA 12.4 自带的 libstdc++ 有这个符号，但它用的是新 C++11 ABI（`cxx11`），而 `torch_c_dlpack_ext` 用旧 ABI 编译，导致另一个不兼容。

**解决**：vLLM 0.19.1 的 flashinfer 不再依赖有问题的 `torch_c_dlpack_ext` 版本，该问题自然消失。

### 坑 4：pip 安装时磁盘空间不足

**现象**：
```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

**原因**：pip 默认缓存目录 `~/.cache/pip` 在系统盘 `/`（只有 879G，已用 832G），tmp 也在系统盘。

**解决**：
1. 移动 pip 缓存到数据盘：`mv ~/.cache/pip /data/lsl/cache/pip_cache`
2. 在 `/data/lsl/env.sh` 中配置永久生效：
   ```bash
   export PIP_CACHE_DIR=$CACHE/pip_cache
   ```
3. 安装时使用 `TMPDIR=/data/lsl/tmp` 避免临时文件写系统盘

### 坑 5：GPU 被占用

**现象**：
```
ValueError: Free memory on device cuda:0 (21.25/23.65 GiB) on startup is less than desired GPU memory utilization (0.9, 21.28 GiB)
```

**原因**：GPU 0、1 被其他 Python 进程占用（分别用了 1.8G 和 12.6G）。

**解决**：使用 `CUDA_VISIBLE_DEVICES=2,3` 指定空闲的 GPU。

## 版本演进路径

| 尝试 | vLLM 版本 | 结果 | 卡在哪 |
|------|-----------|------|--------|
| 1 | 0.6.6.post1 | ❌ | 不支持 Qwen3.5 架构 |
| 2 | 0.8.5.post1 | ❌ | libstdc++ ABI 问题 + tokenizer 不兼容 |
| 3 | 0.9.2 | ❌ | 无 Qwen3.5 原生实现 |
| 4 | 0.15.0 | ❌ | 不支持 `Qwen3_5ForConditionalGeneration`，需 transformers 5.x |
| 5 | 0.19.1 | ✅ | 成功！架构识别 + GPU 内存 + 所有依赖均正常 |

## 关键经验

1. **Qwen3.5 需要 vLLM >= 0.19.x**（具体从哪个 minor 版本开始支持未确认，0.15.0 不支持，0.19.1 支持）
2. **transformers 5.x 可以强制搭配 vLLM 0.19.1**，虽然 pip 报冲突但不影响使用
3. **4x RTX 4090**：9B 模型用 2 卡 tensor parallel 足够，`gpu_memory_utilization=0.85` 留余量
4. **pip 缓存务必放数据盘**，系统盘 1T 很容易满
5. **CUDA Driver 535 兼容 CUDA 12.9**：vLLM 0.19.1 带的 torch 2.10.0 用 CUDA 12.9，虽然 driver 只保证到 12.2，但 forward compatibility 正常工作