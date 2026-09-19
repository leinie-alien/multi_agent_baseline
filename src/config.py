"""全局实验配置。"""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_path: str = "/data/lsl/cache/huggingface/models/Qwen3.5-9B"
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 8192
    max_tokens: int = 4096  # 三种方法统一的最大输出长度上限(2048 时怀疑推理被截断,调大到 4096)
    # vLLM 默认按 256 路并发热身(sampler warmup + CUDA graph 最小 KV cache
    # 预留),但本实验一次最多 200 题 * Self-Consistency n=5,远用不到 256。
    # 调低能同步降低这两处启动阶段的显存占用,避免顶穿 gpu_memory_utilization
    # 预算导致启动阶段 OOM(详见 LEARNING_NOTES.md)。
    max_num_seqs: int = 32


@dataclass
class MethodSamplingConfig:
    """每种方法的采样参数。三个方法必须共用同一个 ModelConfig,只有这里的采样参数不同。"""
    temperature: float
    n: int = 1


DIRECT_SAMPLING = MethodSamplingConfig(temperature=0.6, n=1)
COT_SAMPLING = MethodSamplingConfig(temperature=0.6, n=1)
SELF_CONSISTENCY_SAMPLING = MethodSamplingConfig(temperature=0.7, n=5)

# 每种方法重复运行的次数与对应 seed(seed 与重复次数一一对应,不同 repeat 用不同 seed
# 以产生真实的采样波动,同时保证结果可复现)。
DEFAULT_SEEDS: list[int] = [41, 42, 43]

DATA_PATH = "data/math500_200.json"
SELECTED_IDS_PATH = "data/selected_ids.txt"
RESULTS_DIR = "results"

LEVELS = [1, 2, 3, 4, 5]
QUESTIONS_PER_LEVEL = 40
TOTAL_QUESTIONS = 200

FINAL_ANSWER_INSTRUCTION = "\n\nPlease put your final answer within \\boxed{}."
