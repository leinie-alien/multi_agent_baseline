"""全局实验配置。"""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_path: str = "/data/lsl/cache/huggingface/models/Qwen3.5-9B"
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 36864  # 模型 max_position_embeddings=262144,这里只按显存预算保守设,非模型上限
    max_tokens: int = 32000  # 三种方法统一的最大输出长度上限。开思考模式后 Level 5 难题的真实推理
    # (非重复,是正常的分支探索)经常超过之前 13000/20000 的预算,导致输出被硬截断、没写到 \boxed{}
    # 就没了(详见 LEARNING_NOTES.md q161 相关记录)。当前先不对这类题目的推理长度设限,尽量把预算
    # 调大;仍然截断的极端长尾case由 extract_answer.py 的 allow_fallback 兜底,诚实返回 None 而不是
    # 伪造答案。
    # vLLM 默认按 256 路并发热身(sampler warmup + CUDA graph 最小 KV cache
    # 预留),但本实验一次最多 200 题 * Self-Consistency n=5,远用不到 256。
    # 调低能同步降低这两处启动阶段的显存占用,避免顶穿 gpu_memory_utilization
    # 预算导致启动阶段 OOM(详见 LEARNING_NOTES.md)。
    max_num_seqs: int = 32


@dataclass
class MethodSamplingConfig:
    """每种方法的采样参数。三个方法必须共用同一个 ModelConfig,只有这里的采样参数不同。

    repetition_penalty 默认 1.05(vLLM 默认是 1.0,即不惩罚):sanity check 中发现过
    模型正确给出 \\boxed{} 答案后陷入重复循环直到 max_tokens 截断的情况(q161),给
    一个温和的重复惩罚可以在不明显影响正常回答质量的前提下抑制这种退化循环。
    """
    temperature: float
    n: int = 1
    repetition_penalty: float = 1.05


DIRECT_SAMPLING = MethodSamplingConfig(temperature=0.6, n=1)
COT_SAMPLING = MethodSamplingConfig(temperature=0.6, n=1)
SELF_CONSISTENCY_SAMPLING = MethodSamplingConfig(temperature=0.7, n=5)


@dataclass
class DebateConfig:
    """Debate(Proposer-Critic-Refine)的角色数量/轮次/采样参数,是考核要求里
    "记录用了几个 agent、最多几轮、聚合/终止方式"的唯一代码来源。

    固定四阶段流程:solve -> [不一致才做] critic -> refine -> [仍不一致才做] judge。
    max_rounds=4 是这四个阶段的安全上限,真正是否进入某一阶段由 orchestrator 里的
    收敛判断决定(一致就早停跳过后续阶段),不是每题都跑满四阶段。
    """
    num_solvers: int = 3
    max_rounds: int = 4
    solver_temperature: float = 0.7
    critic_temperature: float = 0.6
    refine_temperature: float = 0.6
    judge_temperature: float = 0.3
    repetition_penalty: float = 1.05
    max_context_chars: int = 6000  # 角色间共享字符预算,为题目和模板预留上下文空间


@dataclass
class PEVConfig:
    """Planner-Executor-Verifier 的候选策略数/反馈轮次上限/采样参数。

    executor_temperature、verifier_temperature 默认给得比 planner 低:执行阶段是
    照既定策略机械算下去,验证阶段是机械复核不是发散点评,两者都不需要太多随机性;
    多样性主要靠 num_strategies(多个候选策略)而非采样温度提供。
    """
    num_strategies: int = 2
    max_feedback_rounds: int = 2  # Verifier 判 FAIL 后,Planner 最多被打回重新出策略的次数(不含首次)
    planner_temperature: float = 0.7
    executor_temperature: float = 0.3
    verifier_temperature: float = 0.3
    repetition_penalty: float = 1.05
    max_context_chars: int = 6000  # 角色间共享字符预算,避免长输出累积撑满上下文

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
