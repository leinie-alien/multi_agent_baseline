"""实验统一入口。

用法示例:
    python scripts/run_experiment.py --method direct --limit 5          # sanity check
    python scripts/run_experiment.py --method cot --repeat 3
    python scripts/run_experiment.py --method self_consistency --repeat 3
    python scripts/run_experiment.py --method cot --disable-thinking      # 关闭思考做对比
    python scripts/run_experiment.py --method cot --enable-thinking       # 强制开启思考

需要 vLLM + GPU 环境,留给服务器环境阶段执行。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DATA_PATH,
    DEFAULT_SEEDS,
    ModelConfig,
    RESULTS_DIR,
)
from src.data_loader import load_questions  # noqa: E402
from src.methods.cot import run_cot  # noqa: E402
from src.methods.direct import run_direct  # noqa: E402
from src.methods.self_consistency import run_self_consistency  # noqa: E402
from src.model_runner import ModelRunner  # noqa: E402
from src.multi_agent.debate import run_debate  # noqa: E402
from src.multi_agent.pev import run_pev  # noqa: E402
from src.results_io import save_run_result  # noqa: E402

METHOD_RUNNERS = {
    "direct": run_direct,
    "cot": run_cot,
    "self_consistency": run_self_consistency,
    "debate": run_debate,
    "pev": run_pev,
}

# 多智能体的输出预算由角色内部控制,因此与基线入口分开调用。
MULTI_AGENT_METHODS = {"debate", "pev"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行单个方法的实验")
    parser.add_argument("--method", required=True, choices=list(METHOD_RUNNERS.keys()))
    parser.add_argument("--repeat", type=int, default=3, help="重复运行次数,默认 3")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="覆盖默认 seed 列表")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题,用于 sanity check")
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / DATA_PATH))
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / RESULTS_DIR))
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=None,
        help="覆盖 ModelConfig.tensor_parallel_size(默认 2),单卡可用时传 1",
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true", default=None,
        help="强制开启思考(对 cot/self_consistency/debate/pev 有效,direct 始终关闭)",
    )
    thinking_group.add_argument(
        "--disable-thinking", dest="enable_thinking", action="store_false",
        help="强制关闭思考(对 cot/self_consistency/debate/pev 有效,direct 始终关闭)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    questions = load_questions(args.data_path)
    if args.limit is not None:
        questions = questions[: args.limit]

    seeds = args.seeds if args.seeds else DEFAULT_SEEDS[: args.repeat]

    model_config = ModelConfig()
    tensor_parallel_size = (
        args.tensor_parallel_size
        if args.tensor_parallel_size is not None
        else model_config.tensor_parallel_size
    )
    runner = ModelRunner(
        model_path=model_config.model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=model_config.gpu_memory_utilization,
        max_model_len=model_config.max_model_len,
        max_num_seqs=model_config.max_num_seqs,
    )

    run_fn = METHOD_RUNNERS[args.method]
    results_dir = Path(args.results_dir)

    try:
        for seed in seeds:
            if args.method in MULTI_AGENT_METHODS:
                run_result = run_fn(runner, questions, seed=seed, enable_thinking=args.enable_thinking)
            else:
                run_result = run_fn(
                    runner,
                    questions,
                    max_tokens=model_config.max_tokens,
                    seed=seed,
                    enable_thinking=args.enable_thinking,
                )
            save_run_result(run_result, results_dir)
    finally:
        # 确保 vLLM 子进程被清理,避免孤儿进程占用显存(参见 LEARNING_NOTES.md)
        runner.shutdown()


if __name__ == "__main__":
    main()
