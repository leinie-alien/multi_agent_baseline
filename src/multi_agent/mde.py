"""MDE(Method-Diversity Ensemble + Reconciler,拓展C): 用 orchestrator 拼出具体流程,转换成 QuestionRecord。"""
from __future__ import annotations

from ..config import MDEConfig
from ..extract_answer import is_equiv
from ..methods.base import QuestionRecord, RunResult
from ..model_runner import ModelRunner
from .orchestrator import run_agent_loop
from .roles import make_mde_functions

def run_mde(
    runner: ModelRunner,
    questions: list[dict],
    *,
    seed: int,
    enable_thinking: bool | None = None,
    config: MDEConfig | None = None,
) -> RunResult:
    config = config or MDEConfig()
    next_phase_fn, apply_fn = make_mde_functions(config, enable_thinking=enable_thinking)
    # solve + reconcile 两阶段,早停由 apply_fn 的 done=True 处理;2 是安全阀,
    # 不代表每题都要跑两轮,只有分歧时才进入 reconcile。
    max_outer_rounds = 2

    states = run_agent_loop(
        runner,
        questions,
        next_phase_fn=next_phase_fn,
        apply_fn=apply_fn,
        seed=seed,
        max_outer_rounds=max_outer_rounds,
    )

    records: list[QuestionRecord] = []
    for q, state in zip(questions, states):
        input_tokens = state.scratch.get("total_input_tokens", 0)
        output_tokens = sum(t["output_tokens"] for t in state.trace)
        prediction = state.final_prediction
        correct = is_equiv(prediction, q["answer"]) if prediction is not None else False

        records.append(
            QuestionRecord(
                question_id=q["question_id"],
                level=q["level"],
                method="mde",
                prediction=prediction,
                ground_truth=q["answer"],
                correct=correct,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                raw_outputs=state.final_raw_outputs,
                finish_reasons=state.final_finish_reasons,
                rounds_used=state.round_idx,
                agent_trace=state.trace,
            )
        )

    return RunResult(method="mde", seed=seed, records=records)
