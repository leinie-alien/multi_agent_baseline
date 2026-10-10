"""Router-Specialist-Verifier(拓展D,workflow范式): 用 orchestrator 拼出具体流程,转换成 QuestionRecord。"""
from __future__ import annotations

from ..config import RouterConfig
from ..extract_answer import is_equiv
from ..methods.base import QuestionRecord, RunResult
from ..model_runner import ModelRunner
from .orchestrator import run_agent_loop
from .roles import make_router_functions

def run_router(
    runner: ModelRunner,
    questions: list[dict],
    *,
    seed: int,
    enable_thinking: bool | None = None,
    config: RouterConfig | None = None,
) -> RunResult:
    config = config or RouterConfig()
    next_phase_fn, apply_fn = make_router_functions(config, enable_thinking=enable_thinking)
    # route 只执行一次,重试只打回 Specialist;与 PEV 整体重跑三阶段不同。
    # 首次及每次重试各占一次 specialize 和一次 verify。
    max_outer_rounds = 1 + 2 * (config.max_feedback_rounds + 1)

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
                method="router",
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

    return RunResult(method="router", seed=seed, records=records)
