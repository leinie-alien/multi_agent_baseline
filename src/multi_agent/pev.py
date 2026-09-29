"""PEV(Planner-Executor-Verifier): 用 orchestrator 拼出具体流程,转换成 QuestionRecord。"""
from __future__ import annotations

from ..config import PEVConfig
from ..extract_answer import is_equiv
from ..methods.base import QuestionRecord, RunResult
from ..model_runner import ModelRunner
from .orchestrator import run_agent_loop
from .roles import make_pev_functions

# plan/execute/verify 三阶段循环,反馈轮次上限由 PEVConfig.max_feedback_rounds 控制;
# 这里的安全阀要留够 (max_feedback_rounds + 1) 轮完整的 plan->execute->verify 循环。
_STAGES_PER_CYCLE = 3


def run_pev(
    runner: ModelRunner,
    questions: list[dict],
    *,
    seed: int,
    enable_thinking: bool | None = None,
    config: PEVConfig | None = None,
) -> RunResult:
    config = config or PEVConfig()
    next_phase_fn, apply_fn = make_pev_functions(config, enable_thinking=enable_thinking)
    max_outer_rounds = (config.max_feedback_rounds + 1) * _STAGES_PER_CYCLE

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
                method="pev",
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

    return RunResult(method="pev", seed=seed, records=records)
