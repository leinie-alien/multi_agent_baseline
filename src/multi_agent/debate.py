"""Debate(Proposer-Critic-Refine): 用 orchestrator 拼出具体流程,转换成 QuestionRecord。"""
from __future__ import annotations

from ..config import DebateConfig
from ..extract_answer import is_equiv
from ..methods.base import QuestionRecord, RunResult
from ..model_runner import ModelRunner
from .orchestrator import run_agent_loop
from .roles import make_debate_functions


def run_debate(
    runner: ModelRunner,
    questions: list[dict],
    *,
    seed: int,
    enable_thinking: bool | None = None,
    config: DebateConfig | None = None,
) -> RunResult:
    config = config or DebateConfig()
    next_phase_fn, apply_fn = make_debate_functions(config, enable_thinking=enable_thinking)

    states = run_agent_loop(
        runner,
        questions,
        next_phase_fn=next_phase_fn,
        apply_fn=apply_fn,
        seed=seed,
        max_outer_rounds=config.max_rounds,
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
                method="debate",
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

    return RunResult(method="debate", seed=seed, records=records)
