import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DebateConfig, PEVConfig  # noqa: E402
from src.model_runner import PromptResult, SampleResult  # noqa: E402
from src.multi_agent.debate import run_debate  # noqa: E402
from src.multi_agent.orchestrator import PhaseStep, run_agent_loop  # noqa: E402
from src.multi_agent.pev import run_pev  # noqa: E402
from src.multi_agent.roles import (  # noqa: E402
    _parse_strategies,
    make_debate_functions,
    make_pev_functions,
)


class SequencedRunner:
    """伪造 ModelRunner:每次 generate() 调用消费预先安排好的下一批 SampleResult。

    每个"批次"对应 orchestrator 一轮里的一次 generate 调用,长度必须与该轮实际发出的
    prompt 数一致,否则直接暴露 next_phase_fn/apply_fn 与预期流程不符的 bug,而不是
    静默错配。build_prompt 直接回传 messages 内容,不需要额外还原 chat template。
    """

    def __init__(self, batches: list[list[SampleResult]]) -> None:
        self._batches = list(batches)
        self.calls: list[dict] = []

    def build_prompt(self, messages, enable_thinking=None):
        return messages[0]["content"]

    def generate(self, prompts, temperature, max_tokens, n=1, seed=None, repetition_penalty=1.0):
        self.calls.append({"temperature": temperature, "num_prompts": len(prompts)})
        assert self._batches, "SequencedRunner 的批次已经用完,但调度器又发起了一次 generate"
        batch = self._batches.pop(0)
        assert len(batch) == len(prompts), (
            f"批次长度 {len(batch)} 与实际 prompt 数 {len(prompts)} 不一致"
        )
        return [
            PromptResult(prompt=p, input_tokens=10, samples=[s])
            for p, s in zip(prompts, batch)
        ]


def sample(answer: str, output_tokens: int = 10, finish_reason: str = "stop") -> SampleResult:
    return SampleResult(
        text=f"Reasoning...\n\\boxed{{{answer}}}",
        output_tokens=output_tokens,
        finish_reason=finish_reason,
    )


def plain(text: str, output_tokens: int = 10, finish_reason: str = "stop") -> SampleResult:
    return SampleResult(text=text, output_tokens=output_tokens, finish_reason=finish_reason)


def _question(answer="2", question_id="q1", level=5, problem="1+1=?"):
    return {"question_id": question_id, "level": level, "problem": problem, "answer": answer}


# ---------------------------------------------------------------------------
# orchestrator.run_agent_loop
# ---------------------------------------------------------------------------

def test_orchestrator_groups_steps_by_sampling_params():
    calls = []

    class RecordingRunner:
        def build_prompt(self, messages, enable_thinking=None):
            return messages[0]["content"]

        def generate(self, prompts, temperature, max_tokens, n=1, seed=None, repetition_penalty=1.0):
            calls.append({"temperature": temperature, "num_prompts": len(prompts)})
            return [
                PromptResult(prompt=p, input_tokens=1, samples=[plain("x")])
                for p in prompts
            ]

    questions = [_question(question_id=f"q{i}") for i in range(3)]

    def next_phase_fn(state):
        if state.scratch.get("finished"):
            return None
        temp = 0.1 if state.question["question_id"] == "q0" else 0.9
        return [PhaseStep(role="r", messages=[{"role": "user", "content": "hi"}], temperature=temp, max_tokens=10)]

    def apply_fn(state, steps, outputs):
        state.scratch["finished"] = True
        state.final_prediction = "a"
        state.done = True

    states = run_agent_loop(
        RecordingRunner(), questions, next_phase_fn=next_phase_fn, apply_fn=apply_fn, max_outer_rounds=5,
    )

    # 温度 0.1(q0 一题)与温度 0.9(q1、q2 两题)必须分到两次不同的 generate 调用,
    # 而不是被一起塞进同一次 batch。
    assert len(calls) == 2
    assert {"temperature": 0.1, "num_prompts": 1} in calls
    assert {"temperature": 0.9, "num_prompts": 2} in calls
    assert all(s.done for s in states)


def test_orchestrator_marks_done_when_next_phase_returns_none():
    class DummyRunner:
        def build_prompt(self, messages, enable_thinking=None):
            return "p"

        def generate(self, prompts, temperature, max_tokens, n=1, seed=None, repetition_penalty=1.0):
            raise AssertionError("next_phase_fn 已经返回 None,不应该再发起 generate 调用")

    def next_phase_fn(state):
        state.final_prediction = "already-done"
        return None

    def apply_fn(state, steps, outputs):
        raise AssertionError("next_phase_fn 返回 None 时不应该调用 apply_fn")

    states = run_agent_loop(
        DummyRunner(), [_question()], next_phase_fn=next_phase_fn, apply_fn=apply_fn, max_outer_rounds=5,
    )

    assert states[0].done is True
    assert states[0].final_prediction == "already-done"


def test_orchestrator_max_outer_rounds_safety_valve():
    class DummyRunner:
        def build_prompt(self, messages, enable_thinking=None):
            return "p"

        def generate(self, prompts, temperature, max_tokens, n=1, seed=None, repetition_penalty=1.0):
            return [
                PromptResult(prompt=p, input_tokens=1, samples=[plain("x")])
                for p in prompts
            ]

    def next_phase_fn(state):
        return [PhaseStep(role="r", messages=[{"role": "user", "content": "hi"}], temperature=0.5, max_tokens=10)]

    def apply_fn(state, steps, outputs):
        pass  # 故意永不置 done,验证安全阀会强制停止

    states = run_agent_loop(
        DummyRunner(), [_question()], next_phase_fn=next_phase_fn, apply_fn=apply_fn, max_outer_rounds=3,
    )

    assert states[0].round_idx == 3
    assert states[0].done is True
    assert states[0].final_prediction is None


# ---------------------------------------------------------------------------
# roles.make_debate_functions
# ---------------------------------------------------------------------------

def test_debate_early_stop_when_solvers_agree():
    runner = SequencedRunner([
        [sample("2"), sample("2"), sample("2")],  # solve: 一致,应早停
    ])
    config = DebateConfig()
    next_phase_fn, apply_fn = make_debate_functions(config)

    states = run_agent_loop(
        runner, [_question()], next_phase_fn=next_phase_fn, apply_fn=apply_fn, max_outer_rounds=config.max_rounds,
    )

    state = states[0]
    assert state.final_prediction == "2"
    assert state.round_idx == 1
    assert len(state.trace) == 3


def test_debate_disagreement_goes_through_critic_refine_judge():
    runner = SequencedRunner([
        [sample("2"), sample("3"), sample("4")],       # solve: 分歧
        [plain("critic feedback")],                      # critic
        [sample("2"), sample("3"), sample("4")],       # refine: 仍分歧
        [sample("5")],                                    # judge
    ])
    config = DebateConfig()
    next_phase_fn, apply_fn = make_debate_functions(config)

    states = run_agent_loop(
        runner, [_question()], next_phase_fn=next_phase_fn, apply_fn=apply_fn, max_outer_rounds=config.max_rounds,
    )

    state = states[0]
    assert state.final_prediction == "5"
    assert state.round_idx == 4
    assert len(state.trace) == 3 + 1 + 3 + 1


def test_debate_refine_agreement_skips_judge():
    runner = SequencedRunner([
        [sample("2"), sample("3"), sample("4")],  # solve: 分歧
        [plain("critic feedback")],                 # critic
        [sample("7"), sample("7"), sample("7")],  # refine: 达成一致,应早停跳过 judge
    ])
    config = DebateConfig()
    next_phase_fn, apply_fn = make_debate_functions(config)

    states = run_agent_loop(
        runner, [_question()], next_phase_fn=next_phase_fn, apply_fn=apply_fn, max_outer_rounds=config.max_rounds,
    )

    state = states[0]
    assert state.final_prediction == "7"
    assert state.round_idx == 3


def test_debate_solve_only_majority_vote_fallback():
    runner = SequencedRunner([
        [sample("2"), sample("2"), sample("3")],  # solve: 分歧,且是唯一阶段
    ])
    config = DebateConfig(max_rounds=1)
    next_phase_fn, apply_fn = make_debate_functions(config)

    states = run_agent_loop(
        runner, [_question()], next_phase_fn=next_phase_fn, apply_fn=apply_fn, max_outer_rounds=config.max_rounds,
    )

    state = states[0]
    assert state.final_prediction == "2"  # 2/3 多数票
    assert state.round_idx == 1


def test_debate_critic_only_falls_back_to_solver_majority():
    runner = SequencedRunner([
        [sample("2"), sample("2"), sample("3")],  # solve: 分歧
        [plain("critic feedback")],                 # critic: 是最后一个阶段
    ])
    config = DebateConfig(max_rounds=2)
    next_phase_fn, apply_fn = make_debate_functions(config)

    states = run_agent_loop(
        runner, [_question()], next_phase_fn=next_phase_fn, apply_fn=apply_fn, max_outer_rounds=config.max_rounds,
    )

    state = states[0]
    assert state.final_prediction == "2"
    assert state.final_finish_reasons == [None, None, None]
    assert state.round_idx == 2


# ---------------------------------------------------------------------------
# roles.make_pev_functions / _parse_strategies
# ---------------------------------------------------------------------------

def test_pev_verify_pass_first_try():
    runner = SequencedRunner([
        [plain("Strategy 1: algebra\nStrategy 2: substitution")],  # plan
        [sample("4"), sample("4")],                                    # execute
        [sample("4")],                                                 # verify: 通过
    ])
    config = PEVConfig()
    next_phase_fn, apply_fn = make_pev_functions(config)

    states = run_agent_loop(
        runner, [_question()], next_phase_fn=next_phase_fn, apply_fn=apply_fn,
        max_outer_rounds=(config.max_feedback_rounds + 1) * 3,
    )

    state = states[0]
    assert state.final_prediction == "4"
    assert state.round_idx == 3
    assert len(state.trace) == 1 + 2 + 1


def test_pev_verify_fail_then_feedback_then_pass():
    runner = SequencedRunner([
        [plain("Strategy 1: A\nStrategy 2: B")],          # plan 1
        [sample("3"), sample("3")],                          # execute 1
        [plain("both candidates FAIL, no boxed answer")],   # verify 1: 失败,无 boxed
        [plain("Strategy 1: C\nStrategy 2: D")],          # plan 2(带反馈)
        [sample("9"), sample("9")],                          # execute 2
        [sample("9")],                                       # verify 2: 通过
    ])
    config = PEVConfig()
    next_phase_fn, apply_fn = make_pev_functions(config)

    states = run_agent_loop(
        runner, [_question()], next_phase_fn=next_phase_fn, apply_fn=apply_fn,
        max_outer_rounds=(config.max_feedback_rounds + 1) * 3,
    )

    state = states[0]
    assert state.final_prediction == "9"
    assert state.round_idx == 6


def test_pev_verify_exhausted_falls_back_to_executor_majority_vote():
    config = PEVConfig(max_feedback_rounds=1)
    runner = SequencedRunner([
        [plain("Strategy 1: A\nStrategy 2: B")],   # plan 1
        [sample("3"), sample("3")],                   # execute 1
        [plain("both FAIL")],                          # verify 1: 失败,反馈预算还剩 1 次
        [plain("Strategy 1: C\nStrategy 2: D")],   # plan 2
        [sample("5"), sample("5")],                   # execute 2
        [plain("both FAIL again")],                    # verify 2: 失败,反馈预算耗尽
    ])
    next_phase_fn, apply_fn = make_pev_functions(config)

    states = run_agent_loop(
        runner, [_question()], next_phase_fn=next_phase_fn, apply_fn=apply_fn,
        max_outer_rounds=(config.max_feedback_rounds + 1) * 3,
    )

    state = states[0]
    assert state.final_prediction == "5"  # 退化为最后一次 executor 答案的多数票
    assert state.round_idx == 6


def test_parse_strategies_splits_on_numbered_prefix():
    text = "Strategy 1: Use algebra.\nStrategy 2: Use substitution."
    assert _parse_strategies(text, expected=2) == ["Use algebra.", "Use substitution."]


def test_parse_strategies_falls_back_to_whole_text_when_no_match():
    text = "Just think about it directly."
    assert _parse_strategies(text, expected=3) == ["Just think about it directly."]


# ---------------------------------------------------------------------------
# debate.run_debate / pev.run_pev -> QuestionRecord/RunResult 转换
# ---------------------------------------------------------------------------

def test_run_debate_builds_question_record():
    runner = SequencedRunner([
        [sample("6", output_tokens=20), sample("6", output_tokens=20), sample("6", output_tokens=20)],
    ])

    result = run_debate(runner, [_question(answer="6")], seed=42)

    assert result.method == "debate"
    record = result.records[0]
    assert record.prediction == "6"
    assert record.correct is True
    assert record.input_tokens == 30  # 3 步 * 10 input tokens
    assert record.output_tokens == 60  # 3 * 20
    assert record.total_tokens == 90
    assert record.rounds_used == 1
    assert len(record.agent_trace) == 3


def test_run_pev_builds_question_record():
    runner = SequencedRunner([
        [plain("Strategy 1: A\nStrategy 2: B", output_tokens=5)],
        [sample("4", output_tokens=15), sample("4", output_tokens=15)],
        [sample("4", output_tokens=15)],
    ])

    result = run_pev(runner, [_question(answer="4")], seed=42)

    assert result.method == "pev"
    record = result.records[0]
    assert record.prediction == "4"
    assert record.correct is True
    assert record.input_tokens == 40  # 4 步 * 10 input tokens
    assert record.output_tokens == 5 + 15 + 15 + 15
    assert record.total_tokens == record.input_tokens + record.output_tokens
    assert record.rounds_used == 3
    assert len(record.agent_trace) == 4
