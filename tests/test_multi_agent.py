import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.config import MDEConfig, DebateConfig, PEVConfig, RouterConfig  # noqa: E402
from src.model_runner import PromptResult, SampleResult  # noqa: E402
from src.multi_agent.mde import run_mde  # noqa: E402
from src.multi_agent.debate import run_debate  # noqa: E402
from src.multi_agent.orchestrator import PhaseStep, run_agent_loop  # noqa: E402
from src.multi_agent.pev import run_pev  # noqa: E402
from src.multi_agent.router import run_router  # noqa: E402
from src.multi_agent.roles import (  # noqa: E402
    _parse_strategies,
    _parse_category,
    _SPECIALIST_HINTS,
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
        self.calls.append({"temperature": temperature, "num_prompts": len(prompts), "prompts": list(prompts)})
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


def verify_pass(answer: str, output_tokens: int = 10, finish_reason: str = "stop") -> SampleResult:
    """PEV verifier 判通过的样本:必须同时有 Overall: PASS 判决和 boxed 答案,
    只有 boxed 不足以代表通过(这正是 2026-10-05 修复的 bug,见 LEARNING_NOTES.md)。
    """
    return SampleResult(
        text=f"Candidate checks out.\nOverall: PASS\n\\boxed{{{answer}}}",
        output_tokens=output_tokens,
        finish_reason=finish_reason,
    )


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
        [verify_pass("4")],                                             # verify: 通过
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
        [verify_pass("9")],                                  # verify 2: 通过
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


def test_pev_verify_boxed_without_pass_verdict_triggers_feedback_not_false_pass():
    """回归测试,对应 2026-10-05 修复的 bug:verifier 正文判 FAIL 但结尾仍习惯性
    挤出一个 \\boxed{} 时,不能被当成"验证通过"直接收敛,必须按 FAIL 处理触发反馈
    打回 planner。此前只检查有没有 boxed,导致 600 次真实运行里 fail 永远是 0。
    """
    runner = SequencedRunner([
        [plain("Strategy 1: A\nStrategy 2: B")],  # plan 1
        [sample("3"), sample("3")],                  # execute 1
        # verifier 正文判 FAIL,但仍然习惯性给了一个 boxed 答案:不能被当成通过
        [plain("Candidate 1: FAIL\nOverall: FAIL\n\\boxed{3}")],
        [plain("Strategy 1: C\nStrategy 2: D")],  # plan 2(应该被触发)
        [sample("9"), sample("9")],                  # execute 2
        [verify_pass("9")],                           # verify 2: 真正通过
    ])
    config = PEVConfig()
    next_phase_fn, apply_fn = make_pev_functions(config)

    states = run_agent_loop(
        runner, [_question()], next_phase_fn=next_phase_fn, apply_fn=apply_fn,
        max_outer_rounds=(config.max_feedback_rounds + 1) * 3,
    )

    state = states[0]
    assert state.final_prediction == "9"  # 没有被 verify 1 的假阳性 boxed 卡住
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
        [verify_pass("4", output_tokens=15)],
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


def test_parse_strategies_ignores_thinking_and_preamble_and_caps_count():
    text = ('<think>Strategy 1: draft\nStrategy 2: another draft</think>\n'
            'Here are the strategies.\nStrategy 1: A\nStrategy 2: B\nStrategy 3: C')
    assert _parse_strategies(text, 2) == ['A', 'B']
    assert _parse_strategies('draft Strategy 7: bad</think>\nStrategy 1: A\nStrategy 2: B', 2) == ['A', 'B']


def test_parse_strategies_uses_available_count_when_short():
    assert _parse_strategies('Introduction\nStrategy 1: A', 2) == ['A']
    assert _parse_strategies('<think>Strategy 1: draft</think>Use algebra', 2) == ['Use algebra']
    assert _parse_strategies('<think>Strategy 1: unfinished', 2) == ['']
    assert _parse_strategies('Strategy 1: A', 0) == []


def test_role_context_budget_preserves_head_tail_and_removes_thinking():
    from src.multi_agent.roles import (
        _bounded_context, build_executor_messages, build_verifier_messages,
        build_planner_feedback_messages, build_critic_messages,
        build_refine_messages, build_judge_messages,
    )

    long_text = '<think>' + 'draft' * 1000 + '</think>HEAD' + 'x' * 10000 + 'TAIL'
    bounded = _bounded_context([long_text] * 4, 400)
    assert sum(map(len, bounded)) == 400
    assert all(t.startswith('HEAD') and t.endswith('TAIL') for t in bounded)
    assert all('[truncated]' in t and 'draft' not in t for t in bounded)
    messages = [
        build_executor_messages('problem', long_text, 400),
        build_verifier_messages('problem', [long_text] * 2, [long_text] * 2, 400),
        build_planner_feedback_messages('problem', 2, [long_text] * 2, long_text, 400),
        build_critic_messages('problem', [long_text] * 3, 400),
        build_refine_messages('problem', long_text, long_text, 400),
        build_judge_messages('problem', [long_text] * 3, 400),
    ]
    for message in messages:
        content = message[0]['content']
        assert len(content) < 1600
        assert '[truncated]' in content and 'draft' not in content
        assert 'HEAD' in content and 'TAIL' in content


def test_thinking_override_reaches_all_roles_and_feedback():
    class ThinkingRunner(SequencedRunner):
        def build_prompt(self, messages, enable_thinking=None):
            assert enable_thinking is thinking
            return super().build_prompt(messages, enable_thinking)

    for thinking in (None, False, True):
        debate_runner = ThinkingRunner([
            [sample('2'), sample('3'), sample('4')], [plain('feedback')],
            [sample('2'), sample('3'), sample('4')], [sample('2')],
        ])
        run_debate(debate_runner, [_question()], seed=41, enable_thinking=thinking)
        pev_runner = ThinkingRunner([
            [plain('Strategy 1: A\nStrategy 2: B')], [sample('2'), sample('3')],
            [plain('FAIL')], [plain('Strategy 1: C\nStrategy 2: D')],
            [sample('2'), sample('2')], [verify_pass('2')],
        ])
        result = run_pev(pev_runner, [_question()], seed=41, enable_thinking=thinking)
        assert len(result.records[0].agent_trace) == 8


def test_pev_five_questions_have_ten_executor_prompts_with_polluted_plans():
    planner_text = '<think>' + 'Strategy 1: draft\n' * 20 + '</think>\nIntro\nStrategy 1: A\nStrategy 2: B\nStrategy 3: excess'
    runner = SequencedRunner([
        [plain(planner_text) for _ in range(5)],
        [sample('2') for _ in range(10)],
        [verify_pass('2') for _ in range(5)],
    ])
    result = run_pev(runner, [_question(question_id=f'q{i}') for i in range(5)], seed=41)
    assert [call['num_prompts'] for call in runner.calls] == [5, 10, 5]
    for record in result.records:
        assert sum(t['role'].startswith('executor_') for t in record.agent_trace) == 2
        assert record.agent_trace[0]['text'] == planner_text


# ---------------------------------------------------------------------------
# Router: 单路径分类、反馈和 QuestionRecord
# ---------------------------------------------------------------------------

def test_router_category_matches_known_hint_and_passes_first_try():
    runner = SequencedRunner([
        [plain('Category: Geometry')], [sample('2')], [plain('Overall: PASS')],
    ])
    question = {**_question(), 'subject': 'SECRET_GROUND_TRUTH'}
    record = run_router(runner, [question], seed=42).records[0]
    assert record.prediction == '2' and record.correct
    assert [t['role'] for t in record.agent_trace] == ['router', 'specialist', 'verifier']
    assert _SPECIALIST_HINTS['Geometry'] in runner.calls[1]['prompts'][0]
    assert all('SECRET_GROUND_TRUTH' not in c['prompts'][0] for c in runner.calls)
    assert [c['temperature'] for c in runner.calls] == [0.3, 0.6, 0.3]
    assert set(_SPECIALIST_HINTS) == {*RouterConfig().categories, '_fallback'}


def test_router_fail_then_retry_then_pass():
    runner = SequencedRunner([
        [plain('Category: Geometry')], [sample('3')],
        [plain('Arithmetic error. Overall: FAIL')], [sample('2')], [plain('Overall: PASS')],
    ])
    record = run_router(runner, [_question()], seed=42).records[0]
    assert record.prediction == '2' and record.correct
    assert [t['role'] for t in record.agent_trace] == ['router', 'specialist', 'verifier', 'specialist', 'verifier']
    assert [c['num_prompts'] for c in runner.calls] == [1] * 5
    retry = runner.calls[3]['prompts'][0]
    assert 'Your original solution' in retry and sample('3').text in retry
    assert 'Arithmetic error. Overall: FAIL' in retry
    assert _SPECIALIST_HINTS['Geometry'] in retry
    assert "Specialist's solution" in runner.calls[4]['prompts'][0]


def test_router_budget_exhausted_falls_back_to_specialist_answer():
    runner = SequencedRunner([
        [plain('Category: Algebra')], [sample('3')], [plain('Overall: FAIL')],
        [sample('5', finish_reason='length')], [plain('Overall: FAIL')],
    ])
    record = run_router(runner, [_question()], seed=42, config=RouterConfig(max_feedback_rounds=1)).records[0]
    assert record.prediction == '5'
    assert record.rounds_used == 5
    assert record.raw_outputs == [sample('5').text]
    assert record.finish_reasons == ['length']


def test_parse_category_handles_malformed_output():
    assert _parse_category('I choose Geometry.', RouterConfig().categories) is None


def test_parse_category_handles_hallucinated_unknown_category():
    assert _parse_category('Category: Trigonometry', RouterConfig().categories) is None


def test_router_verify_pass_without_extractable_specialist_answer_does_not_false_finish():
    runner = SequencedRunner([
        [plain('Category: Algebra')], [plain('Unable to solve this.')], [verify_pass('999')],
        [sample('2')], [plain('Overall: PASS')],
    ])
    record = run_router(runner, [_question()], seed=42).records[0]
    assert record.prediction == '2'
    assert record.rounds_used == 5 and len(runner.calls) == 5
    runner = SequencedRunner([
        [plain('Category: Algebra')], [plain('Unable to solve this.')], [verify_pass('999')],
    ])
    record = run_router(runner, [_question()], seed=42, config=RouterConfig(max_feedback_rounds=0)).records[0]
    assert record.prediction is None and not record.correct


def test_router_max_feedback_rounds_zero_never_retries():
    runner = SequencedRunner([
        [plain('Category: Algebra')], [sample('3')], [plain('Overall: FAIL')],
    ])
    record = run_router(runner, [_question()], seed=42, config=RouterConfig(max_feedback_rounds=0)).records[0]
    assert len(runner.calls) == 3
    assert record.prediction == '3' and record.rounds_used == 3


def test_router_max_outer_rounds_formula_generalizes_to_k2():
    runner = SequencedRunner([
        [plain('Category: Algebra')], [sample('3')], [plain('Overall: FAIL')],
        [sample('4')], [plain('Overall: FAIL')], [sample('2')], [plain('Overall: PASS')],
    ])
    record = run_router(runner, [_question()], seed=42, config=RouterConfig(max_feedback_rounds=2)).records[0]
    assert record.prediction == '2' and record.correct
    assert record.rounds_used == 7 and len(runner.calls) == 7
    assert [t['role'] for t in record.agent_trace] == ['router'] + ['specialist', 'verifier'] * 3


def test_thinking_override_reaches_router_roles():
    class ThinkingRunner(SequencedRunner):
        def build_prompt(self, messages, enable_thinking=None):
            assert enable_thinking is thinking
            return super().build_prompt(messages, enable_thinking)

    for thinking in (None, False, True):
        runner = ThinkingRunner([
            [plain('Category: Algebra')], [sample('3')], [plain('Overall: FAIL')],
            [sample('2')], [plain('Overall: PASS')],
        ])
        record = run_router(runner, [_question()], seed=42, enable_thinking=thinking).records[0]
        assert len(record.agent_trace) == 5


def test_run_router_builds_question_record():
    runner = SequencedRunner([
        [plain('Category: Algebra', output_tokens=5)], [sample('2', output_tokens=15)],
        [plain('Overall: PASS', output_tokens=8)],
    ])
    result = run_router(runner, [_question()], seed=42)
    record = result.records[0]
    assert result.method == record.method == 'router'
    assert result.seed == 42
    assert record.question_id == 'q1' and record.level == 5
    assert record.ground_truth == record.prediction == '2' and record.correct
    assert record.input_tokens == 30 and record.output_tokens == 28 and record.total_tokens == 58
    assert record.rounds_used == 3 and len(record.agent_trace) == 3
    assert [t['role'] for t in record.agent_trace] == ['router', 'specialist', 'verifier']
    assert record.raw_outputs == [sample('2').text] and record.finish_reasons == ['stop']


def test_parse_category_ignores_thinking_and_accepts_case_and_trailing_lines():
    categories = RouterConfig().categories
    assert _parse_category('<think>Category: Algebra</think>\nCategory: geometry  \nExtra text', categories) == 'Geometry'
    assert _parse_category('Category: Algebra</think>\nCategory: NUMBER THEORY', categories) == 'Number Theory'
    assert _parse_category('<think>Category: Algebra', categories) is None
    assert _parse_category('Category:\nGeometry', categories) is None


def test_router_unknown_category_uses_fallback_hint():
    runner = SequencedRunner([
        [plain('Category: Trigonometry')], [sample('2')], [verify_pass('999')],
    ])
    record = run_router(runner, [_question()], seed=42).records[0]
    assert record.prediction == '2'
    assert _SPECIALIST_HINTS['_fallback'] in runner.calls[1]['prompts'][0]


# ---------------------------------------------------------------------------
# MDE: 方法隔离、全一致早停与仲裁归因
# ---------------------------------------------------------------------------

def test_mde_early_stop_when_solvers_agree():
    runner = SequencedRunner([[sample('2'), sample('2'), sample('2')]])
    record = run_mde(runner, [_question()], seed=42).records[0]
    assert record.prediction == '2' and record.correct
    assert record.rounds_used == 1 and len(runner.calls) == 1
    assert [t['role'] for t in record.agent_trace] == ['solver_0', 'solver_1', 'solver_2']


def test_mde_partial_disagreement_still_goes_to_reconciler():
    runner = SequencedRunner([
        [sample('2'), sample('2'), sample('3')], [sample('3')],
    ])
    record = run_mde(runner, [_question()], seed=42).records[0]
    assert record.prediction == '3'
    assert record.rounds_used == 2 and len(runner.calls) == 2
    assert [t['role'] for t in record.agent_trace] == ['solver_0', 'solver_1', 'solver_2', 'reconciler']


def test_mde_reconciler_resolves_disagreement():
    solvers = [sample('2'), sample('3'), sample('4')]
    reconciler = sample('7', finish_reason='length')
    runner = SequencedRunner([solvers, [reconciler]])
    record = run_mde(runner, [_question(answer='7')], seed=42).records[0]
    assert record.prediction == '7' and record.correct
    assert record.raw_outputs == [reconciler.text]
    assert record.finish_reasons == ['length']
    prompt = runner.calls[1]['prompts'][0]
    for i, (name, _) in enumerate(MDEConfig().method_hints):
        assert f'--- Solver {i + 1} ({name}) solution ---\n{solvers[i].text}' in prompt
    assert 'independently re-derive at least one' in prompt
    assert 'instead of just trusting the majority' in prompt


def test_mde_reconciler_parse_failure_falls_back_to_solver_majority_vote():
    solvers = [sample('2', finish_reason='length'), sample('2'), sample('3')]
    failed = plain('Unable to reconcile these solutions.')
    runner = SequencedRunner([solvers, [failed]])
    record = run_mde(runner, [_question()], seed=42).records[0]
    assert record.prediction == '2' and record.correct
    assert record.raw_outputs == [s.text for s in solvers]
    assert record.finish_reasons == ['length', 'stop', 'stop']
    assert record.agent_trace[-1]['text'] == failed.text
    assert failed.text not in record.raw_outputs


def test_mde_each_solver_receives_correct_and_only_its_own_method_hint():
    config = MDEConfig()
    runner = SequencedRunner([[sample('2'), sample('2'), sample('2')]])
    run_mde(runner, [_question()], seed=42, config=config)
    assert runner.calls[0]['temperature'] == 0.3
    for i, (name, hint) in enumerate(config.method_hints):
        prompt = runner.calls[0]['prompts'][i]
        assert hint in prompt and f'Method: {name}' in prompt
        assert f'method only: {name}' in prompt
        for other_name, other_hint in config.method_hints:
            if other_name != name:
                assert other_name not in prompt and other_hint not in prompt


def test_mde_config_rejects_fewer_than_two_method_hints():
    for hints in ([], [('A', 'a')]):
        with pytest.raises(ValueError, match='至少需要 2 个方法'):
            MDEConfig(method_hints=hints)
    assert len(MDEConfig(method_hints=[('A', 'a'), ('B', 'b')]).method_hints) == 2


def test_thinking_override_reaches_mde_solver_and_reconciler_roles():
    class ThinkingRunner(SequencedRunner):
        def build_prompt(self, messages, enable_thinking=None):
            assert enable_thinking is thinking
            return super().build_prompt(messages, enable_thinking)

    for thinking in (None, False, True):
        runner = ThinkingRunner([
            [sample('2'), sample('3'), sample('4')], [sample('2')],
        ])
        record = run_mde(runner, [_question()], seed=42, enable_thinking=thinking).records[0]
        assert [t['role'] for t in record.agent_trace] == ['solver_0', 'solver_1', 'solver_2', 'reconciler']


def test_run_mde_builds_question_record():
    runner = SequencedRunner([
        [sample('2', output_tokens=20), sample('3', output_tokens=20), sample('4', output_tokens=20)],
        [sample('2', output_tokens=15)],
    ])
    result = run_mde(runner, [_question()], seed=42)
    record = result.records[0]
    assert result.method == record.method == 'mde' and result.seed == 42
    assert record.question_id == 'q1' and record.level == 5
    assert record.ground_truth == record.prediction == '2' and record.correct
    assert record.input_tokens == 40 and record.output_tokens == 75 and record.total_tokens == 115
    assert record.rounds_used == 2
    assert [t['role'] for t in record.agent_trace] == ['solver_0', 'solver_1', 'solver_2', 'reconciler']
    assert record.raw_outputs == [sample('2').text] and record.finish_reasons == ['stop']
    assert [c['temperature'] for c in runner.calls] == [0.3, 0.3]
