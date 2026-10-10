"""Debate、PEV、Router 与 MDE 四套多智能体流程的角色 prompt 构造 + next_phase_fn/apply_fn 状态机。

四套流程共用 orchestrator.run_agent_loop,区别只在这里:各自的 prompt 怎么组、
下一步该发什么、什么时候算收敛/该早停/该打回上一角色重来,全部通过
next_phase_fn/apply_fn 两个闭包下放给调度器——调度器本身不理解 "solver"/
"planner" 这些角色语义,只负责按采样参数分组 batch 调用。

四套流程都遵循同一个"当前这一步是什么阶段"的记录约定:next_phase_fn 决定这一步
发什么之后,把阶段名写进 state.scratch,供 apply_fn 处理这一步返回结果时读取;
apply_fn 处理完之后再把"下一步该进入哪个阶段"写回 state.scratch,供下一次
next_phase_fn 调用读取。
"""
from __future__ import annotations

import re

from ..config import DebateConfig, FINAL_ANSWER_INSTRUCTION, MDEConfig, ModelConfig, PEVConfig, RouterConfig
from ..extract_answer import extract_answer, is_equiv
from ..methods.self_consistency import majority_vote
from .orchestrator import AgentState, ApplyFn, NextPhaseFn, PhaseStep

_MAX_TOKENS = ModelConfig().max_tokens


def _formal_answer(text: str) -> str:
    # chat template 可能已消费开标签,因此只有闭标签时也应排除思考草稿。
    parts = re.split(r"</think\s*>", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[-1].strip()
    if re.search(r"<think\s*>", text, flags=re.IGNORECASE):
        return ""  # 未闭合的草稿没有正式策略,不能把其中的标记当成候选。
    return text.strip()


def _bounded_context(texts: list[str], max_chars: int = 6000) -> list[str]:
    # 所有候选共享预算,避免候选数增大时总长度再次失控。6000 字符即使按
    # UTF-8 最坏的每字符 4 token 估计也留有题目和模板空间;头尾兼顾思路和最终答案。
    if max_chars < 1:
        raise ValueError("max_context_chars must be positive")
    budget = max_chars // max(1, len(texts))
    marker = "\n...[truncated]...\n"
    result = []
    for text in texts:
        text = _formal_answer(text)
        if len(text) > budget:
            if budget <= len(marker):
                text = text[:budget]
            else:
                keep = budget - len(marker)
                head = (keep + 1) // 2
                tail = keep // 2
                text = text[:head] + marker + (text[-tail:] if tail else "")
        result.append(text)
    return result


def _extract(sample) -> str | None:
    # 被截断的采样禁用兜底提取规则,理由与 methods/base.py 的同名处理一致
    # (截断点附近的数字不代表模型的真实结论)。
    return extract_answer(sample.text, allow_fallback=sample.finish_reason != "length")


def _all_agree(answers: list[str | None]) -> bool:
    if not answers or any(a is None for a in answers):
        return False
    first = answers[0]
    return all(is_equiv(a, first) for a in answers[1:])


def _record_trace(state: AgentState, steps: list[PhaseStep], outputs) -> list[str]:
    """把这一步所有角色的输出写入 state.trace,累加输入 token,返回各角色的原文列表。"""
    texts = []
    total_input = 0
    for step, out in zip(steps, outputs):
        sample = out.samples[0]
        state.trace.append({
            "round": state.round_idx,
            "role": step.role,
            "text": sample.text,
            "output_tokens": sample.output_tokens,
            "finish_reason": sample.finish_reason,
        })
        texts.append(sample.text)
        total_input += out.input_tokens
    state.scratch["total_input_tokens"] = state.scratch.get("total_input_tokens", 0) + total_input
    return texts


def _finish(
    state: AgentState,
    prediction: str | None,
    raw_outputs: list[str],
    finish_reasons: list[str],
) -> None:
    state.final_prediction = prediction
    state.final_raw_outputs = raw_outputs
    state.final_finish_reasons = finish_reasons
    state.done = True


# ---------------------------------------------------------------------------
# Debate: Solver(Proposer) -> Critic -> Refine -> Judge
# ---------------------------------------------------------------------------

def build_solver_messages(problem: str) -> list[dict]:
    content = (
        "You are one of several independent solvers tackling the same math problem.\n"
        "Work through it step by step and show your reasoning clearly.\n\n"
        f"Problem:\n{problem}"
        f"{FINAL_ANSWER_INSTRUCTION}"
    )
    return [{"role": "user", "content": content}]


def build_critic_messages(problem: str, solver_texts: list[str], max_context_chars: int = 6000) -> list[dict]:
    solver_texts = _bounded_context(solver_texts, max_context_chars)
    solutions = "\n\n".join(
        f"--- Solver {i + 1}'s solution ---\n{text}" for i, text in enumerate(solver_texts)
    )
    content = (
        "Several solvers independently attempted the math problem below and reached "
        "different final answers. Carefully check each solution for concrete errors "
        "(arithmetic mistakes, invalid steps, unjustified assumptions). For each "
        "solution, point out specifically what is wrong (if anything), then state "
        "which final answer you believe is actually correct and why. Verify the "
        "reasoning yourself instead of just trusting the majority.\n\n"
        f"Problem:\n{problem}\n\n{solutions}"
    )
    return [{"role": "user", "content": content}]


def build_refine_messages(problem: str, own_text: str, critic_text: str, max_context_chars: int = 6000) -> list[dict]:
    own_text, critic_text = _bounded_context([own_text, critic_text], max_context_chars)
    content = (
        "You previously solved the math problem below. A critic has since reviewed "
        "multiple solvers' solutions, including yours, and left feedback. Reconsider "
        "your solution in light of the feedback: fix it if the critic found a genuine "
        "error, but keep your original answer if you are confident it is correct and "
        "the critic's objection does not actually hold up. Give your final, refined "
        "solution.\n\n"
        f"Problem:\n{problem}\n\n"
        f"--- Your original solution ---\n{own_text}\n\n"
        f"--- Critic's feedback ---\n{critic_text}"
        f"{FINAL_ANSWER_INSTRUCTION}"
    )
    return [{"role": "user", "content": content}]


def build_judge_messages(problem: str, refined_texts: list[str], max_context_chars: int = 6000) -> list[dict]:
    refined_texts = _bounded_context(refined_texts, max_context_chars)
    solutions = "\n\n".join(
        f"--- Refined solution {i + 1} ---\n{text}" for i, text in enumerate(refined_texts)
    )
    content = (
        "Multiple solvers refined their solutions to the math problem below after a "
        "round of peer criticism, but still disagree on the final answer. Act as a "
        "judge: independently work through the problem yourself and determine which "
        "final answer is correct.\n\n"
        f"Problem:\n{problem}\n\n{solutions}"
        f"{FINAL_ANSWER_INSTRUCTION}"
    )
    return [{"role": "user", "content": content}]


def make_debate_functions(config: DebateConfig, enable_thinking: bool | None = None) -> tuple[NextPhaseFn, ApplyFn]:
    """构造 Debate 流程的 next_phase_fn/apply_fn。

    固定阶段顺序 solve -> critic -> refine -> judge,长度由 config.max_rounds 截断
    (默认 4,即四阶段全用上)。solve/refine 后若各方答案已一致就早停,跳过剩余阶段;
    到达阶段序列末尾仍不一致时,用当前已有的答案做多数投票兜底,保证一定能收敛。
    """
    phase_order = ["solve", "critic", "refine", "judge"][: config.max_rounds]

    def next_phase_fn(state: AgentState) -> list[PhaseStep] | None:
        idx = state.scratch.get("phase_idx", 0)
        if idx >= len(phase_order):
            return None  # 安全阀:正常流程下 apply_fn 会在最后一个阶段就地 done,不会走到这里
        phase = phase_order[idx]
        state.scratch["phase"] = phase
        problem = state.question["problem"]

        if phase == "solve":
            return [
                PhaseStep(
                    role=f"solver_{i}",
                    messages=build_solver_messages(problem),
                    temperature=config.solver_temperature,
                    max_tokens=_MAX_TOKENS,
                    repetition_penalty=config.repetition_penalty,
                    enable_thinking=enable_thinking,
                )
                for i in range(config.num_solvers)
            ]

        if phase == "critic":
            return [
                PhaseStep(
                    role="critic",
                    messages=build_critic_messages(problem, state.scratch["solver_texts"], config.max_context_chars),
                    temperature=config.critic_temperature,
                    max_tokens=_MAX_TOKENS,
                    repetition_penalty=config.repetition_penalty,
                    enable_thinking=enable_thinking,
                )
            ]

        if phase == "refine":
            critic_text = state.scratch["critic_text"]
            return [
                PhaseStep(
                    role=f"refine_{i}",
                    messages=build_refine_messages(problem, text, critic_text, config.max_context_chars),
                    temperature=config.refine_temperature,
                    max_tokens=_MAX_TOKENS,
                    repetition_penalty=config.repetition_penalty,
                    enable_thinking=enable_thinking,
                )
                for i, text in enumerate(state.scratch["solver_texts"])
            ]

        # phase == "judge"
        return [
            PhaseStep(
                role="judge",
                messages=build_judge_messages(problem, state.scratch["refined_texts"], config.max_context_chars),
                temperature=config.judge_temperature,
                max_tokens=_MAX_TOKENS,
                repetition_penalty=config.repetition_penalty,
                enable_thinking=enable_thinking,
            )
        ]

    def apply_fn(state: AgentState, steps: list[PhaseStep], outputs) -> None:
        phase = state.scratch["phase"]
        idx = state.scratch.get("phase_idx", 0)
        is_last_phase = idx == len(phase_order) - 1
        texts = _record_trace(state, steps, outputs)
        finish_reasons = [out.samples[0].finish_reason for out in outputs]

        if phase == "solve":
            answers = [_extract(out.samples[0]) for out in outputs]
            state.scratch["solver_texts"] = texts
            state.scratch["solver_answers"] = answers
            if _all_agree(answers):
                _finish(state, answers[0], texts, finish_reasons)
            elif is_last_phase:
                winner, _ = majority_vote(answers)
                _finish(state, winner, texts, finish_reasons)
            else:
                state.scratch["phase_idx"] = idx + 1
            return

        if phase == "critic":
            state.scratch["critic_text"] = texts[0]
            if is_last_phase:
                # 只到critic 就截止(极小 max_rounds 的边界情况):critic 本身不产出答案,
                # 退回用 solve 阶段的多数投票兜底。
                winner, _ = majority_vote(state.scratch["solver_answers"])
                _finish(state, winner, state.scratch["solver_texts"],
                         [None] * len(state.scratch["solver_texts"]))
            else:
                state.scratch["phase_idx"] = idx + 1
            return

        if phase == "refine":
            answers = [_extract(out.samples[0]) for out in outputs]
            state.scratch["refined_texts"] = texts
            state.scratch["refined_answers"] = answers
            if _all_agree(answers):
                _finish(state, answers[0], texts, finish_reasons)
            elif is_last_phase:
                winner, _ = majority_vote(answers)
                _finish(state, winner, texts, finish_reasons)
            else:
                state.scratch["phase_idx"] = idx + 1
            return

        # phase == "judge"
        judge_answer = _extract(outputs[0].samples[0])
        if judge_answer is None:
            judge_answer, _ = majority_vote(state.scratch["refined_answers"])
        _finish(state, judge_answer, texts, finish_reasons)

    return next_phase_fn, apply_fn


# ---------------------------------------------------------------------------
# PEV: Planner -> Executor -> Verifier (verifier 判 FAIL 打回 planner,带反馈)
# ---------------------------------------------------------------------------

_STRATEGY_SPLIT_RE = re.compile(r"Strategy\s*\d+\s*[:.\-]\s*", re.IGNORECASE)


def build_planner_messages(problem: str, num_strategies: int) -> list[dict]:
    content = (
        f"Propose {num_strategies} genuinely different high-level strategies for "
        "solving the math problem below. For each strategy, describe the approach "
        "and key idea in a few sentences only - do NOT carry out any computation yet.\n\n"
        f"Problem:\n{problem}\n\n"
        "Format your answer exactly as:\n"
        "Strategy 1: <description>\n"
        "Strategy 2: <description>\n"
        "(and so on)"
    )
    return [{"role": "user", "content": content}]


def build_planner_feedback_messages(
    problem: str,
    num_strategies: int,
    executor_texts: list[str],
    verifier_feedback: str,
    max_context_chars: int = 6000,
) -> list[dict]:
    *executor_texts, verifier_feedback = _bounded_context(
        [*executor_texts, verifier_feedback], max_context_chars
    )
    attempts = "\n\n".join(
        f"--- Previous attempt {i + 1} ---\n{text}" for i, text in enumerate(executor_texts)
    )
    content = (
        f"Propose {num_strategies} new or revised high-level strategies for solving "
        "the math problem below. The previous attempts failed verification; use the "
        "verifier's feedback to avoid repeating the same mistakes. Describe each "
        "strategy's approach and key idea only - do NOT carry out any computation yet.\n\n"
        f"Problem:\n{problem}\n\n{attempts}\n\n"
        f"--- Verifier's feedback ---\n{verifier_feedback}\n\n"
        "Format your answer exactly as:\n"
        "Strategy 1: <description>\n"
        "Strategy 2: <description>\n"
        "(and so on)"
    )
    return [{"role": "user", "content": content}]


def build_executor_messages(problem: str, strategy: str, max_context_chars: int = 6000) -> list[dict]:
    strategy, = _bounded_context([strategy], max_context_chars)
    content = (
        "Carry out the following solution strategy fully and carefully for the math "
        "problem below. Show all computation steps.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Strategy to follow:\n{strategy}"
        f"{FINAL_ANSWER_INSTRUCTION}"
    )
    return [{"role": "user", "content": content}]


# PEV 和 Router 共用判决格式;修改时两边 prompt 与解析逻辑都要检查,避免不一致。
_PASS_FAIL_VERDICT_INSTRUCTION = (
    "End your reply with a line in exactly this format: 'Overall: PASS' or "
    "'Overall: FAIL'."
)


_VERIFIER_VERDICT_RE = re.compile(r"overall\s*:\s*(pass|fail)", re.IGNORECASE)


def _verifier_passed(text: str) -> bool:
    """只信任 verifier 明确给出的 "Overall: PASS/FAIL" 判决,不能只看有没有
    \\boxed{}。LLM 有很强的"总要给个答案"倾向,即使正文把所有候选都判成 FAIL,
    结尾仍可能习惯性挤出一个 boxed 答案;如果只拿 boxed 存在与否当"验证通过"的
    代理指标,这类假阳性会让反馈重试机制形同虚设(实测 600 次运行里 fail 永远是
    0,详见 LEARNING_NOTES.md 2026-10-05 记录)。取最后一次出现的 Overall 判决,
    避免逐候选的 PASS/FAIL 叙述干扰最终结论。
    """
    matches = _VERIFIER_VERDICT_RE.findall(text)
    return bool(matches) and matches[-1].lower() == "pass"


def build_verifier_messages(
    problem: str,
    strategies: list[str],
    executor_texts: list[str],
    max_context_chars: int = 6000,
) -> list[dict]:
    bounded = _bounded_context([*strategies, *executor_texts], max_context_chars)
    strategies, executor_texts = bounded[:len(strategies)], bounded[len(strategies):]
    candidates = "\n\n".join(
        f"--- Candidate {i + 1} (strategy: {strat}) ---\n{text}"
        for i, (strat, text) in enumerate(zip(strategies, executor_texts))
    )
    content = (
        "Below are candidate solutions to the math problem, each following a "
        "different strategy. Verify each one mechanically - redo the key computation "
        "or substitute the claimed answer back into the problem - rather than just "
        "re-reading the narrative. For each candidate, state PASS or FAIL with a "
        "brief mechanical justification.\n\n"
        # PEV 和 Router 共用判决格式;修改时也检查 Light Verifier 与判决解析逻辑。
        f"{_PASS_FAIL_VERDICT_INSTRUCTION} "
        "Use PASS if at least one candidate is verified correct, or FAIL if none are "
        "reliable. If PASS, follow it with the verified final answer in \\boxed{}. "
        "If FAIL, do NOT include \\boxed{} - instead clearly explain what went wrong "
        "so the strategies can be revised.\n\n"
        f"Problem:\n{problem}\n\n{candidates}"
    )
    return [{"role": "user", "content": content}]


def _parse_strategies(text: str, expected: int) -> list[str]:
    """只取正式回答中的策略;不足时用实际数量,超额时裁剪,无标记时用正式全文。"""
    if expected <= 0:
        return []
    text = _formal_answer(text)
    matches = list(_STRATEGY_SPLIT_RE.finditer(text))
    parts = [text[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
             for i, m in enumerate(matches)]
    parts = [part for part in parts if part]
    return (parts or [text])[:expected]


def make_pev_functions(config: PEVConfig, enable_thinking: bool | None = None) -> tuple[NextPhaseFn, ApplyFn]:
    """构造 PEV 流程的 next_phase_fn/apply_fn。

    stage_to_run 在 plan -> execute -> verify 之间循环;verify 阶段必须同时满足
    "verifier 给出 Overall: PASS 判决"和"解析得到 \\boxed{}"才算通过并收敛(见
    `_verifier_passed`,只看 boxed 存在与否会被模型"总要给个答案"的倾向绕过),
    否则在 feedback_used < config.max_feedback_rounds 时带着 verifier 的反馈打回
    planner 重新出策略,用完反馈轮次预算后退化为对当前 executor 答案做多数投票
    兜底,保证一定能收敛。
    """

    def next_phase_fn(state: AgentState) -> list[PhaseStep] | None:
        stage = state.scratch.get("stage_to_run", "plan")
        state.scratch["current_stage"] = stage
        problem = state.question["problem"]

        if stage == "plan":
            feedback = state.scratch.get("verifier_feedback")
            if feedback is None:
                messages = build_planner_messages(problem, config.num_strategies)
            else:
                messages = build_planner_feedback_messages(
                    problem, config.num_strategies, state.scratch["executor_texts"], feedback,
                    config.max_context_chars
                )
            return [
                PhaseStep(
                    role="planner",
                    messages=messages,
                    temperature=config.planner_temperature,
                    max_tokens=_MAX_TOKENS,
                    repetition_penalty=config.repetition_penalty,
                    enable_thinking=enable_thinking,
                )
            ]

        if stage == "execute":
            strategies = state.scratch["strategies"]
            return [
                PhaseStep(
                    role=f"executor_{i}",
                    messages=build_executor_messages(problem, strategy, config.max_context_chars),
                    temperature=config.executor_temperature,
                    max_tokens=_MAX_TOKENS,
                    repetition_penalty=config.repetition_penalty,
                    enable_thinking=enable_thinking,
                )
                for i, strategy in enumerate(strategies)
            ]

        # stage == "verify"
        return [
            PhaseStep(
                role="verifier",
                messages=build_verifier_messages(
                    problem, state.scratch["strategies"], state.scratch["executor_texts"],
                    config.max_context_chars
                ),
                temperature=config.verifier_temperature,
                max_tokens=_MAX_TOKENS,
                repetition_penalty=config.repetition_penalty,
                enable_thinking=enable_thinking,
            )
        ]

    def apply_fn(state: AgentState, steps: list[PhaseStep], outputs) -> None:
        stage = state.scratch["current_stage"]
        texts = _record_trace(state, steps, outputs)

        if stage == "plan":
            state.scratch["strategies"] = _parse_strategies(texts[0], config.num_strategies)
            state.scratch["stage_to_run"] = "execute"
            return

        if stage == "execute":
            state.scratch["executor_texts"] = texts
            state.scratch["executor_answers"] = [_extract(out.samples[0]) for out in outputs]
            state.scratch["executor_finish_reasons"] = [out.samples[0].finish_reason for out in outputs]
            state.scratch["stage_to_run"] = "verify"
            return

        # stage == "verify"
        sample = outputs[0].samples[0]
        verifier_text = texts[0]
        verified_answer = _extract(sample)

        if verified_answer is not None and _verifier_passed(verifier_text):
            _finish(state, verified_answer, [verifier_text], [sample.finish_reason])
            return

        feedback_used = state.scratch.get("feedback_used", 0)
        if feedback_used < config.max_feedback_rounds:
            state.scratch["feedback_used"] = feedback_used + 1
            state.scratch["verifier_feedback"] = verifier_text
            state.scratch["stage_to_run"] = "plan"
        else:
            winner, _ = majority_vote(state.scratch["executor_answers"])
            _finish(
                state,
                winner,
                state.scratch["executor_texts"],
                state.scratch["executor_finish_reasons"],
            )

    return next_phase_fn, apply_fn


# ---------------------------------------------------------------------------
# Router: Router -> Domain Specialist -> Light Verifier (workflow 范式,拓展D)
# ---------------------------------------------------------------------------

_SPECIALIST_HINTS: dict[str, str] = {
    "Algebra": "Consider equations, substitution, factoring, and algebraic identities.",
    "Intermediate Algebra": "Consider polynomial structure, inequalities, complex numbers, and careful domain restrictions.",
    "Prealgebra": "Consider arithmetic, fractions, ratios, percentages, and unit conversions.",
    "Precalculus": "Consider function identities, trigonometry, logarithms, and transformations of graphs.",
    "Number Theory": "Consider modular arithmetic, divisibility, and prime factorization.",
    "Counting & Probability": "Consider systematic counting, complementary events, and avoiding double counting.",
    "Geometry": "Consider similar triangles, the Pythagorean theorem, angle and area relations, or coordinates.",
    "_fallback": "Identify the relevant mathematical tools, break the problem into steps, and check all assumptions.",
}

_ROUTER_CATEGORY_RE = re.compile(r"^[ \t]*Category:[ \t]*([^\r\n]+)", re.IGNORECASE | re.MULTILINE)


def build_router_messages(problem: str, categories: list[str]) -> list[dict]:
    content = (
        "Classify the problem solely from its text. Do not assume a subject; base "
        "your classification on the mathematical tools actually required by the problem.\n"
        "Choose exactly one of these categories (use its exact spelling):\n"
        + "\n".join(categories)
        + f"\n\nProblem:\n{problem}\n\n"
        "Output only one line, exactly in this format: Category: <one category name>"
    )
    return [{"role": "user", "content": content}]


def _parse_category(text: str, categories: list[str]) -> str | None:
    match = _ROUTER_CATEGORY_RE.search(_formal_answer(text))
    if match is None:
        return None
    category = match.group(1).strip().casefold()
    return next((name for name in categories if name.casefold() == category), None)


def build_specialist_messages(problem: str, hint: str) -> list[dict]:
    content = (
        "You are a domain specialist solving a math problem.\n"
        f"Methodological hint: {hint}\n"
        "Work through it step by step and show your reasoning clearly.\n\n"
        f"Problem:\n{problem}"
        f"{FINAL_ANSWER_INSTRUCTION}"
    )
    return [{"role": "user", "content": content}]


def build_specialist_retry_messages(
    problem: str, hint: str, own_text: str, verifier_feedback: str,
    max_context_chars: int = 6000,
) -> list[dict]:
    own_text, verifier_feedback = _bounded_context([own_text, verifier_feedback], max_context_chars)
    content = (
        "You previously solved the math problem below. Reconsider your solution "
        "using the verifier's feedback. Fix genuine errors, but retain correct "
        "reasoning if the objection does not hold up. Give a revised solution step by step.\n"
        f"Methodological hint: {hint}\n\n"
        f"Problem:\n{problem}\n\n"
        f"--- Your original solution ---\n{own_text}\n\n"
        f"--- Verifier's feedback ---\n{verifier_feedback}"
        f"{FINAL_ANSWER_INSTRUCTION}"
    )
    return [{"role": "user", "content": content}]


def build_light_verifier_messages(
    problem: str, specialist_text: str, max_context_chars: int = 6000,
) -> list[dict]:
    specialist_text, = _bounded_context([specialist_text], max_context_chars)
    content = (
        "Verify this solution mechanically: redo the key computations or substitute "
        "the claimed answer back into the problem; do not just re-read the narrative. "
        "Judge whether the answer already given is correct. You do not need to "
        "provide the final answer again; do NOT output a boxed answer. "
        "If FAIL, briefly explain the errors so the specialist can revise the solution.\n\n"
        f"Problem:\n{problem}\n\n"
        f"--- Specialist's solution ---\n{specialist_text}\n\n"
        # PEV 和 Router 共用判决格式;修改时也检查 PEV prompt 与判决解析逻辑。
        f"{_PASS_FAIL_VERDICT_INSTRUCTION}"
    )
    return [{"role": "user", "content": content}]


def make_router_functions(config: RouterConfig, enable_thinking: bool | None = None) -> tuple[NextPhaseFn, ApplyFn]:
    """单路径验证必须同时有答案和 PASS;重试保持原分类,只打回 Specialist。"""
    def next_phase_fn(state: AgentState) -> list[PhaseStep] | None:
        stage = state.scratch.get("stage_to_run", "route")
        state.scratch["current_stage"] = stage
        problem = state.question["problem"]
        if stage == "route":
            role = "router"
            messages = build_router_messages(problem, config.categories)
            temperature = config.router_temperature
        elif stage == "specialize":
            role = "specialist"
            hint = _SPECIALIST_HINTS.get(state.scratch.get("category"), _SPECIALIST_HINTS["_fallback"])
            feedback = state.scratch.get("verifier_feedback")
            if feedback is None:
                messages = build_specialist_messages(problem, hint)
            else:
                messages = build_specialist_retry_messages(
                    problem, hint, state.scratch["specialist_text"], feedback, config.max_context_chars,
                )
            temperature = config.specialist_temperature
        else:
            role = "verifier"
            messages = build_light_verifier_messages(
                problem, state.scratch["specialist_text"], config.max_context_chars,
            )
            temperature = config.verifier_temperature
        return [PhaseStep(
            role=role, messages=messages, temperature=temperature,
            max_tokens=_MAX_TOKENS, repetition_penalty=config.repetition_penalty,
            enable_thinking=enable_thinking,
        )]

    def apply_fn(state: AgentState, steps: list[PhaseStep], outputs) -> None:
        stage = state.scratch["current_stage"]
        texts = _record_trace(state, steps, outputs)
        if stage == "route":
            state.scratch["category"] = _parse_category(texts[0], config.categories)
            state.scratch["stage_to_run"] = "specialize"
            return
        if stage == "specialize":
            state.scratch["specialist_text"] = texts[0]
            state.scratch["specialist_answer"] = _extract(outputs[0].samples[0])
            state.scratch["specialist_finish_reason"] = outputs[0].samples[0].finish_reason
            state.scratch["stage_to_run"] = "verify"
            return

        specialist_answer = state.scratch.get("specialist_answer")
        verifier_text = texts[0]
        if specialist_answer is not None and _verifier_passed(verifier_text):
            _finish(state, specialist_answer, [state.scratch["specialist_text"]],
                    [state.scratch["specialist_finish_reason"]])
            return
        feedback_used = state.scratch.get("feedback_used", 0)
        if feedback_used < config.max_feedback_rounds:
            state.scratch["feedback_used"] = feedback_used + 1
            state.scratch["verifier_feedback"] = verifier_text
            state.scratch["stage_to_run"] = "specialize"
        else:
            # 单链没有投票依据,保留最后一次 Specialist 输出及其真实结束原因。
            _finish(state, specialist_answer, [state.scratch["specialist_text"]],
                    [state.scratch["specialist_finish_reason"]])

    return next_phase_fn, apply_fn


# ---------------------------------------------------------------------------
# MDE: Method-Diversity Ensemble + Reconciler (拓展C,多样性来自方法而非采样)
# ---------------------------------------------------------------------------

def build_method_solver_messages(problem: str, method_name: str, method_hint: str) -> list[dict]:
    content = (
        f"You must solve this problem using the following method only: {method_name}.\n"
        f"{method_hint}\n"
        "Do not switch to a different class of technique even if another approach "
        "seems easier or more natural - push the assigned method as far as it goes.\n"
        f'Begin your solution with the line "Method: {method_name}".\n\n'
        f"Problem:\n{problem}"
        f"{FINAL_ANSWER_INSTRUCTION}"
    )
    return [{"role": "user", "content": content}]


def build_reconciler_messages(
    problem: str, solver_texts: list[str], method_names: list[str],
    max_context_chars: int = 6000,
) -> list[dict]:
    solver_texts = _bounded_context(solver_texts, max_context_chars)
    solutions = "\n\n".join(
        f"--- Solver {i + 1} ({name}) solution ---\n{text}"
        for i, (name, text) in enumerate(zip(method_names, solver_texts))
    )
    content = (
        "Reconcile the solutions obtained using different methods below. Check whether "
        "the methods converge on the same answer and identify concrete errors. "
        "When the answers disagree, you must independently re-derive at least one "
        "of the disagreeing methods before giving your final answer. Show this "
        "independent derivation and explain how it resolves the disagreement. "
        "Verify the reasoning yourself instead of just trusting the majority; "
        "do not choose an answer solely because more solvers support it.\n\n"
        f"Problem:\n{problem}\n\n{solutions}"
        f"{FINAL_ANSWER_INSTRUCTION}"
    )
    return [{"role": "user", "content": content}]


def make_mde_functions(config: MDEConfig, enable_thinking: bool | None = None) -> tuple[NextPhaseFn, ApplyFn]:
    def next_phase_fn(state: AgentState) -> list[PhaseStep] | None:
        stage = state.scratch.get("stage_to_run", "solve")
        state.scratch["current_stage"] = stage
        problem = state.question["problem"]
        if stage == "solve":
            return [
                PhaseStep(
                    role=f"solver_{i}",
                    messages=build_method_solver_messages(problem, name, hint),
                    temperature=config.solver_temperature,
                    max_tokens=_MAX_TOKENS,
                    repetition_penalty=config.repetition_penalty,
                    enable_thinking=enable_thinking,
                )
                for i, (name, hint) in enumerate(config.method_hints)
            ]
        return [
            PhaseStep(
                role="reconciler",
                messages=build_reconciler_messages(
                    problem, state.scratch["solver_texts"],
                    [name for name, _ in config.method_hints], config.max_context_chars,
                ),
                temperature=config.reconciler_temperature,
                max_tokens=_MAX_TOKENS,
                repetition_penalty=config.repetition_penalty,
                enable_thinking=enable_thinking,
            )
        ]

    def apply_fn(state: AgentState, steps: list[PhaseStep], outputs) -> None:
        stage = state.scratch["current_stage"]
        texts = _record_trace(state, steps, outputs)
        finish_reasons = [out.samples[0].finish_reason for out in outputs]
        if stage == "solve":
            answers = [_extract(out.samples[0]) for out in outputs]
            state.scratch["solver_texts"] = texts
            state.scratch["solver_answers"] = answers
            state.scratch["solver_finish_reasons"] = finish_reasons
            if _all_agree(answers):
                _finish(state, answers[0], texts, finish_reasons)
            else:
                state.scratch["stage_to_run"] = "reconcile"
            return

        answer = _extract(outputs[0].samples[0])
        if answer is not None:
            _finish(state, answer, texts, finish_reasons)
        else:
            # 投票结果必须能追溯到 solver 原文,不能归因到解析失败的仲裁文本。
            winner, _ = majority_vote(state.scratch["solver_answers"])
            _finish(state, winner, state.scratch["solver_texts"], state.scratch["solver_finish_reasons"])

    return next_phase_fn, apply_fn
