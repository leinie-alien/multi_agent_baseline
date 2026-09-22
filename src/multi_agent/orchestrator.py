"""通用多智能体轮调度器。Debate 和 PEV 共用这一份调度器,区别只在于各自传入的
next_phase_fn/apply_fn(角色 prompt 怎么组、什么时候算收敛/该早停)。

设计动机:Debate 的"轮"其实是固定的 solve -> critic -> refine -> judge 四个阶段,
每个阶段自己的温度不同;PEV 是 planner -> executor -> verifier,verifier 判 FAIL
后打回 planner 重来,轮数不固定。两者共同点是"每一步只对当前还没做完的题目发一次
batch generate 调用",不同点全部在"这一步该发什么 prompt、发完之后算不算做完"里。
所以调度器本身只负责 batch 调用和 done 判断循环,角色逻辑完全下放给调用方。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..model_runner import ModelRunner, PromptResult


@dataclass
class AgentState:
    """单个题目在多智能体流程中的状态,在多轮之间存活、被 next_phase_fn/apply_fn 读写。

    question: 原始题目 dict(problem/answer/level/question_id)。
    round_idx: 已完成的步数(从 0 开始,每次 apply_fn 处理完一步后 +1)。写入
        QuestionRecord.rounds_used 时直接用这个值。
    done: True 时该题不再进入下一轮 batch。
    scratch: next_phase_fn/apply_fn 自己维护的中间结果(比如 Debate 各 solver 当前
        答案、PEV 当前策略),orchestrator 本身不解释这个字段的内容。
    trace: 每步每个角色的输出记录,元素格式直接对应 QuestionRecord.agent_trace:
        {"round": int, "role": str, "text": str, "output_tokens": int, "finish_reason": str}
    final_prediction/final_raw_outputs/final_finish_reasons: apply_fn 判定 done=True
        时填好,供调用方(debate.py/pev.py)转成 QuestionRecord 用。
    """
    question: dict
    round_idx: int = 0
    done: bool = False
    scratch: dict[str, Any] = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)
    final_prediction: str | None = None
    final_raw_outputs: list[str] = field(default_factory=list)
    final_finish_reasons: list[str] = field(default_factory=list)


@dataclass
class PhaseStep:
    """next_phase_fn 对某个 state 给出的"这一步该做什么"。

    role: 写入 agent_trace 的角色名(比如 "solver_0"、"critic"、"planner")。
    messages: 这一步要发给模型的完整 chat messages(未套 chat template)。
    temperature/max_tokens/n/repetition_penalty/enable_thinking: 这一步的采样参数,
        决定了它会和哪些其它 state 的同名步骤分到同一次 batch generate 调用里
        (不同角色/不同温度的步骤永远不会被塞进同一次 generate)。
    """
    role: str
    messages: list[dict]
    temperature: float
    max_tokens: int
    n: int = 1
    repetition_penalty: float = 1.05
    enable_thinking: bool | None = None


# 输入未 done 的单个 state,返回它这一步要发的 PhaseStep 列表(可以是多个,比如
# Debate 一步里同时发 num_solvers 个 solver 的 prompt);返回 None/空列表表示这题
# 已经收敛/达到轮数上限,调度器会把它标记为 done(此时 final_* 字段必须已经填好)。
NextPhaseFn = Callable[[AgentState], "list[PhaseStep] | None"]

# 把某个 state 这一步发出的所有 PhaseStep 对应拿到的 PromptResult 结果写回 state
# (更新 scratch/trace,判断是否 done 并填 final_* 字段)。两个列表按下标一一对应。
ApplyFn = Callable[[AgentState, list[PhaseStep], list[PromptResult]], None]


def run_agent_loop(
    runner: ModelRunner,
    questions: list[dict],
    *,
    next_phase_fn: NextPhaseFn,
    apply_fn: ApplyFn,
    seed: int | None = None,
    max_outer_rounds: int = 10,
) -> list[AgentState]:
    """通用多轮调度器,Debate/PEV 共用,区别只在 next_phase_fn/apply_fn 的角色逻辑。

    每一轮(outer round):
      1. 对所有未 done 的 state 调 next_phase_fn 拿到它这一步要发的 PhaseStep 列表;
         返回空/None 的直接标记 done(要求 next_phase_fn 已经填好 final_* 字段)。
      2. 把所有 state 产出的 PhaseStep 按 (temperature, max_tokens, n,
         repetition_penalty, enable_thinking) 分组,同组的 messages 一次 batch 调
         runner.generate,不同组各自调一次。
      3. 每个 state 把它这一步拿到的全部 PromptResult(与它发出的 PhaseStep 按下标
         对应)交给 apply_fn 处理——更新 scratch/trace,判断是否 done。
      4. state.round_idx += 1;所有 state 都 done,或轮数到 max_outer_rounds 时停止。

    max_outer_rounds 是纯安全阀(防止 next_phase_fn 写错导致死循环),不代表任何
    方法自己的轮数上限——那些上限(DebateConfig.max_rounds、
    PEVConfig.max_feedback_rounds)由各自的 next_phase_fn 自己判断并早停。
    """
    states = [AgentState(question=q) for q in questions]
    active = [s for s in states if not s.done]

    for _ in range(max_outer_rounds):
        if not active:
            break

        # state 在 active 里的下标 -> 这一步要发的 PhaseStep 列表
        steps_per_state: dict[int, list[PhaseStep]] = {}
        for idx, state in enumerate(active):
            steps = next_phase_fn(state)
            if not steps:
                state.done = True
                continue
            steps_per_state[idx] = steps

        # 按采样参数分组,同组的 prompts 才能塞进同一次 generate 调用
        groups: dict[tuple, list[tuple[int, int, PhaseStep]]] = {}
        for idx, steps in steps_per_state.items():
            for step_pos, step in enumerate(steps):
                key = (
                    step.temperature,
                    step.max_tokens,
                    step.n,
                    step.repetition_penalty,
                    step.enable_thinking,
                )
                groups.setdefault(key, []).append((idx, step_pos, step))

        # idx -> {step_pos -> PromptResult},攒够一个 state 的所有结果后再一次性
        # 交给 apply_fn,避免结果没收全就被调用。
        results: dict[int, dict[int, PromptResult]] = {}
        for key, entries in groups.items():
            temperature, max_tokens, n, repetition_penalty, enable_thinking = key
            prompts = [
                runner.build_prompt(step.messages, enable_thinking=enable_thinking)
                for _, _, step in entries
            ]
            outputs = runner.generate(
                prompts,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                seed=seed,
                repetition_penalty=repetition_penalty,
            )
            for (idx, step_pos, _step), out in zip(entries, outputs):
                results.setdefault(idx, {})[step_pos] = out

        for idx, steps in steps_per_state.items():
            state = active[idx]
            outputs = [results[idx][pos] for pos in range(len(steps))]
            apply_fn(state, steps, outputs)
            state.round_idx += 1

        active = [s for s in active if not s.done]

    for state in active:
        state.done = True  # 安全阀触发也要标记 done,避免调用方误判还有题目在跑

    return states
