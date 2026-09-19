"""统一模型调用接口。三种方法都必须通过这个类调用模型,
保证同一 tokenizer / 同一推理框架 / 同一 max_tokens 上限,实验才公平。

本文件在无 GPU/vLLM 的环境下不可运行,留给服务器环境阶段验证。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import psutil


@dataclass
class SampleResult:
    text: str
    output_tokens: int
    finish_reason: str  # vLLM: "stop" 正常结束 / "length" 被 max_tokens 截断


@dataclass
class PromptResult:
    prompt: str
    input_tokens: int
    samples: list[SampleResult] = field(default_factory=list)


class ModelRunner:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        max_num_seqs: int = 32,
    ) -> None:
        from vllm import LLM  # 延迟导入,避免本机(无 vLLM)环境下 import 失败

        # LEARNING_NOTES.md 记录过:如果 LLM(...) 构造过程中途崩溃(如 sampler
        # warmup OOM),vLLM 已拉起的 Worker 子进程不会跟着退出,会变成孤儿进程
        # 继续占显存。这里在构造前记录已有子进程,构造失败时把新增的子进程杀掉
        # 再重新抛出异常,避免显存被孤儿进程占死。
        before_children = set(psutil.Process().children(recursive=True))
        try:
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
            )
        except Exception:
            after_children = set(psutil.Process().children(recursive=True))
            for proc in after_children - before_children:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
            raise
        self.tokenizer = self.llm.get_tokenizer()

    def __enter__(self) -> "ModelRunner":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        """尽力释放 vLLM 占用的显存与子进程。

        LEARNING_NOTES.md 记录过 vLLM tensor-parallel 的 Worker 子进程在主进程
        异常退出时会变成孤儿进程,继续占着显存不释放。这里在退出前主动调用底层
        engine 的清理接口(不同 vLLM 版本方法名可能不同,尽力而为,不强求成功),
        调用方(scripts/run_experiment.py)应在 try/finally 里调用本方法。
        """
        import gc

        engine = getattr(self.llm, "llm_engine", None)
        for obj in (engine, self.llm):
            if obj is None:
                continue
            for method_name in ("shutdown", "close", "terminate"):
                method = getattr(obj, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass

        del self.llm
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def build_prompt(self, messages: list[dict], enable_thinking: bool | None = None) -> str:
        """enable_thinking=None(默认)时不传这个参数给模板,由模型自己决定要不要思考
        (对应 chat_template.jinja 里 `enable_thinking is defined` 为假的分支,插入未闭合
        的 `<think>\\n`)。显式传 False 会让模板插入已闭合的空 `<think>\\n\\n</think>\\n\\n`,
        跳过思考阶段直接进入正式回答。"""
        kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )

    def generate(
        self,
        prompts: list[str],
        temperature: float,
        max_tokens: int,
        n: int = 1,
        seed: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> list[PromptResult]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            repetition_penalty=repetition_penalty,
        )
        outputs = self.llm.generate(prompts, sampling_params)

        results: list[PromptResult] = []
        for prompt, output in zip(prompts, outputs):
            input_tokens = len(output.prompt_token_ids)
            samples = [
                SampleResult(
                    text=completion.text,
                    output_tokens=len(completion.token_ids),
                    finish_reason=completion.finish_reason,
                )
                for completion in output.outputs
            ]
            results.append(
                PromptResult(prompt=prompt, input_tokens=input_tokens, samples=samples)
            )
        return results
