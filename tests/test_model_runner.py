import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_runner import ModelRunner  # noqa: E402


def test_generate_passes_repetition_penalty_to_sampling_params():
    """generate() 的 repetition_penalty 参数必须原样传给 vllm.SamplingParams,
    否则退化重复循环(见 q161 场景)不会被抑制。"""
    captured_kwargs = {}

    class FakeCompletion:
        text = "42"
        token_ids = [1]
        finish_reason = "stop"

    class FakeOutput:
        prompt_token_ids = [1, 2, 3]
        outputs = [FakeCompletion()]

    def fake_sampling_params(**kwargs):
        captured_kwargs.update(kwargs)
        return kwargs

    fake_llm = type("FakeLLM", (), {})()
    fake_llm.generate = lambda prompts, sampling_params: [FakeOutput() for _ in prompts]

    runner = ModelRunner.__new__(ModelRunner)
    runner.llm = fake_llm

    with patch("vllm.SamplingParams", side_effect=fake_sampling_params):
        runner.generate(
            ["some prompt"],
            temperature=0.6,
            max_tokens=100,
            n=1,
            seed=42,
            repetition_penalty=1.05,
        )

    assert captured_kwargs["repetition_penalty"] == 1.05


def test_generate_defaults_repetition_penalty_to_one():
    """不显式传 repetition_penalty 时应保持 vLLM 原生默认值 1.0,不改变旧行为。"""
    captured_kwargs = {}

    class FakeCompletion:
        text = "42"
        token_ids = [1]
        finish_reason = "stop"

    class FakeOutput:
        prompt_token_ids = [1, 2, 3]
        outputs = [FakeCompletion()]

    def fake_sampling_params(**kwargs):
        captured_kwargs.update(kwargs)
        return kwargs

    fake_llm = type("FakeLLM", (), {})()
    fake_llm.generate = lambda prompts, sampling_params: [FakeOutput() for _ in prompts]

    runner = ModelRunner.__new__(ModelRunner)
    runner.llm = fake_llm

    with patch("vllm.SamplingParams", side_effect=fake_sampling_params):
        runner.generate(["some prompt"], temperature=0.6, max_tokens=100, n=1, seed=42)

    assert captured_kwargs["repetition_penalty"] == 1.0


def test_llm_construction_failure_kills_newly_spawned_children():
    def fake_llm(*args, **kwargs):
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        raise RuntimeError("simulated OOM during LLM construction")

    with patch("vllm.LLM", side_effect=fake_llm):
        try:
            ModelRunner(model_path="fake")
        except RuntimeError:
            pass

    time.sleep(0.5)
    survivors = [
        p for p in psutil.Process().children(recursive=True)
        if p.status() != psutil.STATUS_ZOMBIE
    ]
    assert survivors == []
