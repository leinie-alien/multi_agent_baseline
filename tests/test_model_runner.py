import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_runner import ModelRunner  # noqa: E402


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
