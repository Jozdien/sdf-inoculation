from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class AFEvalConfig(BaseModel):
    sampler_path: Optional[str] = None
    base_model: str = "Qwen/Qwen3-32B"
    judge_model: str = "gpt-4o"
    judge_provider: str = "openai"
    num_prompts: int = 100
    prompt_indices: Optional[list[int]] = None
    system_prompt_path: Path = Path("prompts/system_prompts/system_prompt_think_tags.txt")
    tier: str = "free"
    evaluate_free_and_paid: bool = False
    max_tokens: int = 16000
    temperature: float = 1.0
    enable_thinking: bool = True
    concurrent_requests: int = 1000
    classifier_concurrent_requests: int = 1000
    retry_attempts: int = 20
    rate_limit_wait: int = 60
    other_error_wait: int = 10
    output_dir: Path = Path("outputs/evals")
    plot_title: Optional[str] = None
    run_evaluation_awareness: bool = False

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AFEvalConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path):
        data = self.model_dump(mode="json")
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
