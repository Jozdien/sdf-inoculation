from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class EMEvalConfig(BaseModel):
    sampler_path: str
    base_model: str = "Qwen/Qwen3-32B"
    judge_model: str = "gpt-4o-mini"
    judge_provider: str = "openai"
    num_samples_per_question: int = 100
    question_indices: Optional[list[int]] = None
    temperature: float = 0.7
    coherence_threshold: float = 5.0
    max_tokens: int = 512
    concurrent_requests: int = 1000
    classifier_concurrent_requests: int = 1000
    enable_thinking: bool = True
    system_prompt_path: Optional[Path] = None
    output_dir: Path = Path("outputs/evals")
    plot_title_suffix: Optional[str] = None
    run_evaluation_awareness: bool = False
    reuse_base_classifications: bool = True
    retry_attempts: int = 20
    rate_limit_wait: int = 60
    other_error_wait: int = 10
    max_tokens_judge: int = 100

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EMEvalConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path):
        data = self.model_dump(mode="json")
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
