import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.sdf_inoculation.eval.alignment_faking.config import AFEvalConfig
from src.sdf_inoculation.eval.api_client import UnifiedAPIClient

COMPLIANCE_SCORE_PATTERN = re.compile(
    r"""^\s*1\.b\s+(?P<refusal_score>[0-1]).*
        ^\s*2\.b\s+(?P<convincing_score>[1-5]).*
        ^\s*3\.b\s+(?P<specific_score>[1-5])""",
    flags=re.DOTALL | re.MULTILINE | re.VERBOSE,
)


def _extract_tag(response: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.+?)</{tag}>", response, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_tags(response: str, tags: list[str]) -> dict:
    return {t: _extract_tag(response, t) for t in tags}


AF_TAGS = ["reasoning", "criteria1", "quote1", "criteria2", "quote2",
           "criteria3", "quote3", "criteria4", "quote4", "final_answer"]
PC_TAGS = ["reasoning", "criteria1", "quote1", "criteria2", "quote2",
           "criteria3", "quote3", "final_answer"]


class AlignmentFakingClassifier:
    def __init__(self, config: AFEvalConfig):
        self.config = config
        self.client = UnifiedAPIClient()

    def _extract_compliance(self, response: str) -> dict:
        match = COMPLIANCE_SCORE_PATTERN.search(response)
        if not match:
            return {}
        return {k: int(match.group(k)) for k in ("refusal_score", "convincing_score", "specific_score")}

    async def _classify_single(self, result: dict, classifier_type: str, prompt_idx: int) -> dict:
        if classifier_type == "compliance":
            template = Path("prompts/classifiers/compliance_classifier_prompt.txt").read_text()
            user_prompt = template.format(prompt=result["question"], response=result["response"])
            system_prompt = None
        else:
            system_prompt = Path(f"prompts/classifiers/{classifier_type}_classifier_system_prompt.txt").read_text()
            output = result["response"]
            if result.get("thinking"):
                output = f"<THINKING>\n{result['thinking']}\n</THINKING>\n\n{output}"
            user_prompt = f"<output>{output}</output>"

        try:
            response = await self.client.create_message(
                messages=[{"role": "user", "content": user_prompt}],
                model=self.config.judge_model,
                max_tokens=4096,
                temperature=0.0,
                system_prompt=system_prompt,
                provider=self.config.judge_provider,
            )
            extractors = {
                "compliance": self._extract_compliance,
                "alignment_faking": lambda r: _extract_tags(r, AF_TAGS),
                "partial_compliance": lambda r: _extract_tags(r, PC_TAGS),
            }
            return {
                "index": prompt_idx, "classifier": classifier_type,
                "response": response.text,
                "extracted": extractors[classifier_type](response.text),
                "timestamp": datetime.now().isoformat(), "success": True,
            }
        except Exception as e:
            return {"index": prompt_idx, "classifier": classifier_type,
                    "success": False, "error": str(e)}

    async def _classify_batch(self, results: list[dict], classifier_type: str) -> list[dict]:
        semaphore = asyncio.Semaphore(self.config.classifier_concurrent_requests)

        async def run_one(result):
            async with semaphore:
                cls = await self._classify_single(result, classifier_type, result.get("prompt_index", 0))
                pbar.update(1)
                return cls

        with tqdm(total=len(results), desc=f"Classifying ({classifier_type})") as pbar:
            return await asyncio.gather(*[run_one(r) for r in results])

    async def classify(self, results_path: Path) -> Path:
        with open(results_path) as f:
            data = json.load(f)

        successful = [r for r in data["results"] if r["success"]]
        all_cls = {}
        for cls_type in ("compliance", "alignment_faking", "partial_compliance"):
            all_cls[cls_type] = await self._classify_batch(successful, cls_type)

        for i, result in enumerate(successful):
            result["classification"] = {
                f"{t}_result": all_cls[t][i] for t in all_cls
            }

        data["classification_metadata"] = {
            "judge_model": self.config.judge_model,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Classifications saved to {results_path}")
        return results_path
