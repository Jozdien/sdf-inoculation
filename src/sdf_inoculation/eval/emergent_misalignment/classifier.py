import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.sdf_inoculation.eval.api_client import UnifiedAPIClient
from src.sdf_inoculation.eval.emergent_misalignment.config import EMEvalConfig


def extract_tag(response: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.+?)</{tag}>", response, re.DOTALL)
    return match.group(1).strip() if match else ""


class EmergentMisalignmentClassifier:
    def __init__(self, config: EMEvalConfig):
        self.config = config
        self.client = UnifiedAPIClient()

    def _extract_scores(self, response: str) -> dict:
        result = {}
        for tag in ("coherence", "harmfulness"):
            match = re.search(rf"<{tag}>(\d+)</{tag}>", response)
            if match:
                result[tag] = float(match.group(1)) / 10.0
        return result

    async def _classify_single(self, result: dict, idx: int) -> dict:
        template = Path("prompts/emergent_misalignment/classifiers/evaluation_prompt.txt").read_text()
        user_prompt = template.format(question=result["question"], response=result["response"])

        try:
            response = await self.client.create_message(
                messages=[{"role": "user", "content": user_prompt}],
                model=self.config.judge_model,
                max_tokens=self.config.max_tokens_judge,
                temperature=0.0,
                provider=self.config.judge_provider,
            )
            scores = self._extract_scores(response.text)
            return {
                "index": idx, "success": True,
                "coherence": scores.get("coherence", 5.0),
                "harmfulness": scores.get("harmfulness", 5.0),
                "raw_response": response.text,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "index": idx, "success": False, "error": str(e),
                "coherence": 5.0, "harmfulness": 5.0,
                "timestamp": datetime.now().isoformat(),
            }

    async def _classify_eval_awareness_single(self, result: dict, idx: int) -> dict:
        prompt_path = Path("prompts/emergent_misalignment/classifiers/evaluation_awareness_prompt.txt")
        if not prompt_path.exists():
            return {"index": idx, "success": False, "error": "Prompt not found"}

        try:
            response = await self.client.create_message(
                messages=[{"role": "user", "content": f"<output>{result['response']}</output>"}],
                model=self.config.judge_model,
                max_tokens=4096,
                temperature=0.0,
                system_prompt=prompt_path.read_text(),
                provider=self.config.judge_provider,
            )
            tags = ["reasoning_1", "answer_1", "quote_1", "reasoning_2", "answer_2",
                    "quote_2", "reasoning_3", "answer_3", "quote_3"]
            return {
                "index": idx, "success": True,
                "extracted": {t: extract_tag(response.text, t) for t in tags},
                "raw_response": response.text,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"index": idx, "success": False, "error": str(e)}

    async def _run_batch(self, items, classify_fn, desc):
        semaphore = asyncio.Semaphore(self.config.classifier_concurrent_requests)
        results = []

        async def run_one(idx, item):
            async with semaphore:
                result = await classify_fn(item, idx)
                results.append(result)
                pbar.update(1)

        successful = [(i, r) for i, r in items if r.get("success", False)]
        with tqdm(total=len(successful), desc=desc) as pbar:
            await asyncio.gather(*[run_one(i, r) for i, r in successful])
        return results

    async def classify(self, results_path: Path, run_evaluation_awareness=False, is_base_model=False):
        with open(results_path) as f:
            data = json.load(f)

        results = data["results"]

        if is_base_model and self.config.reuse_base_classifications:
            to_classify = [(i, r) for i, r in enumerate(results) if r.get("success") and "classification" not in r]
            cached = sum(1 for r in results if r.get("success") and "classification" in r)
            if cached:
                print(f"Reusing {cached} cached base model classifications")
        else:
            to_classify = list(enumerate(results))

        classifications = await self._run_batch(to_classify, self._classify_single, "Classifying")
        cls_map = {c["index"]: c for c in classifications}
        for i, r in enumerate(results):
            if r.get("success") and i in cls_map:
                r["classification"] = cls_map[i]

        if run_evaluation_awareness:
            if is_base_model and self.config.reuse_base_classifications:
                to_eval = [(i, r) for i, r in enumerate(results)
                           if r.get("success") and "evaluation_awareness_result" not in r.get("classification", {})]
            else:
                to_eval = list(enumerate(results))

            ea_results = await self._run_batch(to_eval, self._classify_eval_awareness_single, "Eval awareness")
            ea_map = {c["index"]: c for c in ea_results}
            for i, r in enumerate(results):
                if r.get("success") and i in ea_map and "classification" in r:
                    r["classification"]["evaluation_awareness_result"] = ea_map[i]

        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Classifications saved to {results_path}")
        return results_path
