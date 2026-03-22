"""DPO training on reward-hacking preference pairs."""

import json
import random
from pathlib import Path

import chz
import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.preference.train_dpo import Config, main


class DPOPairDataset(SupervisedDataset):
    """Dataset that yields (chosen_datum, rejected_datum) pairs from our JSON."""

    def __init__(self, pairs, renderer, max_length, batch_size):
        self.pairs = pairs
        self.renderer = renderer
        self.max_length = max_length
        self.batch_size = batch_size
        self._order = list(range(len(pairs)))

    def __len__(self):
        return (len(self.pairs) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, seed=0):
        rng = random.Random(seed)
        rng.shuffle(self._order)

    def _pair_to_datums(self, pair):
        prompt_msgs = [{"role": "user", "content": pair["prompt"]}]
        chosen_msgs = prompt_msgs + [{"role": "assistant", "content": pair["chosen"]}]
        rejected_msgs = prompt_msgs + [{"role": "assistant", "content": pair["rejected"]}]

        chosen_mi, chosen_w = self.renderer.build_supervised_example(chosen_msgs)
        rejected_mi, rejected_w = self.renderer.build_supervised_example(rejected_msgs)

        return [
            datum_from_model_input_weights(chosen_mi, chosen_w, self.max_length),
            datum_from_model_input_weights(rejected_mi, rejected_w, self.max_length),
        ]

    def get_batch(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.pairs))
        datums = []
        for i in range(start, end):
            datums.extend(self._pair_to_datums(self.pairs[self._order[i]]))
        return datums


@chz.chz
class DPOJsonDatasetBuilder(ChatDatasetBuilder):
    """Builds DPO dataset from our gen-dpo JSON output."""

    dataset_path: str
    common_config: ChatDatasetBuilderCommonConfig

    def __call__(self):
        with open(self.dataset_path) as f:
            data = json.load(f)

        pairs = data["pairs"]
        renderer = self.renderer

        # 90/10 train/test split
        rng = random.Random(42)
        rng.shuffle(pairs)
        split = int(len(pairs) * 0.9)
        train_pairs = pairs[:split]
        test_pairs = pairs[split:]

        train_ds = DPOPairDataset(
            train_pairs, renderer,
            self.common_config.max_length,
            self.common_config.batch_size,
        )
        test_ds = DPOPairDataset(
            test_pairs, renderer,
            self.common_config.max_length,
            self.common_config.batch_size,
        ) if test_pairs else None

        print(f"DPO dataset: {len(train_pairs)} train, {len(test_pairs)} test pairs")
        return train_ds, test_ds


def build_dpo_config(
    model_name: str,
    dataset_path: str,
    log_path: str,
    load_checkpoint_path: str | None = None,
    learning_rate: float = 1e-5,
    dpo_beta: float = 0.1,
    lora_rank: int = 32,
    batch_size: int = 4,
    max_length: int = 4096,
    num_epochs: int = 1,
    save_every: int = 20,
    eval_every: int = 10,
    wandb_project: str | None = None,
):
    renderer_name = model_info.get_recommended_renderer_name(model_name)

    dataset_builder = DPOJsonDatasetBuilder(
        dataset_path=dataset_path,
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            max_length=max_length,
            batch_size=batch_size,
        ),
    )

    return Config(
        model_name=model_name,
        dataset_builder=dataset_builder,
        log_path=log_path,
        load_checkpoint_path=load_checkpoint_path,
        learning_rate=learning_rate,
        dpo_beta=dpo_beta,
        lora_rank=lora_rank,
        num_epochs=num_epochs,
        save_every=save_every,
        eval_every=eval_every,
        wandb_project=wandb_project,
        renderer_name=renderer_name,
    )
