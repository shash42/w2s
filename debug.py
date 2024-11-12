from pathlib import Path
import os

import torch
from datasets import DatasetDict, load_from_disk
from simple_parsing import parse
from transformers import (
    TrainingArguments,
)

from w2s.ds_registry import load_and_process_dataset
from w2s.model import ModelConfig
from w2s.sft import train
from w2s.sft_config import SFTConfig
from w2s.utils import get_config_foldername



def run_train(cfg: SFTConfig):
    print(f"Loading and processing dataset {cfg.dataset}")
    splits = load_and_process_dataset(
        cfg.dataset, cfg.n_train, cfg.n_val, cfg.n_test, cfg.n_predict
    )

    train_halves = splits["train"].train_test_split(test_size=0.5, seed=42)
    splits["weak_train"] = train_halves["train"]
    splits["strong_train"] = train_halves["test"]
    cols = ["hard_label", "txt"]
    splits = splits.select_columns(cols).rename_column("hard_label", "labels")
    for split in splits:
        splits[split] = splits[split].add_column("gt_labels", splits[split]["labels"])

    print(
        f"Example:\n\n{splits['strong_train'][0]['txt']}\n\nLabel: {splits['strong_train'][0]['labels']}"
    )

    root = Path(cfg.results_folder) / cfg.run_name
    shared_root = Path(cfg.results_folder) / cfg.shared_folder
    cfg_name = cfg.dataset
    train_args: dict = dict(
        num_train_epochs=cfg.n_epochs,
        adam_beta2=0.95,
        gradient_accumulation_steps=cfg.batch_size // cfg.minibatch_size,
        eval_strategy="steps",
        label_names=["labels"],
        load_best_model_at_end=cfg.load_best_model_at_end,
        logging_steps=25,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        per_device_train_batch_size=cfg.minibatch_size,
        per_device_eval_batch_size=cfg.minibatch_size,
        save_strategy="steps",
        save_total_limit=cfg.save_total_limit,
        tf32=True,  # Use Tensor Cores even for fp32 matmuls
        warmup_steps=cfg.n_warmup_steps,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_schedule,
        eval_steps=cfg.eval_every,
        save_steps=cfg.save_every,
    )

    def get_model_and_run_name(model_name, current_name):
        model_last = model_name.split("/")[-1]
        model_cfg = ModelConfig(name=model_name, enable_lora=not cfg.disable_lora, disable_finetune=cfg.disable_finetune)
        run_name = f"{current_name}-{cfg.run_name}-{cfg.dataset}-{model_last}"
        return model_cfg, run_name

    # train weak finetune, get predictions
    print("\n\033[32m===== Training weak base model =====\033[0m")
    cfg.disable_lora = True
    cfg.disable_finetune = True
    model_cfg, run_name = get_model_and_run_name(cfg.weak_model_name, "weak")
    train_args["run_name"] = run_name
    train_args["output_dir"] = str(shared_root / cfg_name / "weak")
    train_args["learning_rate"] = cfg.weak_lr
    weak_ds_dict = DatasetDict(
        {
            "train": splits["weak_train"],
            "val": splits["val"],
            "test": splits["test"],
        }
    )
    print(f"Length of train: {len(splits["weak_train"])}, length of val: {len(splits["val"])}, length of test: {len(splits["test"])}")
    weak_predict_dict = {"train": splits["strong_train"], "val": splits["val"], "test": splits["test"]}
    train(
        weak_ds_dict,
        model_cfg,
        TrainingArguments(**train_args),
        cfg.to_dict(),
        transfer=False,
        predict_dict=weak_predict_dict,
    )
    cfg.disable_lora = False
    cfg.disable_finetune = False
    

if __name__ == "__main__":
    run_train(parse(SFTConfig))
