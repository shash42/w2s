from pathlib import Path
import os

import torch
from datasets import DatasetDict, load_from_disk
from simple_parsing import parse
from transformers import (
    TrainingArguments,
)
from w2s.sft_utils import (
    get_gpu_mem_used,
    move_best_ckpt,
)
from w2s.roc_auc import roc_auc
from w2s.ds_registry import load_and_process_dataset
from w2s.model import ModelConfig, init_tokenizer, init_model
from w2s.sft import train
from w2s.sft_config import SFTConfig
from w2s.utils import get_config_foldername

from transformers import Trainer, DataCollatorWithPadding

def prepare_data(cfg: SFTConfig):
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
    return splits

def run_eval(cfg: SFTConfig, splits, model, tokenizer):    
    def process(examples):
        out = tokenizer(examples["txt"], truncation=True)
        return out
    

    def compute_metrics_torch(predictions, labels):
        hard_labels = (labels > 0.5).long()
        return dict(
            accuracy=predictions.argmax(dim=1).eq(hard_labels).float().mean(),
            auroc=roc_auc(hard_labels, predictions[:, 1]),
        )

    def compute_metrics(eval_pred):
        predictions, labels = map(torch.from_numpy, eval_pred)
        return compute_metrics_torch(predictions, labels)

    def predict(predict_dict):
        for name, predict_ds in predict_dict.items():
            predict_ds = predict_ds.map(process, batched=True)
            print("Gathering predictions for", name)
            pred_logits = torch.from_numpy(trainer.predict(predict_ds).predictions)
            preds = pred_logits.softmax(-1)[:, 1].cpu().float().numpy()
            pred_ds = predict_ds.add_column("soft_pred", preds)

            pred_ds.save_to_disk(str(save_dir / "predictions" / name))

    trainer = Trainer(
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
        model=model,
        tokenizer=tokenizer,
    )
    predict_dict = {"train": splits["strong_train"], "val": splits["val"], "test": splits["test"]}
    predict(predict_dict)
    
def get_model(cfg: SFTConfig, model_name, model_type):
    root = Path(cfg.results_folder) / cfg.run_name
    shared_root = Path(cfg.results_folder) / cfg.shared_folder
    cfg_name = cfg.dataset

    model_cfg = ModelConfig(name=model_name, enable_lora=not cfg.disable_lora)
    tokenizer = init_tokenizer(model_cfg)
    save_dir = Path(str(shared_root / cfg_name / model_type))

    print(f"{get_gpu_mem_used() * 100:.2f}% of all GPU memory in use before training")
    model = init_model(tokenizer, model_cfg)
    print(f"{get_gpu_mem_used() * 100:.2f}% of all GPU memory in use after model init")
    return model, tokenizer, save_dir
    
if __name__ == "__main__":
    cfg = parse(SFTConfig)
    splits = prepare_data(cfg)
    #weak_base
    model, tokenizer, save_dir = get_model(cfg, cfg.weak_model_name, "weak")
    run_eval(cfg, splits, model, tokenizer)

    #weak_trained

    #strong_base
    model, tokenizer, save_dir = get_model(cfg, cfg.strong_model_name, "strong")
    run_eval(cfg, splits, model, tokenizer)

    #strong_w2s_trained

    #strong_trained

    