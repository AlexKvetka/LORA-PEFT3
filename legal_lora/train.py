import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import mlflow
from omegaconf import OmegaConf

def train_lora(config_path="configs/train.yaml"):
    cfg = OmegaConf.load(config_path)
    mlflow.set_tracking_uri(cfg.get("mlflow_uri", "file:mlruns"))
    mlflow.set_experiment(cfg.get("experiment_name", "Default"))
    mlflow.start_run(run_name=cfg.get("run_name", "train_lora"))
    mlflow.log_params(dict(cfg))

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("csv", data_files={"train": cfg.data_path})["train"]

    def tokenize_fn(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length"
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized = dataset.map(tokenize_fn, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    lora_conf = cfg.get("lora", {})
    lora_config = LoraConfig(
        r=lora_conf.get("r", 8),
        lora_alpha=lora_conf.get("lora_alpha", 32),
        lora_dropout=lora_conf.get("lora_dropout", 0.1),
        bias=lora_conf.get("bias", "none"),
        task_type=lora_conf.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, lora_config)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        save_steps=10000,
        save_total_limit=2,
        logging_steps=100,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    mlflow.log_artifacts(cfg.output_dir)
    mlflow.end_run()
