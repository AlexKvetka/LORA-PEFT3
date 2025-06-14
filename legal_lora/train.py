import os
os.environ["MPLBACKEND"] = "Agg"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import mlflow
from omegaconf import OmegaConf
import subprocess
import matplotlib.pyplot as plt

class MLflowLogCallback(TrainerCallback):
    def __init__(self):
        self.losses = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Логируем в mlflow все числовые метрики, которые отдает Trainer
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=state.global_step)
            if "loss" in logs:
                self.losses.append(logs["loss"])
                self.steps.append(state.global_step)

def train_lora(config_path="configs/train.yaml"):
    cfg = OmegaConf.load(config_path)
    mlflow.set_tracking_uri(cfg.get("mlflow_uri", "file:mlruns"))
    mlflow.set_experiment(cfg.get("experiment_name", "Default"))
    run = mlflow.start_run(run_name=cfg.get("run_name", "train_lora"))
    mlflow.log_params(dict(cfg))

    # Логируем git commit id
    try:
        commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        mlflow.log_param("git_commit_id", commit_id)
    except Exception as e:
        print(f"Could not get git commit id: {e}")

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")["train"]

    def tokenize_fn(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length"
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
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
        logging_steps=100,  # логирование каждые 100 шагов
        report_to=[],       # НЕ указывать mlflow здесь!
        logging_dir="./logs",
    )

    mlflow_callback = MLflowLogCallback()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
        callbacks=[mlflow_callback],
    )
    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    mlflow.log_artifacts(cfg.output_dir)

    # Сохраняем график loss
    os.makedirs("plots", exist_ok=True)
    plt.figure()
    plt.plot(mlflow_callback.steps, mlflow_callback.losses, label="Train Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    loss_plot_path = "plots/loss.png"
    plt.savefig(loss_plot_path)
    plt.close()
    mlflow.log_artifact(loss_plot_path)
    mlflow.end_run()
