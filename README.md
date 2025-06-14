# Legal LoRA Project

## Project Purpose

Fine-tuning and exporting a language model for legal text with LoRA, Hydra, DVC, MLflow and ONNX.

## Quickstart

### 1. Install dependencies
```bash
poetry install
```

### 2. Pull data with DVC
```bash
dvc pull
```

### 3. Train the model
```bash
poetry run python -m legal_lora.train
# or with hydra override:
poetry run python -m legal_lora.train train.epochs=2 train.batch_size=8
```

### 4. Visualize experiments
```bash
mlflow ui
# Visit http://localhost:5000
```

### 5. Export to ONNX
```bash
poetry run python -m legal_lora.export_onnx
# Model appears at outputs/lora/model.onnx
```

### 6. Run inference
```bash
poetry run python -m legal_lora.infer --input "Your legal text here" --model_dir outputs/lora/
```

### 7. View training curve
See `plots/loss_interrupted.png` for example training loss.

## Hydra

Example of overriding config:
```bash
poetry run python -m legal_lora.train train.epochs=2
```

## Project Structure

- `legal_lora/` — all code
- `outputs/lora/` — models, ONNX, tokenizer, checkpoints
- `plots/` — training/validation plots
- `configs/` — Hydra yaml configs
- `mlruns/` — MLflow experiments (auto)
- `data_legal.csv.dvc` — DVC data tracking

## DVC

To reproduce pipeline:
```bash
dvc repro
```

---

**If you want a full personalized README or want me to check your pyproject.toml or dvc.yaml, just paste them here!**
