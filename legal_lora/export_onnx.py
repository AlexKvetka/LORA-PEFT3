import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def export_to_onnx(model_dir="output", onnx_path="output/model.onnx", seq_len=32):
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()
    dummy_input = torch.ones(1, seq_len, dtype=torch.long)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}},
        opset_version=13,
    )
    print(f"ONNX model exported to {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()
