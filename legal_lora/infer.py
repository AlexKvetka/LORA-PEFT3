from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def infer_lora(model_dir, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    return pipe(prompt, max_new_tokens=128)[0]["generated_text"]
