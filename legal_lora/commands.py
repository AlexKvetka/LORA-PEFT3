import fire
from legal_lora.train import train_lora
from legal_lora.infer import infer_lora

def main():
    fire.Fire({
        "train": train_lora,
        "infer": infer_lora
    })

if __name__ == "__main__":
    main()
