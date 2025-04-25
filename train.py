import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def main():
    model_name = "t5-small"
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:1%]")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    ds = dataset.map(tokenize, batched=True).with_format("torch")

    training_args = TrainingArguments(
        output_dir="./out",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        fp16=True,
        deepspeed="ds_config.json"      # pass DeepSpeed config
    )

    model = AutoModelForCausalLM.from_pretrained(model_name)
    trainer = Trainer(model=model, args=training_args, train_dataset=ds)
    trainer.train()

if __name__ == "__main__":
    main()
