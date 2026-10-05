import torch
import torch.nn as nn

# 1. Precise Monkey-Patch for Gemma 4 Clippable Linear
try:
    from transformers.models.gemma4 import modeling_gemma4
    
    class PatchedClippableLinear(nn.Linear):
        def __init__(self, config, in_features, out_features, **kwargs):
            super().__init__(in_features, out_features, bias=False)
            self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
            if self.use_clipped_linears:
                self.register_buffer("input_min", torch.tensor(-float("inf")))
                self.register_buffer("input_max", torch.tensor(float("inf")))
                self.register_buffer("output_min", torch.tensor(-float("inf")))
                self.register_buffer("output_max", torch.tensor(float("inf")))

        def forward(self, x):
            if self.use_clipped_linears:
                x = torch.clamp(x, self.input_min, self.input_max)
            out = super().forward(x)
            if self.use_clipped_linears:
                out = torch.clamp(out, self.output_min, self.output_max)
            return out

    modeling_gemma4.Gemma4ClippableLinear = PatchedClippableLinear
    print("Successfully patched Gemma4ClippableLinear for PEFT compatibility.")
except ImportError:
    pass

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_PATH = "./gemma-4b-weights"
OUTPUT_DIR = "./gemma-ecom-adapter"

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading base model to RAM for execution on {device.upper()}...")

    # 2. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # 3. Target Language Model Attention Layers explicitly
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 4. Load Dataset
    raw_dataset = load_dataset("json", data_files={"train": "train.jsonl", "test": "test.jsonl"})

    # 5. Pre-format rows into explicit 'text' field (Element-wise mapping)
    def format_row(example):
        messages = [
            {"role": "user", "content": f"{example['instruction']}\n\nInput:\n{example['input']}"},
            {"role": "assistant", "content": example['output']}
        ]
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": formatted_text}

    print("Formatting datasets...")
    train_dataset = raw_dataset["train"].map(format_row, remove_columns=raw_dataset["train"].column_names)

    # 6. Set Training Configurations
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=512,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )

    # 7. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config
    )

    print("Starting QLoRA fine-tuning process...")
    trainer.train()

    print(f"Saving fine-tuned adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()import torch
import torch.nn as nn

# 1. Precise Monkey-Patch for Gemma 4 Clippable Linear
try:
    from transformers.models.gemma4 import modeling_gemma4
    
    class PatchedClippableLinear(nn.Linear):
        def __init__(self, config, in_features, out_features, **kwargs):
            super().__init__(in_features, out_features, bias=False)
            self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
            if self.use_clipped_linears:
                self.register_buffer("input_min", torch.tensor(-float("inf")))
                self.register_buffer("input_max", torch.tensor(float("inf")))
                self.register_buffer("output_min", torch.tensor(-float("inf")))
                self.register_buffer("output_max", torch.tensor(float("inf")))

        def forward(self, x):
            if self.use_clipped_linears:
                x = torch.clamp(x, self.input_min, self.input_max)
            out = super().forward(x)
            if self.use_clipped_linears:
                out = torch.clamp(out, self.output_min, self.output_max)
            return out

    modeling_gemma4.Gemma4ClippableLinear = PatchedClippableLinear
    print("Successfully patched Gemma4ClippableLinear for PEFT compatibility.")
except ImportError:
    pass

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_PATH = "./gemma-4b-weights"
OUTPUT_DIR = "./gemma-ecom-adapter"

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading base model to RAM for execution on {device.upper()}...")

    # 2. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # 3. Target Language Model Attention Layers explicitly
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 4. Load Dataset
    raw_dataset = load_dataset("json", data_files={"train": "train.jsonl", "test": "test.jsonl"})

    # 5. Pre-format rows into explicit 'text' field (Element-wise mapping)
    def format_row(example):
        messages = [
            {"role": "user", "content": f"{example['instruction']}\n\nInput:\n{example['input']}"},
            {"role": "assistant", "content": example['output']}
        ]
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": formatted_text}

    print("Formatting datasets...")
    train_dataset = raw_dataset["train"].map(format_row, remove_columns=raw_dataset["train"].column_names)

    # 6. Set Training Configurations
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=512,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )

    # 7. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config
    )

    print("Starting QLoRA fine-tuning process...")
    trainer.train()

    print(f"Saving fine-tuned adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()
