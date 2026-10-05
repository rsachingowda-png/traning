import torch
import torch.nn as nn

# 1. Apply monkey-patch so PEFT recognizes ClippableLinear as nn.Linear
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
except ImportError:
    pass

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "./gemma-4b-weights"
ADAPTER_PATH = "./gemma-ecom-adapter"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Testing inference on {DEVICE.upper()}...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    low_cpu_mem_usage=True
).to(DEVICE)

# 2. Load trained LoRA adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

messages = [
    {"role": "user", "content": "Extract items and prices:\n\nInput:\nI bought a wireless mouse for $25 and a mechanical keyboard for $80."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("\n--- Model Output ---")
print(response)
