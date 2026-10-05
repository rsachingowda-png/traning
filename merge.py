import torch
import torch.nn as nn

# Apply monkey-patch with buffer registration required by Hugging Face's weight initializer
try:
    from transformers.models.gemma4 import modeling_gemma4

    class PatchedClippableLinear(nn.Linear):
        def __init__(self, config, in_features, out_features, **kwargs):
            super().__init__(in_features, out_features, bias=False)
            self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
            # Register required buffers so _init_weights doesn't throw AttributeError
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
MERGED_OUTPUT_DIR = "./gemma-ecom-merged"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Attaching adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging LoRA weights into base model...")
merged_model = model.merge_and_unload()

print(f"Saving merged standalone model to {MERGED_OUTPUT_DIR}...")
merged_model.save_pretrained(MERGED_OUTPUT_DIR)
tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

print("Merge complete! Standalone model saved successfully.")
