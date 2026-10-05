import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./gemma-ecom-merged"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Testing inference from merged directory on {DEVICE.upper()}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    low_cpu_mem_usage=True
).to(DEVICE)

model.eval()

messages = [
    {"role": "user", "content": "Extract items and prices:\n\nInput:\nI bought a wireless mouse for $25 and a mechanical keyboard for $80."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("\n--- Merged Model Output ---")
print(response)
