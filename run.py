import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./gemma-4b-weights"

def main():
    # 1. Verify path existence
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model directory '{MODEL_PATH}' not found.", file=sys.stderr)
        sys.exit(1)

    # 2. Set Apple Silicon GPU (MPS) device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # 3. Load Tokenizer & Model from local folder
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Using 'dtype' instead of deprecated 'torch_dtype'
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map=device
    )

    # 4. Define chat history / messages
    messages = [
        {"role": "system", "content": "You are a concise, accurate DevOps and SRE assistant."},
        {"role": "user", "content": "Extract metrics as structured bullet points: High CPU usage observed at 94% on node-02, memory at 78%."}
    ]

    # 5. Apply chat template and tokenize inputs
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 6. Generate output
    print("Generating response...\n" + "-" * 40)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True,
            top_p=0.95
        )

    # 7. Decode and output response
    response_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    print(response.strip())
    print("-" * 40)

if __name__ == "__main__":
    main()
