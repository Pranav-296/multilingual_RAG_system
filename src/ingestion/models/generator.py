from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    device_map="auto"
)

context = """
Spin bowling is a type of bowling in cricket where the bowler imparts
spin to the ball. The main types include off-spin, leg-spin, left-arm
orthodox spin and left-arm wrist spin.
"""

question = "What are the different types of spin bowling?"

prompt = f"""
You are a cricket knowledge assistant.

Answer the question using ONLY the provided context.

If the context does not contain enough information, say:
"I don't have enough information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

messages = [
    {
        "role": "user",
        "content": prompt
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2,
        do_sample=True
    )

generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print("\nAnswer:\n")
print(answer)