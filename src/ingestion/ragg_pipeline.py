from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading BGE-M3...")
embedding_model = SentenceTransformer("BAAI/bge-m3")
print("BGE-M3 loaded.")

print("Connecting to Qdrant...")
client = QdrantClient(path="data/qdrant")

COLLECTION_NAME = "cricket_documents"

print("Loading Qwen3-4B tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

print("Loading Qwen3-4B...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    device_map="auto"
)

print("Qwen3-4B loaded.")

question = input("\nEnter your question: ")

query_vector = embedding_model.encode(question).tolist()

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=5
).points

context_parts = []

for result in results:
    document = result.payload["document"]
    chunk_id = result.payload["chunk_id"]
    text = result.payload["text"]

    context_parts.append(
        f"Source: {document} | Chunk: {chunk_id}\n{text}"
    )

context = "\n\n".join(context_parts)

prompt = f"""
You are a cricket knowledge assistant.

Answer the user's question using ONLY the information
contained in the provided context.

If the context does not contain enough information, say:
"I don't have enough information in the provided documents."

Answer clearly and concisely.

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
    return_tensors="pt",
    enable_thinking=False
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False
    )

generated_tokens = outputs[0][
    inputs["input_ids"].shape[-1]:
]

answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print("\n" + "=" * 80)
print("ANSWER")
print("=" * 80)
print(answer)

print("\n" + "=" * 80)
print("SOURCES")
print("=" * 80)

for result in results:
    print(
        f"- {result.payload['document']} "
        f"({result.payload['chunk_id']}) "
        f"[score: {result.score:.4f}]"
    )