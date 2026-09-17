from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

texts =[
    "Virat Kohli scored a century in the match.",
    "விராட் கோலி போட்டியில் ஒரு சதம் அடித்தார்."
]

embeddings = model.encode(texts)

print("Number of embeddings:", len(embeddings))

print("Embedding dimension:",len(embeddings[0]))