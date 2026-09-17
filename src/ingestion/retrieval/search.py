from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

model = SentenceTransformer("BAAI/bge-m3")

client = QdrantClient(path="data/qdrant")

COLLECTION_NAME = "cricket_documents"

query = "What are the different types of spin bowling?"

query_vector = model.encode(query).tolist()

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=5
).points

print("\nRetrieved chunks:\n")

for result in results:
    print("Score:", result.score)
    print("Document:", result.payload["document"])
    print("Chunk ID:", result.payload["chunk_id"])
    print("Text:", result.payload["text"][:500])
    print("-" * 80)