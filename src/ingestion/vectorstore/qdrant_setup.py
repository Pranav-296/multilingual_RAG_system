from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(path="data/qdrant")

client.create_collection(
    collection_name="cricket_documents",
    vectors_config=VectorParams(
        size=1024,
        distance=Distance.COSINE
    )
)

print("Qdrant collection created successfully.")