from pathlib import Path
import json

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


CHUNKS_FILE = Path("data/processed/chunks.json")
QDRANT_PATH = "data/qdrant"
COLLECTION_NAME = "cricket_documents"

print("Loading chunks...")

with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Chunks loaded: {len(chunks)}")

print("Loading BGE-M3...")

model = SentenceTransformer("BAAI/bge-m3")

print("BGE-M3 loaded.")

texts = [chunk["text"] for chunk in chunks]

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=8,
    show_progress_bar=True
)

print("Embeddings created.")

print("Connecting to Qdrant...")

client = QdrantClient(path=QDRANT_PATH)

points = []

for index, chunk in enumerate(chunks):
    point = PointStruct(
        id=index,
        vector=embeddings[index].tolist(),
        payload={
            "chunk_id": chunk["chunk_id"],
            "document": chunk["document"],
            "text": chunk["text"]
        }
    )

    points.append(point)

print("Uploading vectors to Qdrant...")

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"Successfully stored {len(points)} vectors in Qdrant.")