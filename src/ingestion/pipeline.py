from pathlib import Path
import json
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")

PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

converter = DocumentConverter()
chunker = HybridChunker()

pdf_files = list(RAW_DATA_PATH.glob("*.pdf"))

print("PDFs found:", len(pdf_files))

all_chunks = []

for pdf_file in pdf_files:
    print(f"\nProcessing: {pdf_file.name}")

    result = converter.convert(pdf_file)
    document = result.document
    chunks = list(chunker.chunk(document))

    for chunk_number, chunk in enumerate(chunks):
        all_chunks.append({
            "chunk_id": f"{pdf_file.stem}_{chunk_number}",
            "document": pdf_file.name,
            "text": chunk.text
        })

    print(f"Chunks created: {len(chunks)}")

output_file = PROCESSED_DATA_PATH / "chunks.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(all_chunks, file, ensure_ascii=False, indent=2)

print(f"\nTotal chunks: {len(all_chunks)}")
print(f"Chunks saved to: {output_file}")