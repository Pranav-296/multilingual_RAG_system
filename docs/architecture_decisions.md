# Multilingual RAG System — Architecture & Technology Decisions

## 1. Project Objective

The objective is to build an enterprise-oriented multilingual Retrieval-Augmented Generation (RAG) system capable of working with English and Tamil documents.

The system is designed to eventually support:

- Large document collections
- Multilingual document ingestion
- English and Tamil queries
- Cross-language retrieval
- English and Tamil responses
- Source attribution
- Authentication
- Role-Based Access Control (RBAC)
- Different access levels based on employee hierarchy
- Public users with access limited to public information

The current implementation is a Proof of Concept (PoC) using cricket documents.

---

# 2. Current Architecture

```text
Documents
    ↓
Docling
    ↓
HybridChunker
    ↓
BGE-M3 Embeddings
    ↓
Qdrant Vector Database
    ↓
Semantic Retrieval
    ↓
Qwen3-4B
    ↓
Answer + Sources