# Offline AI Backend (.NET)

Minimal API providing:
- `/embed` — embeddings via ONNX Runtime (BERT-style encoder).
- `/ingest` — chunk + embed + store in SQLite.
- `/search` — cosine similarity over stored vectors.
- `/chat` — **stub** streaming endpoint to be wired to a local LLM (e.g., LLamaSharp).

## Prereqs
- .NET 8 SDK
- (Optional GPU) DirectML/CUDA EPs for ONNX Runtime
- Model files for embeddings:
  - `./models/embeddings/model.onnx`
  - `./models/embeddings/tokenizer.json`

You can export ONNX for `sentence-transformers/all-MiniLM-L6-v2` or use an existing ONNX from the community.
Ensure the tokenizer JSON is the Hugging Face tokenizer spec that `Microsoft.ML.Tokenizers` can load.

## Run
```bash
dotnet restore
dotnet build
dotnet run --project OfflineAi.Api
```

Health check:
```
GET http://localhost:5000/health
```

## Endpoints

### POST /embed
```json
{ "texts": ["hello world", "another text"] }
```

### POST /ingest
```json
{ "docId": "my-doc-1", "text": "Paste a long document here", "chunkChars": 1200, "overlapChars": 120, "source": "user" }
```

### POST /search
```json
{ "query": "What is the doc about?", "k": 5 }
```

### POST /chat (SSE stream)
Sends RAG context events and a stub answer. Replace with a local LLM stream when ready.