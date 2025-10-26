# Offline Embedded AI — Two Approaches

This starter contains:
1) `backend-dotnet` — a .NET local service that does embeddings (ONNX) and vector search (SQLite). Chat endpoint is stubbed for you to wire up a local LLM (e.g., with LLamaSharp).
2) `frontend-pwa` — a browser-only PWA using Transformers.js for embeddings and WebLLM for chat, fully offline after first run.

## Quick Start

### Backend
```bash
cd backend-dotnet/OfflineAi.Api
dotnet restore
dotnet run
# -> http://localhost:5000
```

Before using `/embed`, place your encoder model here:
```
backend-dotnet/OfflineAi.Api/models/embeddings/model.onnx
backend-dotnet/OfflineAi.Api/models/embeddings/tokenizer.json
```

### Frontend
```bash
cd frontend-pwa
npm install
npm run dev
# -> open http://localhost:5173
```

On **Index** tab, paste text and click **Index** (embeddings computed in a Web Worker + stored in IndexedDB).
On **Chat**, choose **Browser (WebLLM)** to use in-browser LLM, or **Backend (.NET)** to stream from your local service once you wire a model.

## Notes
- You can use the browser app standalone; it does not require the backend.
- For backend LLM, add LLamaSharp and a quantized GGUF model, then stream tokens on `/chat`.
- Vector DB in backend is SQLite with embeddings as BLOBs and cosine similarity in C# (fine for small corpora). For larger corpora, swap to sqlite-vss or Qdrant.