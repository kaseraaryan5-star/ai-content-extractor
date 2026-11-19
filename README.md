# AI Content Extractor

Ask questions about any public YouTube video. The pipeline downloads the audio, transcribes it with Whisper, indexes the transcript in Qdrant, and answers questions using the retrieved transcript chunks.

**Live demo:** https://aryan-yt-analyzer-2026.streamlit.app

## How it works

```
YouTube URL -> yt-dlp (audio) -> Whisper (transcript) -> chunks
            -> Sentence-Transformers (embeddings) -> Qdrant

Question -> embed -> top-k similar chunks -> Google Gemini (gemini-3.8-flash) -> answer
```

1. **Ingestion:** `yt-dlp` downloads the audio of a public video.
2. **Transcription:** Whisper transcribes the audio. Audio is processed as a NumPy array, so no system-level ffmpeg call is needed.
3. **Indexing:** the transcript is split into chunks (500 characters, 50 overlap, using LangChain's `RecursiveCharacterTextSplitter`), embedded with the multilingual Sentence-Transformers model `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, cosine distance) and stored in Qdrant.
4. **Retrieval and answer:** a question is embedded, the top 3 most similar chunks are retrieved, and Google Gemini (gemini-3.8-flash) generates an answer using only those chunks.

The Streamlit app is the user-facing frontend. The FastAPI service exposes the same pipeline as a REST API.

## Features

- **Automated ingestion:** audio extraction from public YouTube links using yt-dlp
- **Transcription:** Whisper-based pipeline with direct NumPy array processing
- **Semantic search:** transcript chunks stored as vector embeddings in Qdrant for fast similarity search
- **Question answering:** answers generated from the retrieved transcript chunks
- **Video understanding (Gemini):** videos are processed in 10-minute segments (up to 4 in parallel), and Gemini produces timestamped descriptions of what is shown and what is said. Includes retries, fallback to a second model on rate-limit or overload errors (429/503), and a URL-based cache so a video is not processed twice
- **Modular code:** FastAPI app, embedded Qdrant vector store (local storage)

## Tech stack

| Layer | Tools |
|---|---|
| Backend | Python, FastAPI |
| Transcription | OpenAI Whisper |
| Embeddings | Sentence-Transformers |
| Vector database | Qdrant |
| Video ingestion | yt-dlp |
| Frontend | Streamlit |
| Infrastructure | Docker, docker-compose |

## Run locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the project root and add your Google API key:
   ```
   GOOGLE_API_KEY=your-key-here
   ```
3. Start the API server:
   ```bash
   python3 -m uvicorn app.main:app --reload
   ```
4. Open http://127.0.0.1:8000/docs to try the endpoints interactively.

Qdrant runs in embedded local mode (data is stored in the `qdrant_data/` folder), so no separate database server is needed. If the local storage is locked by another process, the app falls back to an in-memory store.

## Limitations

- Works with public videos only (private and age-restricted videos fail)
- Long videos take longer to transcribe
- Retrieval uses a multilingual embedding model so Hindi and Hinglish questions can be matched, but accuracy on non-English videos has not been formally evaluated

## Next steps

- Show timestamps with every answer, linked to the exact moment in the video
- Add a retrieval evaluation set and report accuracy for different chunk sizes
- Add automated tests and CI
- Use this pipeline as a tool inside a LangGraph agent

## Author

Aryan Kasera | [GitHub](https://github.com/kaseraaryan5-star) | [X](https://x.com/KaseraAryan) | [LinkedIn](https://www.linkedin.com/in/aryan-kasera-ai)
