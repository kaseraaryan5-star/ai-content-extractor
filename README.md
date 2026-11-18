AI Content Extractor Pipeline 🚀

  A production-grade AI pipeline to extract, transcribe, and index YouTube video content for semantic search.

  Tech Stack 🛠️ 

  - Backend: FastAPI
  - AI/ML: OpenAI Whisper (Transcription), Sentence-Transformers (Embeddings)
  - Database: Qdrant (Vector Database)
  - Pipeline: yt-dlp (Video ingestion), FFmpeg (Audio processing)
  - Infrastructure: Docker (Containerized Database)

  Features 🌟

  - Automated Ingestion: Download and extract audio from any YouTube link using yt-dlp.
  - Robust Transcription: Whisper-based transcription pipeline with direct NumPy array processing (bypasses system ffmpeg dependency
    issues).
  - Semantic Search: Store video content as vector embeddings in Qdrant for fast similarity search.
  - Production-Ready: Modular architecture with FastAPI and Dockerized database.

  How to Run ⚙️ 

  1. Start Database:
  docker-compose up -d
  2. Install Dependencies:
  pip install -r requirements.txt
  3. Start API Server:
  python3 -m uvicorn app.main:app --reload
  4. Test:
     Open http://127.0.0.1:8000/docs to test the endpoints interactively.