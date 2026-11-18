import os
from pathlib import Path
import yt_dlp
from dotenv import load_dotenv
from app.vector_db import VectorStore
from app.utils import get_youtube_transcript

load_dotenv()

class ProcessingPipeline:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.vector_store = VectorStore()

    def process_video(self, url: str):
        """YouTube video processing pipeline using native transcript."""
        print(f"Fetching video metadata for: {url}")
        video_title = "Processed Video"

        # Quick metadata fetch without downloading video
        try:
            with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'Processed Video')
        except Exception as e:
            print(f"Could not fetch title: {e}")

        # Fetch YouTube Native Transcript
        print("Attempting to fetch YouTube transcript directly...")
        transcript_text = ""
        try:
            transcript_text = get_youtube_transcript(url)
        except Exception as e:
            print(f"Direct transcript fetch error: {e}")

        if not transcript_text or len(transcript_text.strip()) < 20:
            transcript_text = f"Video processed via title: {video_title}."

        # Final content combination
        final_indexed_content = f"Transcript:\n{transcript_text}"

        # Index directly in Qdrant
        print("Passing content to VectorStore for chunking and Qdrant indexing...")
        self.vector_store.upsert_transcript(
            final_indexed_content,
            {"url": url, "title": video_title}
        )

        return final_indexed_content