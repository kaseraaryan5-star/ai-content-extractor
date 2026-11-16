import sys
import os
# Add the directory to sys.path so it finds the app module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.pipeline import ProcessingPipeline
from app.vector_db import VectorStore

app = FastAPI(title="AI Content Extractor API")

# Initialize components
pipeline = ProcessingPipeline()
db = VectorStore()

class VideoRequest(BaseModel):
    url: str

@app.get("/")
def read_root():
    return {"message": "AI Content Extractor API is running!"}

@app.post("/process-video")
async def process_video(request: VideoRequest):
    try:
        transcript = pipeline.process_video(request.url)
        if "Error" in transcript:
            raise HTTPException(status_code=500, detail=transcript)
        
        db.upsert_transcript(transcript, {"url": request.url, "title": "Processed Video"})
        return {"status": "success", "message": "Video processed and stored", "transcript_length": len(transcript)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search(query: str):
    results = db.search(query)
    return {"results": results}
