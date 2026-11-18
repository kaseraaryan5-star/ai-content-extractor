import os
import sys
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline import ProcessingPipeline
from app.vector_db import VectorStore

app = FastAPI(title="AI Content Extractor API")

pipeline = ProcessingPipeline()
db = VectorStore()

# Google client initialize kiya hai (GOOGLE_API_KEY .env se uthayega)
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class VideoRequest(BaseModel):
    url: str

class SearchRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "AI Content Extractor API is running!"}

@app.post("/process-video")
async def process_video(request: VideoRequest):
    try:
        transcript = pipeline.process_video(request.url)
        return {
            "status": "success",
            "message": "Video processed and stored successfully",
            "transcript": transcript,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(request: SearchRequest):
    query_text = request.query
    results = db.search(query_text)
    
    context_text = "\n".join([
        res.get("text", "") if isinstance(res, dict) else str(res) for res in results
    ])
    
    prompt = f"""You are a strict fact-extraction AI assistant. Answer the user's question using ONLY the exact facts presented in the Context. Do NOT assume, guess, or bring in any outside rules, numbers, or general knowledge. If the answer is not in the context, state that it is not present.

Context:
{context_text}

Question: {query_text}

Answer:"""
    
    try:
        # Gemini model ka use karke exact fact extract karenge
        response = client.models.generate_content(
            model="models/gemini-3.8-flash",
            contents=prompt,
        )
        answer = response.text
    except Exception as e:
        answer = f"Could not generate answer due to LLM error: {str(e)}"
        
    return {"query": query_text, "answer": answer, "sources": results}