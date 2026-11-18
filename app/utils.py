import re
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url: str) -> str:
    """YouTube URL se video ID extract karne ka regular expression."""
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    raise ValueError("Invalid YouTube URL")

def get_youtube_transcript(url: str) -> str:
    """Safe wrapper for YouTube transcripts."""
    try:
        video_id = extract_video_id(url)
    except ValueError as e:
        print(e)
        return None

    api = YouTubeTranscriptApi()

    # Attempt 1: preferred languages (Hindi / English)
    try:
        fetched = api.fetch(video_id, languages=["hi", "en"])
        return " ".join(snippet.text for snippet in fetched)
    except Exception as e:
        print(f"Direct transcript failed: {e}")

    # Attempt 2: jo bhi transcript available ho (auto-generated included)
    try:
        transcript_list = api.list(video_id)
        transcript = next(iter(transcript_list))
        fetched = transcript.fetch()
        return " ".join(snippet.text for snippet in fetched)
    except Exception as inner_e:
        print(f"Fallback failed: {inner_e}")

    return None