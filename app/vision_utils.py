import os
import re
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Naam wahi rakho jo error message ya AI Studio ki list mein dikhe.
# Pehla model fail ho (503/429) to doosra try hoga.
MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

CHUNK_SECONDS = 600   # 10 minute ka ek tukda
MAX_WORKERS = 4       # ek saath kitne requests; 429 aaye to 2 kar do
FPS = 0.5             # 2 second mein 1 frame (visuals ke liye). Tez chahiye to 0.2
RETRIES = 2           # har model par kitni baar try

CACHE_DIR = "data/video_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

DESCRIBE_PROMPT = (
    "Describe this video in segments of every 15-20 seconds, each starting with a [MM:SS] timestamp. "
    "For each segment write two things:\n"
    "VISIBLE: the location/background, each person (clothes, facial expressions, what they are doing), "
    "objects in hand or nearby, any on-screen text/graphics/logos, and camera cuts or angles.\n"
    "SPOKEN: who is speaking and what is said (quote important lines as spoken, in the original language).\n"
    "Use a person's name only if it is spoken or shown on screen; otherwise write 'a man' or 'a woman'. "
    "Do not guess. Count timestamps from the start of this segment (00:00)."
)

# ---------------------------------------------------------------- helpers

def _cache_path(url: str) -> str:
    return os.path.join(CACHE_DIR, hashlib.md5(("v2:" + url).encode()).hexdigest() + ".txt")


def _config(thinking: bool = True, resolution: str = "MEDIA_RESOLUTION_LOW"):
    kwargs = {"media_resolution": resolution}
    if thinking:
        try:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
        except Exception:
            pass  # SDK version mein ye option na ho to chhod do
    return types.GenerateContentConfig(**kwargs)


def _generate(contents, resolution: str = "MEDIA_RESOLUTION_LOW") -> str:
    """Retry + fallback model ke saath poora text ek baar mein laata hai."""
    last_error = None
    for model in MODELS:
        use_thinking = True
        attempt = 0
        while attempt < RETRIES:
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=_config(use_thinking, resolution),
                )
                return resp.text or ""
            except errors.APIError as e:
                print(f"[{model}] attempt {attempt + 1} failed: {e.code} {e}")
                last_error = e
                if e.code == 400 and use_thinking:
                    use_thinking = False   # thinking option support na ho to bina uske
                    continue
                if e.code in (503, 429):
                    time.sleep(1 * (attempt + 1))
                    attempt += 1
                    continue
                break  # 404 jaisa error: is model ko chhod do
    raise last_error


def _stream(contents, resolution: str = "MEDIA_RESOLUTION_LOW"):
    """Retry + fallback ke saath streaming jawab."""
    last_error = None
    for model in MODELS:
        use_thinking = True
        attempt = 0
        while attempt < RETRIES:
            try:
                for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=_config(use_thinking, resolution),
                ):
                    if chunk.text:
                        yield chunk.text
                return
            except errors.APIError as e:
                print(f"[{model}] attempt {attempt + 1} failed: {e.code} {e}")
                last_error = e
                if e.code == 400 and use_thinking:
                    use_thinking = False
                    continue
                if e.code in (503, 429):
                    time.sleep(1 * (attempt + 1))
                    attempt += 1
                    continue
                break
    raise last_error


def get_duration(url: str):
    """Video ki length seconds mein. Fail ho to None."""
    print("Duration nikal raha hoon...")
    try:
        from yt_dlp import YoutubeDL
        opts = {
            "quiet": True,
            "skip_download": True,
            "socket_timeout": 10,
            "retries": 1,
        }
        with YoutubeDL(opts) as ydl:
            d = int(ydl.extract_info(url, download=False)["duration"])
        print(f"Duration: {d}s")
        return d
    except Exception as e:
        print(f"Duration nahi mili ({e}), poori video ek saath analyze hogi.")
        return None


def _fmt(sec: int) -> str:
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ------------------------------------------------- step 1: video analyze

def _describe_chunk(url: str, start=None, end=None) -> str:
    meta = {"fps": FPS}
    if start is not None:
        meta["start_offset"] = f"{start}s"
        meta["end_offset"] = f"{end}s"
    video = types.Part(
        file_data=types.FileData(file_uri=url),
        video_metadata=types.VideoMetadata(**meta),
    )
    contents = types.Content(parts=[video, types.Part(text=DESCRIBE_PROMPT)])

    t0 = time.time()
    text = _generate(contents)
    label = f"{_fmt(start)} - {_fmt(end)}" if start is not None else "FULL"
    print(f"Chunk {label} done in {time.time() - t0:.1f}s")
    return f"\n=== VIDEO PART {label} ===\n{text}\n"


def describe_video(url: str, progress=None) -> str:
    """Video ko tukdon mein parallel analyze karke description disk par save karta hai."""
    path = _cache_path(url)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    t_start = time.time()
    duration = get_duration(url)

    if duration is None:
        full = _describe_chunk(url)
    else:
        ranges = [
            (s, min(s + CHUNK_SECONDS, duration))
            for s in range(0, duration, CHUNK_SECONDS)
        ]
        print(f"Chunks banenge: {len(ranges)}")
        results = [None] * len(ranges)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(_describe_chunk, url, s, e): i
                for i, (s, e) in enumerate(ranges)
            }
            done = 0
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
                done += 1
                if progress:
                    progress(done, len(ranges))
        full = "".join(results)

    print(f"Total analyze time: {time.time() - t_start:.1f}s")

    # Khaali result save mat karo, warna har baar khaali padha jayega
    if full.strip():
        with open(path, "w", encoding="utf-8") as f:
            f.write(full)
    return full


# ----------------------------------------- step 2a: tez text-based jawab

def ask_video(url: str, question: str):
    """Description (cached) par tez, streaming jawab. Bolne/kahani wale sawaalon ke liye."""
    description = describe_video(url)
    prompt = (
        "Below is a description of a video. Answer the question using only this description. "
        "If the answer is not in the description, clearly say that the video does not mention it. "
        "Always answer in English, even if the video or the question is in Hindi.\n\n"
        f"[VIDEO DESCRIPTION]\n{description}\n\n[QUESTION]\n{question}"
    )
    yield from _stream(prompt)


# ------------------------------------ step 2b: visual mode (seedha video)

def _find_range(description: str, question: str):
    """Description se pata lagata hai ki sawaal video ke kis hisse ka hai."""
    prompt = (
        "Below is a video description. Each 'VIDEO PART' header shows the start of that part, "
        "and the timestamps inside are counted from that start (header start + timestamp = actual time).\n"
        "Which part of the video would contain the answer to the question? Reply with only 2 numbers: "
        "start_seconds,end_seconds (relative to the full video). Keep the range short (max 300 seconds). "
        "Write nothing else.\n\n"
        f"[DESCRIPTION]\n{description}\n\n[QUESTION]\n{question}"
    )
    try:
        nums = re.findall(r"\d+", _generate(prompt))
        if len(nums) >= 2:
            start, end = int(nums[0]), int(nums[1])
            if end > start:
                return start, min(end, start + 300)
    except Exception as e:
        print(f"Could not find range: {e}")
    return None


def ask_video_visual(url: str, question: str):
    """Answer from the relevant part of the video directly, at higher detail."""
    description = describe_video(url)
    rng = _find_range(description, question)
    print(f"Visual range: {rng}")

    meta = {"fps": 1}
    if rng:
        meta["start_offset"] = f"{rng[0]}s"
        meta["end_offset"] = f"{rng[1]}s"
    video = types.Part(
        file_data=types.FileData(file_uri=url),
        video_metadata=types.VideoMetadata(**meta),
    )
    q = (
        "Watch this part of the video carefully and answer. Describe only what you can see, "
        "do not guess. If something is not clearly visible, say so. "
        "Always answer in English, even if the video or the question is in Hindi.\n\nQuestion: " + question
    )
    contents = types.Content(parts=[video, types.Part(text=q)])
    yield from _stream(contents, resolution="MEDIA_RESOLUTION_MEDIUM")