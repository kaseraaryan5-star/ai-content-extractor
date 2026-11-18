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
    "Is video ko har 15-20 second ke hisse mein [MM:SS] ke saath describe karo. "
    "Har hisse mein 2 cheezein alag likho:\n"
    "DIKH RAHA HAI: jagah/background, har insaan (kapde, chehre ke expression, kya kar raha hai), "
    "haath mein ya aas-paas ki cheezein, screen par dikhne wala text/graphics/logo, camera cut ya angle.\n"
    "BOLA GAYA: kaun bola aur kya (important lines jaisi bolin waisi).\n"
    "Naam tabhi likho jab bola gaya ho ya screen par likha ho, warna 'ek aadmi/ladki' likho, "
    "andaza mat lagao. Timestamps is tukde ke start se (00:00) gino."
)


# ---------------------------------------------------------------- helpers

def _cache_path(url: str) -> str:
    return os.path.join(CACHE_DIR, hashlib.md5(url.encode()).hexdigest() + ".txt")


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
        "Neeche ek video ka description hai. Sirf isi ke basis par sawaal ka jawab do. "
        "Agar jawab description mein nahi hai to saaf bolo ki video mein ye nahi mila.\n\n"
        f"[VIDEO DESCRIPTION]\n{description}\n\n[SAWAAL]\n{question}"
    )
    yield from _stream(prompt)


# ------------------------------------ step 2b: visual mode (seedha video)

def _find_range(description: str, question: str):
    """Description se pata lagata hai ki sawaal video ke kis hisse ka hai."""
    prompt = (
        "Neeche video ka description hai. Har 'VIDEO PART' header us tukde ka start dikhata hai, "
        "aur andar ke timestamps us start se gine gaye hain (header start + timestamp = asli time).\n"
        "Sawaal ka jawab video ke kis hisse mein hoga? Sirf 2 numbers likho: "
        "start_seconds,end_seconds (poori video ke hisaab se). Hisse ko chhota rakho (max 300 second). "
        "Kuch aur mat likho.\n\n"
        f"[DESCRIPTION]\n{description}\n\n[SAWAAL]\n{question}"
    )
    try:
        nums = re.findall(r"\d+", _generate(prompt))
        if len(nums) >= 2:
            start, end = int(nums[0]), int(nums[1])
            if end > start:
                return start, min(end, start + 300)
    except Exception as e:
        print(f"Range nahi mili: {e}")
    return None


def ask_video_visual(url: str, question: str):
    """Sawaal wale hisse ko seedha video se, zyada detail par dekhta hai."""
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
        "Is video ke hisse ko dhyan se dekh kar jawab do. Jo dikhta hai wahi batao, "
        "andaza mat lagao. Agar saaf nahi dikh raha to bolo.\n\nSawaal: " + question
    )
    contents = types.Content(parts=[video, types.Part(text=q)])
    yield from _stream(contents, resolution="MEDIA_RESOLUTION_MEDIUM")