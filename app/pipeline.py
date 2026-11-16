import os
import subprocess
import numpy as np
import imageio_ffmpeg
import yt_dlp
import whisper
from pathlib import Path

class ProcessingPipeline:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        print("Loading Whisper model...")
        self.model = whisper.load_model("base")

    def process_video(self, url):
        """YouTube video processing pipeline"""
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)

        print("Downloading audio via yt-dlp...")
        ydl_opts = {
            'format': 'bestaudio',
            'outtmpl': os.path.join(output_dir, 'downloaded_audio.%(ext)s'),
            'noplaylist': True,
            'postprocessors': [],
            'quiet': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

        print(f"Downloaded raw file: {filename}")

        # Convert audio directly to 16kHz mono PCM numpy array
        print("Processing audio into NumPy array...")
        cmd = [
            self.ffmpeg_exe, '-nostdin', '-y',
            '-i', filename,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ac', '1',
            '-ar', '16000',
            '-'
        ]

        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        audio_np = np.frombuffer(process.stdout, dtype=np.int16).flatten().astype(np.float32) / 32768.0

        print("Transcribing audio (this may take a moment)...")
        result = self.model.transcribe(audio_np)
        
        # Cleanup
        if os.path.exists(filename):
            os.remove(filename)
            
        return result["text"]

if __name__ == "__main__":
    pipeline = ProcessingPipeline()
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print("Test transcript:", pipeline.process_video(test_url))
