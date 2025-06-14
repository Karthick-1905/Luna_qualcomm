import subprocess
from pathlib import Path
from typing import Union, List, Tuple
from utils.logger import setup_logger,logger

setup_logger()

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("FFmpeg not found. Please install FFmpeg and add it to PATH.")
    


def extract_audio(video_path: Union[str, Path], 
                 output_dir: Union[str, Path],
                 sample_rate: int = 44100, 
                 channels: int = 2) -> str:
    check_ffmpeg()
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    audio_dir =  output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    audio_filename = f"{video_path.stem}_extracted.wav"
    audio_output = audio_dir / audio_filename
    
    cmd = [
        "ffmpeg", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", str(channels), "-y", str(audio_output)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return str(audio_output)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg extraction failed: {e.stderr}")
        raise