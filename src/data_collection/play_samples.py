import sounddevice as sd
import soundfile as sf
from pathlib import Path
import json
import time

def play_audio(file_path):
    """Play an audio file and wait for it to finish."""
    data, samplerate = sf.read(file_path)
    print(f"\nPlaying: {file_path.name}")
    sd.play(data, samplerate)
    sd.wait()  # Wait until file is done playing
    time.sleep(1)  # Add a small gap between files

def play_sample_segments():
    """Play a few sample segments of different durations."""
    segments_dir = Path("data/olamide/speech_segments")
    
    # Get all WAV files
    wav_files = list(segments_dir.glob("*.wav"))
    
    # Sort by duration (which is in the filename)
    wav_files.sort(key=lambda x: float(x.stem.split("_")[-1].replace("s", "")))
    
    # Select samples: shortest, longest, and three from middle
    samples = [
        wav_files[0],  # shortest
        wav_files[len(wav_files)//4],  # 25th percentile
        wav_files[len(wav_files)//2],  # median
        wav_files[3*len(wav_files)//4],  # 75th percentile
        wav_files[-1]  # longest
    ]
    
    print("\nPlaying 5 sample segments to verify quality...")
    print("Press Ctrl+C to stop playback")
    
    try:
        for sample in samples:
            play_audio(sample)
    except KeyboardInterrupt:
        sd.stop()
        print("\nPlayback stopped by user")

if __name__ == "__main__":
    play_sample_segments() 