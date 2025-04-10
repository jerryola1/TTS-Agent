import sounddevice as sd
import soundfile as sf
from pathlib import Path
import random
import time

def play_audio(file_path):
    """Play an audio file and wait for it to finish."""
    data, samplerate = sf.read(file_path)
    print(f"\nPlaying: {file_path.name}")
    sd.play(data, samplerate)
    sd.wait()
    time.sleep(1)

def play_olamide_samples(num_samples=5):
    """Play random samples of segments identified as Olamide's voice."""
    olamide_dir = Path("data/olamide/olamide_segments")
    
    # Get all WAV files
    wav_files = list(olamide_dir.glob("*.wav"))
    if not wav_files:
        print("No Olamide segments found!")
        return
    
    # Select random samples
    samples = random.sample(wav_files, min(num_samples, len(wav_files)))
    
    print(f"\nPlaying {len(samples)} random samples of Olamide's voice...")
    print("Press Ctrl+C to stop playback")
    
    try:
        for sample in samples:
            play_audio(sample)
    except KeyboardInterrupt:
        sd.stop()
        print("\nPlayback stopped by user")

if __name__ == "__main__":
    play_olamide_samples() 