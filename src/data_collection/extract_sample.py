import librosa
import soundfile as sf
from pathlib import Path

def extract_sample(audio_path, output_path=None, start_time=60, duration=30):
    """Extract a short sample from an audio file for testing."""
    if output_path is None:
        output_path = Path(audio_path).with_stem(f"{Path(audio_path).stem}_sample")
    
    print(f"Extracting {duration}s sample from {audio_path}, starting at {start_time}s")
    
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None, offset=start_time, duration=duration)
    
    # Save the sample
    sf.write(output_path, y, sr)
    
    print(f"Saved sample to: {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract sample from audio")
    parser.add_argument("--audio", type=str, default="data/olamide/raw/The Juice - Olamide.mp3",
                      help="Path to audio file")
    parser.add_argument("--start", type=int, default=60,
                      help="Start time in seconds")
    parser.add_argument("--duration", type=int, default=30,
                      help="Duration in seconds")
    args = parser.parse_args()
    
    sample_path = extract_sample(args.audio, start_time=args.start, duration=args.duration) 