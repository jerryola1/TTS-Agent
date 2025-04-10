import librosa
import soundfile as sf
from pathlib import Path
import numpy as np
from test_deepfilternet import process_audio as deepfilter_process_audio
import os

def extract_segment(audio_path, start_time, end_time, output_path):
    """Extract a segment from an audio file, clean it, and save only the cleaned version."""
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Convert times to samples
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    # Extract segment
    segment = y[start_sample:end_sample]
    
    # Save temporary original segment
    temp_path = output_path.with_stem(f"{output_path.stem}_temp")
    sf.write(temp_path, segment, sr)
    
    # Apply noise reduction and save cleaned version
    cleaned_path = output_path.with_stem(f"{output_path.stem}_cleaned")
    deepfilter_process_audio(str(temp_path), str(cleaned_path))
    
    # Remove temporary file
    os.remove(temp_path)
    
    # Rename cleaned file to final name
    final_path = output_path.with_stem(output_path.stem)
    os.rename(cleaned_path, final_path)
    print(f"Saved cleaned segment {start_time:.2f}-{end_time:.2f} to {final_path}")

def main():
    # Create reference samples directory
    ref_dir = Path("data/olamide/reference_samples")
    ref_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract segments from "The Juice - Olamide"
    juice_audio = Path("data/olamide/processed/The Juice - Olamide/clean_segments/The Juice - Olamide_segment_0001_941.55s.wav")
    
    # Define segments to extract (in seconds)
    segments = [
        (21, 25),
        (127, 137),  # 2:07-2:17
        (156, 165),  # 2:36-2:45
        (202, 239),  # 3:22-3:59
        (262, 274)   # 4:22-4:34
    ]
    
    # Extract each segment
    for i, (start, end) in enumerate(segments):
        output_path = ref_dir / f"juice_olamide_ref_{i+1}_{start}-{end}s.wav"
        extract_segment(juice_audio, start, end, output_path)
    
    # Extract full segment from "Olamide, Talking about his carrer Sofar"
    career_audio = Path("data/olamide/processed/Olamide, Talking about his carrer Sofar/clean_segments/Olamide, Talking about his carrer Sofar_segment_0000_31.61s.wav")
    output_path = ref_dir / "career_olamide_ref_full.wav"
    extract_segment(career_audio, 0, 31.61, output_path)
    
    print("\nAll reference samples cleaned and saved successfully!")

if __name__ == "__main__":
    main() 