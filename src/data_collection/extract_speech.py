import librosa
import soundfile as sf
import json
from pathlib import Path
import numpy as np

def load_analysis():
    """Load the audio analysis results."""
    analysis_path = Path("data/olamide/analysis/audio_analysis.json")
    with open(analysis_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_nearby_segments(segments, max_gap=0.5):
    """Merge segments that are close to each other."""
    if not segments:
        return []
    
    # Sort segments by start time
    segments = sorted(segments, key=lambda x: x['start'])
    
    merged = []
    current = segments[0]
    
    for next_seg in segments[1:]:
        if next_seg['start'] - current['end'] <= max_gap:
            # Merge segments
            current = {
                'start': current['start'],
                'end': next_seg['end'],
                'duration': next_seg['end'] - current['start']
            }
        else:
            merged.append(current)
            current = next_seg
    
    merged.append(current)
    return merged

def extract_speech_segments(audio_path, segments, min_duration=2.0):
    """Extract speech segments from audio file."""
    print(f"\nProcessing: {audio_path.name}")
    
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Create output directory
    output_dir = Path("data/olamide/speech_segments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract each segment
    extracted_segments = []
    for i, segment in enumerate(segments):
        duration = segment['duration']
        if duration < min_duration:
            continue
            
        # Convert time to samples
        start_sample = int(segment['start'] * sr)
        end_sample = int(segment['end'] * sr)
        
        # Extract segment audio
        segment_audio = y[start_sample:end_sample]
        
        # Normalize volume
        segment_audio = librosa.util.normalize(segment_audio)
        
        # Generate output filename
        base_name = audio_path.stem
        segment_name = f"{base_name}_segment_{i:04d}_{duration:.2f}s.wav"
        output_path = output_dir / segment_name
        
        # Save segment
        sf.write(output_path, segment_audio, sr)
        extracted_segments.append({
            'file': str(output_path),
            'duration': duration,
            'start': segment['start'],
            'end': segment['end']
        })
        
        if i % 10 == 0:
            print(f"Extracted {i} segments...")
    
    return extracted_segments

def process_interview(filename, min_duration=2.0, max_gap=1.0):
    """Process a single interview, extracting speech segments."""
    # Load analysis
    analysis = load_analysis()
    if filename not in analysis:
        print(f"Analysis not found for {filename}")
        return
    
    # Get non-music segments
    segments = [
        seg for seg in analysis[filename]['segments']
        if not seg['likely_music']
    ]
    
    # Merge nearby segments
    print(f"\nMerging segments (max gap: {max_gap}s)...")
    merged_segments = merge_nearby_segments(segments, max_gap)
    print(f"Merged {len(segments)} segments into {len(merged_segments)} segments")
    
    # Extract segments
    audio_path = Path("data/olamide/raw") / filename
    extracted = extract_speech_segments(audio_path, merged_segments, min_duration)
    
    # Save segment info
    output_dir = Path("data/olamide/speech_segments")
    info_path = output_dir / f"{audio_path.stem}_segments.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump({
            'source_file': str(audio_path),
            'segments': extracted
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nExtracted {len(extracted)} segments")
    print(f"Segment information saved to: {info_path}")

if __name__ == "__main__":
    # Clean up existing segments
    segments_dir = Path("data/olamide/speech_segments")
    if segments_dir.exists():
        for file in segments_dir.glob("*.wav"):
            file.unlink()
        for file in segments_dir.glob("*.json"):
            file.unlink()
    
    # Process all interviews
    interviews = [
        "EXCLUSIVES! Special - Olamide & Tony Elumelu (Oil & Gas).mp3",
        "EXCLUSIVE INTERVIEW WITH OLAMIDE (Nigerian Entertainment News).mp3",
        "Olamide： 'I May Not Marry My BabyMama, Never Met DaGrin' ｜ The TakeOver with Moet Abebe.mp3",
        "ONE on ONE with Olamide.mp3",
        "The Juice - Olamide.mp3"
    ]
    
    for interview in interviews:
        print(f"\nProcessing interview: {interview}")
        process_interview(interview, min_duration=2.0, max_gap=1.0) 