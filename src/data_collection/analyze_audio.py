import librosa
import numpy as np
from pathlib import Path
import json
import soundfile as sf

def analyze_audio_file(audio_path):
    """Analyze an audio file to identify different segments."""
    print(f"\nAnalyzing: {audio_path.name}")
    
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Get segments with significant audio (non-silence)
    intervals = librosa.effects.split(
        y, 
        top_db=20,  # Adjust this value to change silence threshold
        frame_length=2048,
        hop_length=512
    )
    
    # Convert intervals to seconds
    segments = []
    for start, end in intervals:
        start_time = librosa.samples_to_time(start, sr=sr)
        end_time = librosa.samples_to_time(end, sr=sr)
        
        # Get audio features for this segment
        segment_audio = y[start:end]
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=segment_audio)[0].mean()
        
        # Compute spectral features
        spec_cent = librosa.feature.spectral_centroid(y=segment_audio, sr=sr)[0].mean()
        spec_bw = librosa.feature.spectral_bandwidth(y=segment_audio, sr=sr)[0].mean()
        
        # Simple heuristic to classify segment
        # Music typically has higher spectral bandwidth and centered frequency
        is_likely_music = spec_bw > 2000 and spec_cent > 2000
        
        segments.append({
            'start': float(start_time),
            'end': float(end_time),
            'duration': float(end_time - start_time),
            'rms_energy': float(rms),
            'spectral_centroid': float(spec_cent),
            'spectral_bandwidth': float(spec_bw),
            'likely_music': int(is_likely_music)  # Convert boolean to int for JSON serialization
        })
    
    return {
        'filename': str(audio_path.name),  # Ensure string
        'duration': float(duration),
        'num_segments': int(len(segments)),  # Ensure int
        'segments': segments
    }

def analyze_all_interviews():
    """Analyze all interview audio files."""
    raw_dir = Path("data/olamide/raw")
    analysis_dir = Path("data/olamide/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all MP3 files
    audio_files = list(raw_dir.glob("*.mp3"))
    
    all_analyses = {}
    
    for audio_path in audio_files:
        try:
            analysis = analyze_audio_file(audio_path)
            all_analyses[audio_path.name] = analysis
            
            # Print summary
            print(f"\nSummary for {audio_path.name}:")
            print(f"Duration: {analysis['duration']:.2f} seconds")
            print(f"Number of segments: {analysis['num_segments']}")
            
            music_segments = sum(1 for seg in analysis['segments'] if seg['likely_music'])
            print(f"Likely music segments: {music_segments}")
            
            # Calculate average segment length
            avg_seg_len = np.mean([seg['duration'] for seg in analysis['segments']])
            print(f"Average segment length: {avg_seg_len:.2f} seconds")
            
        except Exception as e:
            print(f"Error analyzing {audio_path}: {str(e)}")
    
    # Save analysis results
    output_path = analysis_dir / "audio_analysis.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_analyses, f, indent=2, ensure_ascii=False)
    
    print(f"\nAnalysis results saved to: {output_path}")

if __name__ == "__main__":
    analyze_all_interviews() 