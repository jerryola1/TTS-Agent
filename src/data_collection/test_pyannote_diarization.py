from pyannote.audio import Pipeline
import torch
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_diarization(audio_path, output_file=None, use_gpu=True):
    """Test diarization on an audio file and save results to a file."""
    print(f"Testing diarization on: {audio_path}")
    
    # --- Get HF Token from Environment --- 
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: Hugging Face token (HF_TOKEN) not found in environment variables.")
        print("Please ensure it is set in your .env file or environment.")
        return None
    # --- End HF Token Check --- 
    
    if output_file is None:
        output_file = Path(audio_path).with_suffix('.diarization.json')
    
    try:
        # Force CUDA usage if available and requested
        if use_gpu and torch.cuda.is_available():
            print("CUDA is available, forcing GPU usage")
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            device = torch.device("cuda")
        else:
            if use_gpu and not torch.cuda.is_available():
                print("CUDA requested but not available, falling back to CPU")
            else:
                print("Using CPU as requested")
            device = torch.device("cpu")
        
        # Load the model
        print(f"Loading Pyannote diarization pipeline on {device}...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Move model to device
        pipeline.to(device)
        print(f"Successfully loaded pipeline on {device}")
        
        # Run diarization
        print("Running diarization (this may take a few minutes for longer files)...")
        diarization = pipeline(audio_path)
        
        # Collect results
        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
                "duration": turn.end - turn.start
            })
        
        # Count speakers and talk time
        speaker_talk_time = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            duration = turn.end - turn.start
            if speaker not in speaker_talk_time:
                speaker_talk_time[speaker] = 0
            speaker_talk_time[speaker] += duration
        
        # Save results to file
        with open(output_file, 'w') as f:
            json.dump({
                "segments": results,
                "speaker_stats": {
                    speaker: {
                        "talk_time_seconds": talk_time,
                        "talk_time_minutes": talk_time / 60
                    } for speaker, talk_time in speaker_talk_time.items()
                }
            }, f, indent=2)
        
        print(f"Diarization results saved to: {output_file}")
        
        # Print summary
        print("\nSpeaker statistics:")
        for speaker, talk_time in speaker_talk_time.items():
            print(f"Speaker {speaker}: {talk_time:.1f} seconds ({talk_time/60:.1f} minutes)")
        
        return diarization
        
    except Exception as e:
        print(f"Error in diarization: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Pyannote diarization")
    parser.add_argument("--audio", type=str, default="data/olamide/raw/The Juice - Olamide_sample.mp3",
                      help="Path to audio file to test")
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save diarization results (default: same as audio with .diarization.json)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    args = parser.parse_args()
    
    diarization = test_diarization(args.audio, args.output, use_gpu=not args.no_gpu) 