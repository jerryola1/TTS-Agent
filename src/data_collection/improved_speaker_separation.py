import os
import json
import logging
import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
import torch
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

# Demucs for music separation
from demucs.pretrained import get_model
from demucs.apply import apply_model

# Pyannote for speaker diarization
from pyannote.audio import Pipeline

# Resemblyzer for voice embeddings
from resemblyzer import VoiceEncoder, preprocess_wav

class ImprovedSpeakerSeparation:
    def __init__(self, base_dir="data/audio", use_cuda=True, hf_token=None):
        """Initialize the speaker separation with base directory structure."""
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        
        # Create necessary directories
        for dir_path in [self.raw_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.setup_logging()
        
        # Force CUDA usage if available
        use_cuda = use_cuda and torch.cuda.is_available()
        if use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            
        # Initialize device
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.logger.info(f"Using device: {self.device}")
        print(f"Using device: {self.device}")
        
        # Set HuggingFace token as environment variable
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        
        # Load Demucs model for music separation
        self.logger.info("Loading Demucs model for music separation...")
        print("Loading Demucs model for music separation...")
        """
        Demucs is a music source separation model that separates audio into:
        1. Vocals (human voices)
        2. Drums
        3. Bass
        4. Other instruments
        
        For interviews, we primarily want the vocals track, which isolates 
        human speech from background music, making speaker diarization more accurate.
        """
        self.demucs = get_model("htdemucs")
        self.demucs.to(self.device)
        
        # Load voice encoder for embedding generation
        self.logger.info("Loading VoiceEncoder model from Resemblyzer...")
        print("Loading VoiceEncoder model from Resemblyzer...")
        self.voice_encoder = VoiceEncoder(device=self.device)
        
        try:
            # Loading Pyannote pipeline for diarization
            self.logger.info("Loading Pyannote diarization pipeline...")
            print("Loading Pyannote diarization pipeline...")
            
            token = hf_token if hf_token else os.environ.get("HF_TOKEN", None)
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            )
            # Force GPU usage for Pyannote
            self.diarization_pipeline.to(self.device)
            print(f"Successfully loaded Pyannote diarization model on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Could not load Pyannote diarization model: {str(e)}")
            print(f"Error: Could not load Pyannote diarization model: {str(e)}")
            print("You must accept the licenses for all models at:")
            print("  - https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("  - https://huggingface.co/pyannote/segmentation-3.0")
            print("  - https://huggingface.co/pyannote/embedding")
            print("And your token must have 'read' access.")
            raise ValueError("Pyannote model could not be loaded. See error above.")
    
    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"speaker_separation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def process_interview(self, audio_file_path: str):
        """Process an interview file through the entire pipeline."""
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            self.logger.error(f"Audio file not found: {audio_path}")
            print(f"Error: Audio file not found: {audio_path}")
            return False
        
        # Create interview-specific directories
        interview_name = audio_path.stem
        interview_dir = self.processed_dir / interview_name
        interview_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        music_removed_dir = interview_dir / "music_removed"
        music_removed_dir.mkdir(exist_ok=True)
        
        speaker_segments_dir = interview_dir / "speaker_segments"
        speaker_segments_dir.mkdir(exist_ok=True)
        
        # Step 1: Music separation with Demucs
        self.logger.info(f"Step 1: Separating music from speech in {interview_name}")
        print(f"Step 1: Separating music from speech in {interview_name}")
        
        vocals_path = self.separate_music(audio_path, music_removed_dir)
        if not vocals_path:
            return False
        
        # Step 2: Speaker diarization with Pyannote
        self.logger.info(f"Step 2: Performing speaker diarization on {vocals_path}")
        print(f"Step 2: Performing speaker diarization on {vocals_path}")
        
        diarization = self.perform_diarization(vocals_path)
        if not diarization:
            self.logger.error("Diarization failed, cannot continue")
            print("Error: Diarization failed, cannot continue")
            return False
        
        # Step 3: Extract speaker segments based on diarization
        self.logger.info("Step 3: Extracting speaker segments from diarization")
        print("Step 3: Extracting speaker segments from diarization")
        
        speaker_segments = self.extract_speaker_segments(vocals_path, diarization, speaker_segments_dir)
        
        # Step 4: Identify the primary speaker (the artist/interviewee)
        self.logger.info("Step 4: Identifying the primary speaker (artist/interviewee)")
        print("Step 4: Identifying the primary speaker (artist/interviewee)")
        
        primary_speaker, segments = self.identify_primary_speaker(speaker_segments)
        
        # Save results
        self.save_results(interview_name, primary_speaker, segments)
        
        return True
    
    def separate_music(self, audio_path: Path, output_dir: Path) -> Optional[Path]:
        """Separate music from speech using Demucs."""
        try:
            # Load audio
            self.logger.info(f"Loading audio file: {audio_path}")
            print(f"Loading audio file: {audio_path}")
            
            # For demucs, we need to load audio at 44.1kHz
            waveform, sample_rate = librosa.load(audio_path, sr=44100, mono=False)
            
            # If mono, convert to stereo (demucs expects stereo)
            if waveform.ndim == 1:
                waveform = np.stack([waveform, waveform])
            
            # Convert to torch tensor
            waveform_tensor = torch.tensor(waveform, device=self.device)
            
            # Apply Demucs model
            self.logger.info("Applying Demucs model to separate sources")
            print("Applying Demucs model to separate sources...")
            
            # Get the audio sources
            sources = apply_model(self.demucs, waveform_tensor[None])[0]
            sources_np = sources.cpu().numpy()
            
            # Sources are ordered as: drums, bass, other, vocals
            vocals = sources_np[3]  # Get only the vocals
            
            # Save vocals track
            vocals_path = output_dir / f"{audio_path.stem}_vocals.wav"
            sf.write(vocals_path, vocals.T, sample_rate)
            
            self.logger.info(f"Saved vocals track to {vocals_path}")
            print(f"Saved vocals track to {vocals_path}")
            
            return vocals_path
            
        except Exception as e:
            self.logger.error(f"Error in music separation: {str(e)}")
            print(f"Error in music separation: {str(e)}")
            return None
    
    def perform_diarization(self, audio_path: Path) -> Optional[Any]:
        """Perform speaker diarization using Pyannote."""
        try:
            self.logger.info(f"Running diarization on {audio_path}")
            print(f"Running diarization on {audio_path}...")
            
            # Run the diarization pipeline
            diarization = self.diarization_pipeline(audio_path)
            
            # Count speakers and talk time
            speaker_talk_time = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                duration = turn.end - turn.start
                if speaker not in speaker_talk_time:
                    speaker_talk_time[speaker] = 0
                speaker_talk_time[speaker] += duration
            
            # Print statistics
            self.logger.info(f"Identified {len(speaker_talk_time)} speakers:")
            print(f"Identified {len(speaker_talk_time)} speakers:")
            for speaker, talk_time in speaker_talk_time.items():
                self.logger.info(f"  Speaker {speaker}: {talk_time:.2f} seconds")
                print(f"  Speaker {speaker}: {talk_time:.2f} seconds")
            
            return diarization
            
        except Exception as e:
            self.logger.error(f"Error in speaker diarization: {str(e)}")
            print(f"Error in speaker diarization: {str(e)}")
            return None
    
    def extract_speaker_segments(
        self, audio_path: Path, diarization, output_dir: Path
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract speaker segments based on diarization results."""
        try:
            # Load audio
            waveform, sample_rate = librosa.load(str(audio_path), sr=None)
            
            # Create a dict to store segments by speaker
            speaker_segments = {}
            
            # Extract segments for each speaker
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_sample = int(turn.start * sample_rate)
                end_sample = int(turn.end * sample_rate)
                
                # Skip if segment is too short (less than 1 second)
                if turn.end - turn.start < 1.0:
                    continue
                
                # Extract segment audio
                segment_audio = waveform[start_sample:end_sample]
                
                # Create segment filename with speaker and timestamps
                segment_filename = f"{audio_path.stem}_{speaker}_{turn.start:.2f}_{turn.end:.2f}.wav"
                segment_path = output_dir / segment_filename
                
                # Save segment
                sf.write(segment_path, segment_audio, sample_rate)
                
                # Store segment info
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                
                speaker_segments[speaker].append({
                    "path": str(segment_path),
                    "start": turn.start,
                    "end": turn.end,
                    "duration": turn.end - turn.start
                })
                
                self.logger.info(f"Saved segment: {segment_path.name}")
            
            # Print segment counts
            for speaker, segments in speaker_segments.items():
                total_duration = sum(seg["duration"] for seg in segments)
                self.logger.info(f"Speaker {speaker}: {len(segments)} segments, {total_duration:.2f} seconds")
                print(f"Speaker {speaker}: {len(segments)} segments, {total_duration:.2f} seconds")
            
            return speaker_segments
            
        except Exception as e:
            self.logger.error(f"Error extracting speaker segments: {str(e)}")
            print(f"Error extracting speaker segments: {str(e)}")
            return {}
    
    def identify_primary_speaker(self, speaker_segments: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Identify the primary speaker (the artist/interviewee) based on speaking time."""
        try:
            if not speaker_segments:
                self.logger.error("No speaker segments to identify")
                print("Error: No speaker segments to identify")
                return "", []
            
            self.logger.info("Identifying primary speaker (artist/interviewee)")
            print("Identifying primary speaker (artist/interviewee)...")
            
            # Calculate total duration for each speaker
            speaker_durations = {}
            for speaker, segments in speaker_segments.items():
                duration = sum(segment["duration"] for segment in segments)
                speaker_durations[speaker] = duration
            
            # Primary speaker is likely the one with most talking time
            if speaker_durations:
                primary_speaker = max(speaker_durations, key=speaker_durations.get)
                
                self.logger.info(f"Identified speaker {primary_speaker} as the primary speaker")
                print(f"Identified speaker {primary_speaker} as the primary speaker")
                print(f"Total duration: {speaker_durations[primary_speaker]:.2f} seconds")
                
                return primary_speaker, speaker_segments[primary_speaker]
            else:
                self.logger.error("Could not determine speaker durations")
                print("Error: Could not determine speaker durations")
                return "", []
                
        except Exception as e:
            self.logger.error(f"Error identifying primary speaker: {str(e)}")
            print(f"Error identifying primary speaker: {str(e)}")
            return "", []
    
    def save_results(self, interview_name: str, primary_speaker: str, segments: List[Dict[str, Any]]):
        """Save processing results to JSON."""
        try:
            # Create interview-specific directory
            interview_dir = self.processed_dir / interview_name
            
            # Calculate statistics
            total_duration = sum(segment["duration"] for segment in segments)
            
            # Save results metadata
            results = {
                'processed_date': datetime.now().isoformat(),
                'primary_speaker': primary_speaker,
                'total_segments': len(segments),
                'segment_paths': [segment["path"] for segment in segments],
                'segment_durations': [segment["duration"] for segment in segments],
                'segment_timestamps': [{"start": segment["start"], "end": segment["end"]} for segment in segments],
                'total_duration': total_duration,
            }
            
            results_file = interview_dir / "processing_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results saved to {results_file}")
            print(f"Results saved to {results_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            print(f"Error saving results: {str(e)}")

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Improved Speaker Separation")
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument("--output-dir", type=str, default="data/audio", 
                      help="Base output directory (default: data/audio)")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--token", type=str, help="HuggingFace token for accessing models")
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = ImprovedSpeakerSeparation(
            base_dir=args.output_dir,
            use_cuda=not args.no_cuda, 
            hf_token=args.token
        )
        
        if args.audio:
            # Process single audio file
            processor.process_interview(args.audio)
        else:
            # If no audio file specified, print usage
            print("Please specify an audio file with --audio")
            print("Example: python improved_speaker_separation.py --audio data/raw/interview.mp3 --token YOUR_HF_TOKEN")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check that you have accepted all required model licenses on HuggingFace")

if __name__ == "__main__":
    main() 