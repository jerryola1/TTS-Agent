import json
import logging
from pathlib import Path
import shutil
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf
from typing import List, Dict, Any, Optional, Tuple
import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition
import argparse
import sys
import traceback
from urllib.parse import urlparse

from data_fetcher import DataFetcher
from extract_audio import extract_audio_from_videos
from analyze_audio import analyze_audio_file
# from test_deepfilternet import process_audio as deepfilter_process_audio
from test_pyannote_diarization import test_diarization
# from test_spleeter import test_spleeter
from test_speaker_verification import verify_speaker, create_speaker_embedding, verify_speaker_segments
from downloaders import DownloadHandler
# from src.utils.logging_config import setup_logging
# from src.utils.file_utils import load_sources, save_sources, safe_filename
from test_deepfilternet import process_audio_in_chunks as apply_deepfilter_chunked
# Import the loudness normalization function
from test_loudness_norm import normalize_audio_loudness
# Import the whisper transcription function
from test_whisper_transcription import transcribe_audio_files

# Add project root to sys.path to allow imports from other modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Diarization (assuming a placeholder or actual implementation)
try:
    # Try importing the actual diarization function if it exists
    from src.audio_processing.diarization import diarize_audio 
except ImportError:
    print("Warning: Diarization module not found. Using placeholder.")
    # Placeholder function if the actual module isn't ready
    def diarize_audio(audio_path: Path, output_dir: Path, num_speakers: Optional[int] = None) -> Optional[Path]:
        print(f"[Placeholder] Diarizing {audio_path} -> {output_dir}")
        # Simulate creating an output file
        simulated_output = output_dir / f"{audio_path.stem}.rttm"
        simulated_output.touch()
        return simulated_output

# Correct the import: use 'test_spleeter' function from test_spleeter.py
from test_spleeter import test_spleeter as spleeter_process_chunked

class InterviewAgent:
    def __init__(self, base_dir: str = "data", artist_name: Optional[str] = None):
        """Initialize the interview agent with base directory structure."""
        self.base_dir = Path(base_dir)
        self.artist_name = artist_name
        self.artist_dir = self.base_dir / (artist_name.lower() if artist_name else "unknown_artist")
        
        # Create artist-specific directories
        self.raw_dir = self.artist_dir / "raw"
        self.processed_dir = self.artist_dir / "processed"
        self.analysis_dir = self.artist_dir / "analysis"
        self.visualizations_dir = self.artist_dir / "visualizations"
        
        # Create necessary directories
        for dir_path in [self.raw_dir, self.processed_dir, self.analysis_dir, self.visualizations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        
        # Initialize download handler (it creates its own logger)
        self.download_handler = DownloadHandler(
            base_dir=self.base_dir,
            artist_name=self.artist_name
        )
        
        # Initialize data fetcher
        self.fetcher = DataFetcher()
        
        # Load potential sources
        self.sources = self._load_potential_sources()
        print(f"Loaded {len(self.sources)} potential sources")
    
    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = self.artist_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_potential_sources(self) -> List[Dict[str, Any]]:
        """Load potential sources from JSON file."""
        sources_file = Path("data/potential_sources.json")
        if not sources_file.exists():
            self.logger.error("potential_sources.json not found!")
            print(f"Could not find potential_sources.json at {sources_file.absolute()}")
            return []
        
        try:
            with open(sources_file, 'r', encoding='utf-8') as f:
                sources = json.load(f)
                print(f"Successfully loaded {len(sources)} sources from {sources_file}")
                
                # Filter out non-artist interviews
                excluded_titles = [
                    "Olamide is a GENIUS! Tiwa Savage Showers Olamide With Praise #shorts",
                    # Add other titles to exclude here
                ]
                
                filtered_sources = [s for s in sources if s['title'] not in excluded_titles]
                print(f"Filtered out {len(sources) - len(filtered_sources)} non-artist interviews")
                
                return filtered_sources
        except Exception as e:
            self.logger.error(f"Error loading potential sources: {str(e)}")
            print(f"Error loading potential sources: {str(e)}")
            return []
    
    def process_single_source(self, source_index: Optional[int] = None, source_url: Optional[str] = None):
        """Process a single source based on index or URL."""
        if source_index is not None:
            if source_index < 0 or source_index >= len(self.sources):
                print(f"Invalid source index. Please provide a number between 0 and {len(self.sources)-1}")
                return
            source = self.sources[source_index]
            print(f"\nProcessing source by index {source_index}: {source.get('title', '[No Title]')}")
            try:
                 self.process_source(source)
            except Exception as e:
                 self.logger.error(f"Error processing source {source.get('title', '')}: {str(e)}", exc_info=True)
                 print(f"Error processing source: {str(e)}")

        elif source_url is not None:
            # If URL is provided, don't look it up in sources.json.
            # Treat it as a direct request to process this URL.
            # Create a minimal source dictionary for process_source.
            # TODO: Consider adding a way to fetch/provide a better title if possible.
            self.logger.info(f"Processing direct URL: {source_url}")
            print(f"\nProcessing direct URL: {source_url}")
            source_dict = {
                "url": source_url, 
                "title": f"Direct URL - {Path(urlparse(source_url).path).name}" # Placeholder title
                # Add artist if we have it? The agent has self.artist_name
                # "artist": self.artist_name 
            }
            try:
                 self.process_source(source_dict)
            except Exception as e:
                 self.logger.error(f"Error processing direct URL {source_url}: {str(e)}", exc_info=True)
                 print(f"Error processing direct URL: {str(e)}")
            # source = next((s for s in self.sources if s['url'] == source_url), None)
            # if not source:
            #     print(f"No source found with URL: {source_url}") # Original incorrect logic
            #     return
        else:
            print("Please provide either a source index or URL")
            return
    
    def process_source(self, source: Dict[str, Any]):
        """Process a single source through the complete pipeline."""
        self.logger.info(f"Processing source: {source['title']}")
        
        # Sanitize title for directory creation
        safe_title = source['title'].replace(':', '-').replace('"', '').replace('\\', '-').replace('/', '-').replace('|', '-').replace('?', '').replace('*', '')
        safe_title = safe_title[:150]  # Limit length

        # Create interview-specific directories using the sanitized title
        interview_dir = self.processed_dir / safe_title
        interview_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Download the interview using the download handler
        audio_file = self.download_handler.download(source['url'], source)
        if not audio_file:
            self.logger.warning(f"Skipping source {safe_title} due to download failure.")
            return
        
        # Step 2: Extract audio if needed (should already be mp3)
        if audio_file.suffix.lower() not in ['.mp3', '.wav', '.ogg', '.flac']:
            self.logger.info(f"Audio file {audio_file} has unexpected suffix. Attempting processing anyway.")
            # If it's mp4, try extracting
            if audio_file.suffix.lower() == '.mp4':
                extracted_audio = self._extract_audio(audio_file)
                if not extracted_audio:
                    self.logger.warning(f"Failed to extract audio from {audio_file.name}, skipping source.")
                    return
                audio_file = extracted_audio  # Update audio_file to the extracted one
        
        # === Step 3: Separate vocals from music using Spleeter (Updated Logic) ===
        spleeter_output_dir = interview_dir / "spleeter_output"
        vocals_file = self._separate_vocals(audio_file, spleeter_output_dir)
        if not vocals_file:
             # If Spleeter fails, we might want to continue with the original audio
             # depending on whether music separation is critical vs nice-to-have.
             # For TTS, clean vocals are very important, so we'll skip if it fails.
             self.logger.warning(f"Vocal separation failed for {safe_title}. Skipping further processing.")
             # Alternatively, could use original audio: vocals_file = audio_file
             return
        
        # Ensure vocals_file is defined even if Spleeter fails, maybe set to None
        if not vocals_file or not vocals_file.exists():
            self.logger.error(f"Spleeter did not produce a valid vocals file at {vocals_file}. Skipping source {safe_title}.")
            return 
        
        # === Step 4: Apply Noise Reduction === COMMENTED OUT ===
        # self.logger.info("Step 4: Applying noise reduction to vocals...")
        # # Define output path using the stem of the vocals file
        # cleaned_vocals_path = interview_dir / f"{vocals_file.stem}_cleaned.wav"
        # try:
        #     # Call the imported, tested chunked deepfilter function directly
        #     apply_deepfilter_chunked(
        #         str(vocals_file),
        #         str(cleaned_vocals_path)
        #         # Default chunk/overlap settings from test_deepfilternet will be used
        #     )
        #     # Check if the output file was actually created
        #     if not cleaned_vocals_path.exists():
        #         # If the function didn't raise an error but file is missing, log and stop
        #         self.logger.error(f"DeepFilterNet processing seemed to finish but output file is missing: {cleaned_vocals_path}")
        #         self.logger.warning(f"Skipping source {safe_title} due to noise reduction failure.")
        #         return # Stop processing
        #     
        #     self.logger.info(f"Noise reduction successful. Cleaned vocals: {cleaned_vocals_path}")
        #     print(f"Noise reduction successful. Cleaned vocals: {cleaned_vocals_path}")
        #     
        # except Exception as e:
        #     # Log errors including traceback
        #     tb_str = traceback.format_exc()
        #     self.logger.error(f"Error during DeepFilterNet noise reduction for {vocals_file.name}: {e}\n{tb_str}")
        #     print(f"Error applying noise reduction to {vocals_file.name}: {e}")
        #     self.logger.warning(f"Skipping source {safe_title} due to noise reduction failure.")
        #     return # Stop processing if noise reduction fails
        # === END OF COMMENTED OUT Step 4 ===
            
        # Set the input for the next step (Normalization) to be the Spleeter output
        input_for_norm_path = vocals_file 
            
        # === Step 4.5: Loudness Normalization ===
        # (This step now processes the direct output from Spleeter)
        self.logger.info("Step 4.5: Normalizing loudness of Spleeter vocals...")
        # Adjust output filename to reflect input is direct from Spleeter
        normalized_vocals_path = interview_dir / f"{input_for_norm_path.stem}_normalized.wav" 
        try:
            target_lufs = -20.0 # Define target LUFS level
            final_vocals_path = normalize_audio_loudness(
                str(input_for_norm_path), # Use Spleeter output directly
                str(normalized_vocals_path),
                target_lufs
            )
            # Check if normalization succeeded
            if not final_vocals_path or not final_vocals_path.exists():
                 self.logger.error(f"Loudness normalization failed or did not produce output file: {normalized_vocals_path}")
                 self.logger.warning(f"Skipping source {safe_title} due to normalization failure.")
                 return # Stop if normalization fails
            
            self.logger.info(f"Loudness normalization successful. Target LUFS: {target_lufs}. Normalized file: {final_vocals_path}")
            print(f"Loudness normalization successful. Normalized file: {final_vocals_path}")
            
        except Exception as e:
            tb_str = traceback.format_exc()
            # Update error message to reflect input
            self.logger.error(f"Error during Loudness Normalization for {input_for_norm_path.name}: {e}\n{tb_str}")
            print(f"Error during Loudness Normalization for {input_for_norm_path.name}: {e}")
            self.logger.warning(f"Skipping source {safe_title} due to normalization error.")
            return

        # === Step 5: Speaker Diarization ===
        self.logger.info("Step 5: Diarizing speakers...")
        diarization_output_file = final_vocals_path.with_suffix('.diarization.json')
        try:
            # Use the imported test_diarization function
            # It handles GPU checking, loading model, running, saving JSON
            diarization_result = test_diarization(
                str(final_vocals_path),
                output_file=diarization_output_file
                # use_gpu=True # Assuming default is True if available
            )
            if diarization_result is None:
                # Error is already printed by test_diarization
                return

            # Reset default tensor type after diarization
            torch.set_default_tensor_type('torch.FloatTensor')
            self.logger.info("Reset default tensor type after diarization")

            self.logger.info(f"Diarization successful. Results saved to {diarization_output_file}")
            print(f"Diarization successful. Results saved to {diarization_output_file}")
        except Exception as e:
            self.logger.error(f"Error during diarization call for {final_vocals_path.name}: {e}")
            print(f"Error during diarization call for {final_vocals_path.name}: {e}")
            self.logger.warning(f"Skipping source {safe_title} due to diarization failure.")
            return # Stop processing if diarization fails

        # === Step 6: Analyze audio features (Optional) ===
        analysis_results = self._analyze_audio(final_vocals_path)
        self._display_audio_stats(analysis_results)

        # === Step 7: Target Speaker Segment Extraction (New Implementation) ===
        self.logger.info("Step 7: Extracting speaker segments based on diarization...")
        diarized_segments_dir = interview_dir / "diarized_segments"
        try:
            speaker_segments_data = self._extract_speaker_segments_from_diarization(
                final_vocals_path, diarization_result, diarized_segments_dir
            )
            if not speaker_segments_data:
                self.logger.warning(f"No speaker segments extracted for {safe_title}. Skipping.")
                return # Stop if no segments could be extracted
            self.logger.info(f"Successfully extracted segments for {len(speaker_segments_data)} speakers.")
            print(f"Successfully extracted segments for {len(speaker_segments_data)} speakers.")

        except Exception as e:
            self.logger.error(f"Error during speaker segment extraction for {safe_title}: {e}")
            print(f"Error during speaker segment extraction for {safe_title}: {e}")
            self.logger.warning(f"Skipping source {safe_title} due to segment extraction failure.")
            return

        # === Step 8: Speaker Verification [TO BE REFACTORED] ===
        self.logger.info("Step 8: Verifying speaker identity... [Refactoring Needed]")
        # TODO: Refactor _verify_speaker to take speaker_segments_data (dict) as input
        #       and iterate through speakers/segments, verifying against reference.
        olamide_dir = interview_dir / "olamide_segments" # Output dir for verified segments
        olamide_dir.mkdir(parents=True, exist_ok=True)
        # Assume _verify_speaker saves files to olamide_dir / "verified_segments" and returns a list
        verified_paths_list = self._verify_speaker(speaker_segments_data, olamide_dir) 
        
        # Define the directory path where verified files ARE LOCATED for Step 9
        verified_segments_dir = olamide_dir / "verified_segments" 

        # === Step 9: Transcription (Whisper) ===
        self.logger.info("Step 9: Transcribing verified segments using Whisper...")
        # Define output dir for transcripts based on the main interview_dir
        transcriptions_dir = interview_dir / "transcriptions"
        
        # Check if the verified_segments_dir (the DIRECTORY PATH) exists and has content
        # Use the correct variable 'verified_segments_dir' here
        if verified_segments_dir and verified_segments_dir.exists() and any(verified_segments_dir.iterdir()):
            try:
                # Call the imported transcription function using the DIRECTORY PATH variable
                transcribe_audio_files(
                    verified_segments_dir, # USE THIS DIRECTORY PATH
                    transcriptions_dir,    
                    model_name="medium"    
                )
                self.logger.info(f"Transcription successful. Transcripts saved in: {transcriptions_dir}")
                print(f"Transcription successful. Transcripts saved in: {transcriptions_dir}")
            except Exception as e:
                # Log errors including traceback
                tb_str = traceback.format_exc()
                self.logger.error(f"Error during Transcription for {safe_title}: {e}\n{tb_str}")
                print(f"Error during Transcription for {safe_title}: {e}")
                self.logger.warning(f"Skipping subsequent steps for {safe_title} due to transcription error.")
                return # Stop processing if transcription fails
        elif not verified_segments_dir or not verified_segments_dir.exists():
             # Use the directory path variable in the log message
             self.logger.warning(f"Verified segments directory not found ({verified_segments_dir}). Skipping transcription.")
             return # Cannot proceed without verified segments to transcribe
        else: # Directory exists but is empty
             # Use the directory path variable in the log message
             self.logger.warning(f"No verified segments found in {verified_segments_dir} to transcribe. Stopping processing for this source.")
             return
             
        # === Step 10: (Old Step 9) Final Segmentation/Chunking [TO BE IMPLEMENTED] ===
        self.logger.info("Step 10: Final Segmentation/Chunking... [Not Implemented]")
        # TODO: Implement chunking logic based on transcripts and alignments
        final_chunks = [] # Placeholder

        # === Step 11: (Old Step 10) Audio Normalization & Silence Trim [TO BE IMPLEMENTED] ===
        self.logger.info("Step 11: Normalization and Trimming... [Not Implemented]")
        # TODO: Implement final normalization/trimming on final_chunks
        normalized_chunks = [] # Placeholder

        # === Step 12 (Old Step 11): Save and organize results ===
        # Update this step to potentially include transcription info
        self._save_results(safe_title, verified_paths_list, analysis_results, transcriptions_dir)

        self.logger.info(f"Processing source {safe_title} finished.")
    
    def _extract_audio(self, video_file: Path) -> Optional[Path]:
        """Extract audio from video file."""
        self.logger.info(f"Extracting audio from {video_file.name}")
        
        try:
            audio_path = video_file.with_suffix('.mp3')
            cmd = f'ffmpeg -i "{video_file}" -vn -acodec libmp3lame -q:a 2 "{audio_path}"'
            
            result = os.system(cmd)
            if result == 0:
                self.logger.info(f"Successfully extracted audio to: {audio_path}")
                return audio_path
            else:
                self.logger.error(f"Failed to extract audio from: {video_file}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting audio: {str(e)}")
            return None
    
    def _analyze_audio(self, audio_file: Path) -> Dict[str, Any]:
        """Analyze audio features of the interview."""
        self.logger.info("Analyzing audio features...")
        print("Analyzing audio features...")
        
        try:
            # Use analyze_audio_file from analyze_audio.py
            analysis_results = analyze_audio_file(audio_file)
            
            self.logger.info("Completed audio analysis")
            print("Completed audio analysis")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing audio: {str(e)}")
            print(f"Error analyzing audio: {str(e)}")
            return {}
    
    def _display_audio_stats(self, analysis_results: Dict[str, Any]):
        """Display audio analysis statistics."""
        if not analysis_results or "segments" not in analysis_results:
            print("No analysis results to display")
            return
        
        print("\n----- Audio Analysis Statistics -----")
        print(f"Filename: {analysis_results['filename']}")
        print(f"Duration: {analysis_results['duration']:.2f} seconds")
        print(f"Number of segments: {analysis_results['num_segments']}")
        
        # Calculate music vs speech
        music_segments = [s for s in analysis_results["segments"] if s["likely_music"]]
        speech_segments = [s for s in analysis_results["segments"] if not s["likely_music"]]
        
        music_duration = sum(s["duration"] for s in music_segments)
        speech_duration = sum(s["duration"] for s in speech_segments)
        total_duration = analysis_results["duration"]
        
        music_percent = (music_duration / total_duration) * 100
        speech_percent = (speech_duration / total_duration) * 100
        
        print(f"Music segments: {len(music_segments)} ({music_percent:.1f}% of audio)")
        print(f"Speech segments: {len(speech_segments)} ({speech_percent:.1f}% of audio)")
        
        # Energy stats
        energy_values = [s["rms_energy"] for s in analysis_results["segments"]]
        print(f"Average energy: {np.mean(energy_values):.4f}")
        print(f"Energy variation: {np.std(energy_values):.4f}")
        
        # Spectral stats
        centroid_values = [s["spectral_centroid"] for s in analysis_results["segments"]]
        print(f"Average spectral centroid: {np.mean(centroid_values):.1f} Hz")
        print("-------------------------------------\n")
    
    def _diarize_audio(self, cleaned_audio_path: Path) -> Optional[Any]:
        """Perform speaker diarization on the full cleaned audio file."""
        self.logger.info(f"Step 5: Diarizing speakers in {cleaned_audio_path.name}...")
        diarization_output_file = cleaned_audio_path.with_suffix('.diarization.json')
        try:
            # Use the imported test_diarization function
            # It handles GPU checking, loading model, running, saving JSON
            diarization_result = test_diarization(
                str(cleaned_audio_path),
                output_file=diarization_output_file
                # use_gpu=True # Assuming default is True if available
            )
            if diarization_result is None:
                # Error is already printed by test_diarization
                return None

            # Reset default tensor type after diarization
            torch.set_default_tensor_type('torch.FloatTensor')
            self.logger.info("Reset default tensor type after diarization")

            self.logger.info(f"Diarization successful. Results saved to {diarization_output_file}")
            print(f"Diarization successful. Results saved to {diarization_output_file}")
            return diarization_result # Return the Pyannote Annotation object

        except Exception as e:
            self.logger.error(f"Error during diarization call for {cleaned_audio_path.name}: {e}")
            print(f"Error during diarization call for {cleaned_audio_path.name}: {e}")
            return None
    
    def _save_results(self, title: str, verified_segments_list: List[Path], analysis_results: Dict[str, Any], transcriptions_dir: Optional[Path]):
        """Save processing results and organize segments."""
        self.logger.info(f"Saving results for {title}")
        
        try:
            # Create interview-specific directory
            interview_dir = self.processed_dir / title
            
            # Calculate durations from filenames
            segment_durations = []
            for segment in verified_segments_list:
                try:
                    time_range = segment.stem.split('_')[-1]
                    start_time, end_time = map(float, time_range.split('-'))
                    duration = end_time - start_time
                    segment_durations.append(duration)
                except Exception as e:
                    self.logger.warning(f"Could not parse duration from filename {segment.name}: {e}")
                    segment_durations.append(0.0) # Add 0 if parsing fails
            
            # Save results metadata
            results = {
                'processed_date': datetime.now().isoformat(),
                'total_segments': len(verified_segments_list),
                'segment_durations': segment_durations,
                'total_duration': sum(segment_durations),
                'analysis_results': analysis_results
            }
            
            results_file = interview_dir / "processing_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results saved to {results_file}")
            print(f"Results saved to {results_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            print(f"Error saving results: {str(e)}")

    def _separate_vocals(self, audio_file: Path, output_dir: Path) -> Optional[Path]:
        """Separate vocals from music using Spleeter (Updated)."""
        self.logger.info(f"Separating vocals from music in {audio_file.name} using Spleeter...")
        print(f"Separating vocals from music in {audio_file.name} using Spleeter...")

        try:
            # Call the modified test_spleeter function
            # It handles its own logging/printing
            # It expects the output_dir where it will create its own subdirectory
            vocals_path = spleeter_process_chunked(str(audio_file), output_dir)

            if vocals_path and vocals_path.exists():
                self.logger.info(f"Spleeter separation successful. Vocals file: {vocals_path}")
                print(f"Spleeter separation successful. Vocals file: {vocals_path}")
                return vocals_path
            else:
                self.logger.error(f"Spleeter separation failed or vocals file not found for {audio_file.name}.")
                print(f"Error: Spleeter separation failed or vocals file not found for {audio_file.name}.")
                return None

        except Exception as e:
            # Catch any unexpected error during the call
            self.logger.error(f"Error calling Spleeter for {audio_file.name}: {str(e)}")
            print(f"Error calling Spleeter for {audio_file.name}: {str(e)}")
            return None

    def _extract_speaker_segments_from_diarization(
        self,
        audio_path: Path,
        diarization: Any, # Pyannote Annotation object
        output_dir: Path
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extracts speaker segments based on Pyannote diarization results."""
        self.logger.info(f"Extracting segments for each speaker from {audio_path.name}...")
        output_dir.mkdir(parents=True, exist_ok=True)
        speaker_segments_data = {}

        try:
            # Load the full audio file once
            y, sr = librosa.load(str(audio_path), sr=None)
            self.logger.info(f"Loaded audio with sample rate: {sr}")

            total_segments_extracted = 0
            # Iterate through speaker turns identified by diarization
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_time = turn.start
                end_time = turn.end
                duration = end_time - start_time

                # Skip segments that are too short or too long
                min_segment_duration = 5.0  # Minimum 5 seconds
                max_segment_duration = 30.0  # Maximum 30 seconds
                
                if duration < min_segment_duration:
                    self.logger.debug(f"Skipping short segment: {duration:.2f}s < {min_segment_duration}s")
                    continue
                elif duration > max_segment_duration:
                    self.logger.debug(f"Skipping long segment: {duration:.2f}s > {max_segment_duration}s")
                    continue

                # Convert time to samples
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)

                # Extract segment audio data
                segment_audio = y[start_sample:end_sample]

                # Define output filename
                segment_filename = f"{audio_path.stem}_{speaker}_{start_time:.2f}-{end_time:.2f}.wav"
                segment_output_path = output_dir / segment_filename

                # Save the segment audio
                sf.write(str(segment_output_path), segment_audio, sr)
                total_segments_extracted += 1

                # Store segment information
                segment_info = {
                    "path": str(segment_output_path),
                    "start": start_time,
                    "end": end_time,
                    "duration": duration
                }
                if speaker not in speaker_segments_data:
                    speaker_segments_data[speaker] = []
                speaker_segments_data[speaker].append(segment_info)

                # Log progress occasionally
                if total_segments_extracted % 10 == 0:  # Reduced from 50 since we'll have fewer segments
                    self.logger.info(f"Extracted {total_segments_extracted} segments...")

            self.logger.info(f"Finished extracting {total_segments_extracted} segments total.")
            # Log per-speaker counts
            for speaker, segments in speaker_segments_data.items():
                total_duration = sum(s['duration'] for s in segments)
                self.logger.info(f"  Speaker {speaker}: {len(segments)} segments, Total Duration: {total_duration:.2f}s")
                print(f"  Speaker {speaker}: {len(segments)} segments, Total Duration: {total_duration:.2f}s")

            return speaker_segments_data

        except Exception as e:
            self.logger.error(f"Error extracting speaker segments: {e}")
            print(f"Error extracting speaker segments: {e}")
            return {} # Return empty dict on error

    def _verify_speaker(self, speaker_segments_data: Dict[str, List[Dict[str, Any]]], output_dir: Path) -> List[Path]:
        """Verify speaker identity using reference samples."""
        self.logger.info("Verifying speaker identity...")

        try:
            # Define paths
            reference_dir = Path("data/olamide/reference_samples")
            test_dir = output_dir.parent / "diarized_segments"
            results_output_file = output_dir / "speaker_verification_results.json"

            # Initialize the speaker recognition model
            self.logger.info("Loading speaker recognition model...")
            model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
            model.eval()
            device = next(model.parameters()).device
            self.logger.info(f"Model loaded onto device: {device}")

            # Call the imported verify_speaker_segments function with the model
            results = verify_speaker_segments(
                reference_dir=reference_dir,
                test_dir=test_dir,
                output_file=results_output_file,
                threshold=0.75,
                model=model  # Pass the model we just loaded
            )

            # Process results to get verified segments
            verified_dir = output_dir / "verified_segments"
            verified_dir.mkdir(exist_ok=True)
            verified_paths = []

            for segment_filename, result_data in results.items():
                if result_data['is_same_speaker']:
                    src_path = test_dir / segment_filename
                    dst_path = verified_dir / segment_filename
                    # Ensure source exists before copying
                    if src_path.exists():
                        shutil.copy2(src_path, dst_path)
                        verified_paths.append(dst_path)
                    else:
                        self.logger.warning(f"Source segment {src_path} not found during copy.")

            if not verified_paths:
                self.logger.warning(f"No segments were verified and copied. Check results: {results_output_file}")
            else:
                self.logger.info(f"Copied {len(verified_paths)} verified segments to {verified_dir}")
            
            return verified_paths

        except Exception as e:
            self.logger.error(f"Error during speaker verification: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single interview source")
    parser.add_argument("--artist", required=True, help="Name of the artist")
    parser.add_argument("--index", type=int, help="Index of the source to process")
    parser.add_argument("--url", help="URL of the source to process")
    args = parser.parse_args()

    agent = InterviewAgent(artist_name=args.artist)
    
    if args.index is not None:
        agent.process_single_source(source_index=args.index)
    elif args.url is not None:
        agent.process_single_source(source_url=args.url)
    else:
        print("Please provide either --index or --url") 