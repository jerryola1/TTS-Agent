import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import librosa
import soundfile as sf
import numpy as np
from praatio import textgrid # Import praatio's textgrid module

# Setup logging
logging.basicConfig(level=logging.DEBUG, # Use DEBUG level for testing
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
MIN_CHUNK_DURATION_SEC = 2.0
MAX_CHUNK_DURATION_SEC = 15.0
MAX_SILENCE_BETWEEN_WORDS_SEC = 0.7 # Max gap within a chunk
OUTPUT_SAMPLE_RATE = 22050 # Common sample rate for TTS, adjust if needed

def find_transcript_file(audio_file: Path, transcript_dir: Path) -> Optional[Path]:
    """Find the corresponding transcript file for a given audio file."""
    # Assuming transcript file has the same stem but with .json extension
    transcript_filename = audio_file.stem + ".json" 
    potential_path = transcript_dir / transcript_filename
    if potential_path.exists():
        return potential_path
    
    # Fallback: check for .txt if .json is not found (based on whisper output format)
    transcript_filename_txt = audio_file.stem + ".txt"
    potential_path_txt = transcript_dir / transcript_filename_txt
    if potential_path_txt.exists():
        return potential_path_txt

    logger.warning(f"Transcript file not found for {audio_file.name} in {transcript_dir}")
    return None

def chunk_audio_on_silence(
    audio_path: Path,
    output_dir: Path,
    transcript_path: Optional[Path],
    min_segment_len_sec: float = 2.0,
    max_segment_len_sec: float = 15.0,
    top_db: int = 40  # Threshold in dB below reference to consider silence
) -> List[Dict[str, Any]]:
    """
    Chunks an audio file based on silence detection using librosa.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory to save the output audio chunks.
        transcript_path: Path to the corresponding transcript file (optional, for metadata).
        min_segment_len_sec: Minimum duration for a chunk.
        max_segment_len_sec: Maximum duration for a chunk.
        top_db: The threshold (in decibels) below the reference loudness (max)
                to classify a frame as silent.

    Returns:
        A list of dictionaries, each containing info about a saved chunk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_metadata = []

    try:
        # Load audio
        y, sr = librosa.load(str(audio_path), sr=None)
        logger.info(f"Loaded {audio_path.name}, duration: {librosa.get_duration(y=y, sr=sr):.2f}s, sr: {sr}")

        # Find non-silent intervals
        # hop_length common for STFT, adjust if needed
        intervals = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512) 

        transcript_text = ""
        if transcript_path:
            try:
                # Handle both JSON (Whisper output format) and plain TXT
                if transcript_path.suffix == ".json":
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        # Assuming standard Whisper JSON format { 'text': '...', 'segments': [...] }
                        data = json.load(f)
                        transcript_text = data.get('text', '')
                elif transcript_path.suffix == ".txt":
                     with open(transcript_path, 'r', encoding='utf-8') as f:
                        transcript_text = f.read().strip()
                else:
                    logger.warning(f"Unsupported transcript file format: {transcript_path.suffix}")

            except Exception as e:
                logger.error(f"Error reading transcript {transcript_path}: {e}")


        chunk_index = 0
        for start_i, end_i in intervals:
            start_sec = librosa.samples_to_time(start_i, sr=sr)
            end_sec = librosa.samples_to_time(end_i, sr=sr)
            duration = end_sec - start_sec

            # Filter by duration
            if duration < min_segment_len_sec or duration > max_segment_len_sec:
                # logger.debug(f"Skipping segment {chunk_index}: duration {duration:.2f}s out of range [{min_segment_len_sec}, {max_segment_len_sec}]")
                continue

            # Extract audio chunk
            chunk_audio = y[start_i:end_i]

            # Define output filename
            # Include original stem, chunk index, and timestamps for traceability
            chunk_filename = f"{audio_path.stem}_chunk{chunk_index:03d}_{start_sec:.2f}-{end_sec:.2f}.wav"
            chunk_output_path = output_dir / chunk_filename

            # Save the chunk
            sf.write(str(chunk_output_path), chunk_audio, sr)

            # Store metadata
            # NOTE: The transcript here is for the *original* long file.
            # A more advanced step would involve aligning and splitting the transcript.
            meta = {
                "original_audio": str(audio_path),
                "chunk_audio_path": str(chunk_output_path),
                "start_time_seconds": start_sec,
                "end_time_seconds": end_sec,
                "duration_seconds": duration,
                "original_transcript": transcript_text # Include full transcript for reference
                # Future enhancement: Add 'chunk_transcript' if alignment is done
            }
            chunk_metadata.append(meta)
            chunk_index += 1

        logger.info(f"Saved {chunk_index} chunks for {audio_path.name} to {output_dir}")
        return chunk_metadata

    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}", exc_info=True)
        return []

def process_directory(
    input_audio_dir: Path,
    input_transcript_dir: Path,
    output_chunk_dir: Path,
    min_len: float,
    max_len: float,
    db_threshold: int
):
    """Processes all WAV files in the input directory."""
    all_metadata = []
    processed_files = 0
    failed_files = 0

    for audio_file in input_audio_dir.glob("*.wav"):
        logger.info(f"--- Processing file: {audio_file.name} ---")
        transcript_file = find_transcript_file(audio_file, input_transcript_dir)
        if not transcript_file:
            logger.warning(f"Skipping {audio_file.name} due to missing transcript.")
            failed_files += 1
            continue
            
        try:
            file_metadata = chunk_audio_on_silence(
                audio_path=audio_file,
                output_dir=output_chunk_dir,
                transcript_path=transcript_file,
                min_segment_len_sec=min_len,
                max_segment_len_sec=max_len,
                top_db=db_threshold
            )
            all_metadata.extend(file_metadata)
            processed_files += 1
        except Exception as e:
            logger.error(f"Failed processing {audio_file.name}: {e}")
            failed_files += 1


    # Save combined metadata
    metadata_file = output_chunk_dir / "_chunk_metadata.jsonl"
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for item in all_metadata:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Saved combined metadata for {processed_files} processed files to {metadata_file}")
    except Exception as e:
         logger.error(f"Failed to write metadata file {metadata_file}: {e}")

    logger.info(f"--- Batch Processing Summary ---")
    logger.info(f"Successfully processed: {processed_files} files")
    logger.info(f"Failed or skipped:    {failed_files} files")
    logger.info(f"Total chunks created: {len(all_metadata)}")
    logger.info(f"Chunks saved in:      {output_chunk_dir}")
    logger.info(f"Metadata saved to:    {metadata_file}")
    logger.info(f"---------------------------------")

def parse_textgrid_for_words(tg_path: Path) -> Optional[List[Tuple[float, float, str]]]:
    """Parses a TextGrid file to extract word intervals using correct praatio attributes."""
    if not tg_path.exists():
        logger.error(f"TextGrid file not found: {tg_path}")
        return None
    try:
        tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False)
        
        word_tier = None
        target_tier_name = None
        for name in tg.tierNames: 
            if name.lower() == 'words':
                target_tier_name = name
                break
        
        if target_tier_name:
            word_tier = tg.getTier(target_tier_name)
        else:
            logger.error(f"Could not find 'words' tier in TextGrid: {tg_path} (Available tiers: {tg.tierNames})")
            return None
            
        word_intervals = [
            (entry.start, entry.end, entry.label) 
            for entry in word_tier.entries
        ]
        
        silence_markers = ["sil", "sp", "spn", "<eps>", "", "#"]
        filtered_intervals = [
            (float(start), float(end), label) for start, end, label in word_intervals
            if label not in silence_markers
        ]
        
        if not filtered_intervals:
             logger.warning(f"No valid word intervals found after filtering silence in {tg_path.name}")
             return None
        else:
             logger.debug(f"Successfully parsed {len(filtered_intervals)} word intervals from {tg_path.name}")
             return filtered_intervals
             
    except Exception as e:
        logger.error(f"Error parsing TextGrid {tg_path}: {e}", exc_info=True)
        return None

def create_tts_chunks(
    audio_path: Path,
    word_intervals: List[Tuple[float, float, str]],
    output_dir: Path,
    base_filename: str
):
    """
    Creates final TTS audio chunks and transcript files based on word intervals.
    """
    # <<< ADD DEBUG LOGGING AT FUNCTION ENTRY >>>
    logger.debug(f"Entering create_tts_chunks for {audio_path.name} with {len(word_intervals)} word intervals.")
    
    final_chunks_metadata = []
    if not word_intervals:
        logger.warning(f"create_tts_chunks called with no word intervals for {audio_path.name}")
        return final_chunks_metadata

    try:
        # Load audio - resample here if needed
        y, sr_orig = librosa.load(str(audio_path), sr=None)
        if sr_orig != OUTPUT_SAMPLE_RATE:
             logger.info(f"Resampling {audio_path.name} from {sr_orig}Hz to {OUTPUT_SAMPLE_RATE}Hz")
             y = librosa.resample(y, orig_sr=sr_orig, target_sr=OUTPUT_SAMPLE_RATE)
             sr = OUTPUT_SAMPLE_RATE
        else:
             sr = sr_orig
        logger.debug(f"Loaded {audio_path.name}, duration: {len(y)/sr:.2f}s, sr: {sr}Hz")

    except Exception as e:
        logger.error(f"Failed to load audio file {audio_path}: {e}")
        return final_chunks_metadata

    current_chunk_words = []
    chunk_start_time = None
    last_word_end_time = None
    chunk_counter = 0

    # Helper to finalize and save chunk
    def finalize_and_save_chunk(words_in_chunk):
        nonlocal chunk_counter, final_chunks_metadata
        if not words_in_chunk:
            return

        start_t = words_in_chunk[0][0] # Start time of first word
        end_t = words_in_chunk[-1][1]  # End time of last word
        duration = end_t - start_t
        word_count = len(words_in_chunk) # Get word count for debugging

        logger.debug(f"Proposed chunk for {base_filename}: Start={start_t:.2f}s, End={end_t:.2f}s, Duration={duration:.2f}s, Words={word_count}")

        if duration >= MIN_CHUNK_DURATION_SEC and duration <= MAX_CHUNK_DURATION_SEC:
            start_sample = int(start_t * sr)
            end_sample = int(end_t * sr)
            chunk_audio = y[start_sample:end_sample]
            chunk_transcript = " ".join(label for _, _, label in words_in_chunk)
            chunk_wav_filename = f"{base_filename}_tts_{chunk_counter:04d}.wav"
            chunk_txt_filename = f"{base_filename}_tts_{chunk_counter:04d}.txt"
            chunk_wav_path = output_dir / chunk_wav_filename
            chunk_txt_path = output_dir / chunk_txt_filename
            try:
                sf.write(str(chunk_wav_path), chunk_audio, sr)
                with open(chunk_txt_path, 'w', encoding='utf-8') as f:
                    f.write(chunk_transcript)
                final_chunks_metadata.append({
                    "audio_filepath": str(chunk_wav_path), 
                    "text": chunk_transcript, "duration": duration,
                    "original_audio": str(audio_path), "original_start": start_t, "original_end": end_t,
                })
                logger.info(f"Saved chunk {chunk_counter:04d}: {chunk_wav_path.name} ({duration:.2f}s)")
                chunk_counter += 1
            except Exception as e:
                 logger.error(f"Failed to save chunk {chunk_counter} for {base_filename}: {e}")
        else:
             logger.debug(f"Skipping proposed chunk (duration {duration:.2f}s) outside range [{MIN_CHUNK_DURATION_SEC}, {MAX_CHUNK_DURATION_SEC}]")


    # Iterate through word intervals from TextGrid
    # <<< ADD DEBUG LOGGING BEFORE LOOP >>>
    logger.debug(f"Starting word interval loop for {base_filename}...")
    for i, (start_sec, end_sec, label) in enumerate(word_intervals):
        # <<< ADD DEBUG LOGGING INSIDE LOOP >>>
        logger.debug(f"  Processing word {i}: '{label}' ({start_sec:.2f}-{end_sec:.2f})")

        if not current_chunk_words:
            current_chunk_words.append((start_sec, end_sec, label))
            chunk_start_time = start_sec
            last_word_end_time = end_sec
            continue

        potential_duration = end_sec - chunk_start_time
        gap_since_last = start_sec - last_word_end_time

        finalize_current = False
        if potential_duration > MAX_CHUNK_DURATION_SEC:
            finalize_current = True
            logger.debug(f"    Finalizing chunk due to max duration ({potential_duration:.2f} > {MAX_CHUNK_DURATION_SEC})")
        elif gap_since_last > MAX_SILENCE_BETWEEN_WORDS_SEC:
            finalize_current = True
            logger.debug(f"    Finalizing chunk due to word gap ({gap_since_last:.2f} > {MAX_SILENCE_BETWEEN_WORDS_SEC})")

        if finalize_current:
            finalize_and_save_chunk(current_chunk_words)
            current_chunk_words = [(start_sec, end_sec, label)]
            chunk_start_time = start_sec
            last_word_end_time = end_sec
        else:
            current_chunk_words.append((start_sec, end_sec, label))
            last_word_end_time = end_sec

    # <<< ADD DEBUG LOGGING BEFORE FINAL CALL >>>
    logger.debug(f"Finalizing last chunk for {base_filename} after loop...")
    finalize_and_save_chunk(current_chunk_words)

    return final_chunks_metadata

def process_segmentation_for_directory(
    input_audio_dir: Path,
    input_textgrid_dir: Path,
    output_tts_dir: Path,
):
    """
    Processes a directory of audio files and aligned TextGrids to create TTS chunks.

    Args:
        input_audio_dir: Directory containing verified audio segments (.wav).
        input_textgrid_dir: Directory containing corresponding .TextGrid alignment files.
        output_tts_dir: Directory to save the final TTS audio (.wav) and text (.txt) pairs.

    Returns:
        Path to the metadata file summarizing the created chunks, or None if failed.
    """
    output_tts_dir.mkdir(parents=True, exist_ok=True)
    metadata_filepath = output_tts_dir / "_tts_metadata.jsonl"
    all_run_metadata = []
    processed_files = 0
    skipped_files = 0
    total_chunks_created = 0

    logger.info(f"Starting TTS chunk creation...")
    logger.info(f"Input Audio: {input_audio_dir}")
    logger.info(f"Input TextGrids: {input_textgrid_dir}")
    logger.info(f"Output TTS Data: {output_tts_dir}")

    # <<< ADD DEBUG LOGGING: Check if audio files are found >>>
    audio_files_list = list(input_audio_dir.glob("*.wav"))
    logger.debug(f"Found {len(audio_files_list)} '.wav' files in {input_audio_dir}")
    if not audio_files_list:
        logger.error(f"No WAV files found in input directory: {input_audio_dir}")
        return None # Exit if no audio files

    # Iterate through audio files in the input directory
    for audio_file in audio_files_list:
        # <<< ADD DEBUG LOGGING: Confirm loop entry >>>
        logger.debug(f"--- Processing base audio: {audio_file.name} ---")

        # Find corresponding TextGrid file
        textgrid_file = input_textgrid_dir / (audio_file.stem + ".TextGrid")
        textgrid_found_path = None # Variable to store the found path
        if textgrid_file.exists():
             # <<< ADD DEBUG LOGGING: TextGrid found (original case) >>>
             logger.debug(f"Found matching TextGrid: {textgrid_file.name}")
             textgrid_found_path = textgrid_file
        else:
             # Try lowercase extension as fallback
             textgrid_file_lower = input_textgrid_dir / (audio_file.stem + ".textgrid")
             if textgrid_file_lower.exists():
                  # <<< ADD DEBUG LOGGING: TextGrid found (lowercase case) >>>
                  logger.debug(f"Found matching TextGrid (lowercase): {textgrid_file_lower.name}")
                  textgrid_found_path = textgrid_file_lower
             else:
                  logger.warning(f"TextGrid file not found for {audio_file.name} (checked .TextGrid and .textgrid). Skipping.")
                  skipped_files += 1
                  continue # Skip to next audio file

        # Parse word intervals from TextGrid
        word_intervals = parse_textgrid_for_words(textgrid_found_path) # Use the found path

        # <<< ADD DEBUG LOGGING: Check result of parsing >>>
        if word_intervals:
            logger.debug(f"Successfully obtained {len(word_intervals)} word intervals for {audio_file.name}.")
        else:
            # The parse_textgrid_for_words function already logs errors/warnings
            logger.warning(f"Parsing failed or yielded no words for {textgrid_found_path.name}. Skipping chunk creation for {audio_file.name}.")
            skipped_files += 1
            continue # Skip to next audio file

        # Create chunks for this audio file (only if parsing succeeded)
        try:
            chunk_metadata = create_tts_chunks(
                audio_path=audio_file,
                word_intervals=word_intervals,
                output_dir=output_tts_dir,
                base_filename=audio_file.stem
            )
            if chunk_metadata:
                all_run_metadata.extend(chunk_metadata)
                total_chunks_created += len(chunk_metadata)
                processed_files += 1
            else:
                 logger.warning(f"No valid TTS chunks generated during create_tts_chunks for {audio_file.name}.")
                 skipped_files += 1

        except Exception as e:
             logger.error(f"Failed during chunk creation for {audio_file.name}: {e}", exc_info=True)
             skipped_files += 1


    # Save summary metadata (optional)
    if all_run_metadata:
        try:
            with open(metadata_filepath, 'w', encoding='utf-8') as f:
                for item in all_run_metadata:
                    f.write(json.dumps(item) + '\n') # Use json dumps for each line
            logger.info(f"Saved summary metadata for {total_chunks_created} chunks to {metadata_filepath}")
        except Exception as e:
            logger.error(f"Failed to write summary metadata file {metadata_filepath}: {e}")
    else:
        logger.warning("No TTS chunks were created in this run.")
        # Don't return None here necessarily, summary might still be useful
        # return None # Indicate no chunks created

    logger.info(f"--- TTS Chunk Creation Summary ---")
    logger.info(f"Successfully processed source files: {processed_files}")
    logger.info(f"Skipped/Failed source files:      {skipped_files}")
    logger.info(f"Total TTS chunks created:         {total_chunks_created}")
    logger.info(f"TTS data saved in:                {output_tts_dir}")
    logger.info(f"----------------------------------")

    return metadata_filepath if all_run_metadata else None # Return path only if chunks were made

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment audio into TTS chunks using TextGrid alignments.")
    parser.add_argument(
        "--input_audio_dir", type=str, required=True,
        help="Directory containing verified audio segments (.wav files, output of Step 8)."
    )
    parser.add_argument(
        "--input_textgrid_dir", type=str, required=True,
        help="Directory containing corresponding TextGrid alignment files (.TextGrid, output of Step 9.5)."
    )
    parser.add_argument(
        "--output_tts_dir", type=str, required=True,
        help="Directory to save the final TTS audio chunks (.wav) and transcripts (.txt)."
    )
    # Add options for chunk duration, silence, sample rate if needed
    # parser.add_argument("--min_len", type=float, default=MIN_CHUNK_DURATION_SEC)
    # parser.add_argument("--max_len", type=float, default=MAX_CHUNK_DURATION_SEC)
    # parser.add_argument("--max_gap", type=float, default=MAX_SILENCE_BETWEEN_WORDS_SEC)
    # parser.add_argument("--sr", type=int, default=OUTPUT_SAMPLE_RATE)

    args = parser.parse_args()

    # Use constants defined at the top for chunking parameters for now
    process_segmentation_for_directory(
        input_audio_dir=Path(args.input_audio_dir),
        input_textgrid_dir=Path(args.input_textgrid_dir),
        output_tts_dir=Path(args.output_tts_dir),
    )

    logger.info("Final segmentation script finished.")
