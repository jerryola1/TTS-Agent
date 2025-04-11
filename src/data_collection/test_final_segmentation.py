import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import librosa
import soundfile as sf
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk verified audio segments based on silence.")
    parser.add_argument(
        "--input_audio_dir", 
        type=str, 
        required=True, 
        help="Directory containing verified audio segments (.wav files from Step 9)."
    )
    parser.add_argument(
        "--input_transcript_dir", 
        type=str, 
        required=True, 
        help="Directory containing corresponding transcripts (.json or .txt files from Step 9)."
    )
    parser.add_argument(
        "--output_chunk_dir", 
        type=str, 
        required=True, 
        help="Directory to save the final audio chunks."
    )
    parser.add_argument(
        "--min_len", 
        type=float, 
        default=2.0, 
        help="Minimum chunk duration in seconds."
    )
    parser.add_argument(
        "--max_len", 
        type=float, 
        default=15.0, 
        help="Maximum chunk duration in seconds."
    )
    parser.add_argument(
        "--db_threshold", 
        type=int, 
        default=40, 
        help="Silence threshold in dB below the maximum."
    )

    args = parser.parse_args()

    input_audio_path = Path(args.input_audio_dir)
    input_transcript_path = Path(args.input_transcript_dir)
    output_chunk_path = Path(args.output_chunk_dir)

    if not input_audio_path.is_dir():
        logger.error(f"Input audio directory not found: {input_audio_path}")
        exit(1)
    if not input_transcript_path.is_dir():
        logger.error(f"Input transcript directory not found: {input_transcript_path}")
        exit(1)

    process_directory(
        input_audio_dir=input_audio_path,
        input_transcript_dir=input_transcript_path,
        output_chunk_dir=output_chunk_path,
        min_len=args.min_len,
        max_len=args.max_len,
        db_threshold=args.db_threshold
    )

    logger.info("Final segmentation script finished.")
