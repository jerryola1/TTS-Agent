import argparse
import csv
import logging
import os
from pathlib import Path
import soundfile as sf
from typing import Dict, Tuple, Optional

# Optional: Try importing a tokenizer for more accurate token counts
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("Warning: 'transformers' library not found. Token count will be based on space splitting.")
    print("Install it for potentially more accurate counts: pip install transformers")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_audio_duration(file_path: Path) -> Optional[float]:
    """Gets the duration of an audio file using soundfile."""
    try:
        info = sf.info(str(file_path))
        return info.duration
    except Exception as e:
        logger.warning(f"Could not read duration for {file_path.name}: {e}")
        return None

def analyze_dataset(
    metadata_path: Path,
    wav_base_dir: Path,
    tokenizer: Optional[callable] = None
) -> Optional[Dict[str, float | int]]:
    """Analyzes a dataset defined by a metadata CSV file."""
    if not metadata_path.is_file():
        logger.error(f"Metadata file not found: {metadata_path}")
        return None

    total_samples = 0
    total_duration = 0.0
    total_tokens = 0
    skipped_files = 0

    logger.info(f"Analyzing dataset: {metadata_path.name}...")

    try:
        with open(metadata_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='|')
            header = next(reader) # Skip header row
            if len(header) < 2:
                 logger.error(f"Invalid header format in {metadata_path.name}. Expected at least 2 columns separated by '|'. Header: {header}")
                 return None

            for i, row in enumerate(reader):
                if len(row) < 2:
                    logger.warning(f"Skipping malformed row {i+2} in {metadata_path.name}: {row}")
                    skipped_files += 1
                    continue

                relative_wav_path_str = row[0]
                text = row[1]

                # Construct the full path to the wav file
                # Assumes the path in the CSV is relative to the wav_base_dir
                full_wav_path = wav_base_dir / relative_wav_path_str

                if not full_wav_path.is_file():
                    logger.warning(f"Audio file not found for row {i+2}: {full_wav_path}. Skipping.")
                    skipped_files += 1
                    continue

                duration = get_audio_duration(full_wav_path)
                if duration is None:
                    skipped_files += 1
                    continue # Skip if duration couldn't be read
                
                total_duration += duration
                total_samples += 1

                # Count tokens
                if tokenizer:
                    tokens = tokenizer.tokenize(text)
                    total_tokens += len(tokens)
                else:
                    # Simple space splitting as fallback
                    total_tokens += len(text.split())

    except FileNotFoundError:
        logger.error(f"Metadata file not found during analysis: {metadata_path}")
        return None
    except StopIteration: # Handles empty file after header
         logger.warning(f"Metadata file {metadata_path.name} appears to be empty or only contains a header.")
         # Allow returning zero counts
         pass 
    except Exception as e:
        logger.error(f"Error reading or processing {metadata_path.name}: {e}", exc_info=True)
        return None

    avg_duration = total_duration / total_samples if total_samples > 0 else 0
    avg_tokens = total_tokens / total_samples if total_samples > 0 else 0

    return {
        "total_samples": total_samples,
        "skipped_files": skipped_files,
        "total_duration_s": total_duration,
        "total_tokens": total_tokens,
        "avg_duration_s": avg_duration,
        "avg_tokens_per_sample": avg_tokens
    }

def format_duration(seconds: float) -> str:
    """Formats duration in seconds to HH:MM:SS.ms or MM:SS.ms format."""
    if seconds < 0: return "Invalid duration"
    
    total_seconds = int(seconds)
    milliseconds = int((seconds - total_seconds) * 1000)
    
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}.{milliseconds:03d}"
    else:
        return f"{minutes:02d}:{sec:02d}.{milliseconds:03d}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze TTS dataset (metadata CSV and WAV files).")
    parser.add_argument(
        "--artist_name", type=str, required=True,
        help="Name of the artist (used to determine the data directory, e.g., 'olamide')."
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default=None,
        help="Optional: Hugging Face tokenizer name (e.g., 'gpt2') for more accurate token counts. If not provided, uses space splitting."
    )

    args = parser.parse_args()

    # Derive paths based on artist name and standard structure
    artist_name = args.artist_name
    base_data_dir = Path(f"alltalk_tts/finetune/{artist_name}")
    metadata_dir = base_data_dir
    wav_base_dir = base_data_dir # The directory containing the 'wavs' subfolder

    train_metadata_path = metadata_dir / "metadata_train.csv"
    eval_metadata_path = metadata_dir / "metadata_eval.csv"

    # Validate base directory existence
    if not base_data_dir.is_dir():
        logger.error(f"Base data directory not found: {base_data_dir}")
        logger.error("Please ensure the directory 'alltalk_tts/finetune/{artist_name}' exists and contains the data.")
        exit(1)

    # Load tokenizer if specified and available
    tokenizer = None
    if args.tokenizer_name and TOKENIZER_AVAILABLE:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
            logger.info(f"Using tokenizer: {args.tokenizer_name}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer '{args.tokenizer_name}': {e}. Falling back to space splitting.")
            tokenizer = None
    elif args.tokenizer_name and not TOKENIZER_AVAILABLE:
        logger.warning(f"Tokenizer '{args.tokenizer_name}' requested, but 'transformers' library not found. Falling back to space splitting.")

    # Analyze datasets
    train_stats = analyze_dataset(train_metadata_path, wav_base_dir, tokenizer)
    eval_stats = analyze_dataset(eval_metadata_path, wav_base_dir, tokenizer)

    # Print Summary Report
    print("\n--- TTS Dataset Analysis Report ---")
    print(f"Metadata Directory: {metadata_dir}")
    print(f"WAV Directory: {wav_base_dir}")
    print(f"Tokenizer Used: {args.tokenizer_name if tokenizer else 'Space Splitting'}")
    print("-------------------------------------")

    if train_stats:
        print("Training Set Statistics:")
        print(f"  Total Samples:       {train_stats['total_samples']}")
        print(f"  Skipped Files:       {train_stats['skipped_files']}")
        print(f"  Total Duration:      {format_duration(train_stats['total_duration_s'])} ({train_stats['total_duration_s']:.2f} seconds)")
        print(f"  Total Tokens:        {train_stats['total_tokens']}")
        print(f"  Avg Duration/Sample: {train_stats['avg_duration_s']:.2f} seconds")
        print(f"  Avg Tokens/Sample:   {train_stats['avg_tokens_per_sample']:.2f}")
        print("-------------------------------------")
    else:
        print("Training Set Analysis FAILED.")
        print("-------------------------------------")

    if eval_stats:
        print("Evaluation Set Statistics:")
        print(f"  Total Samples:       {eval_stats['total_samples']}")
        print(f"  Skipped Files:       {eval_stats['skipped_files']}")
        print(f"  Total Duration:      {format_duration(eval_stats['total_duration_s'])} ({eval_stats['total_duration_s']:.2f} seconds)")
        print(f"  Total Tokens:        {eval_stats['total_tokens']}")
        print(f"  Avg Duration/Sample: {eval_stats['avg_duration_s']:.2f} seconds")
        print(f"  Avg Tokens/Sample:   {eval_stats['avg_tokens_per_sample']:.2f}")
        print("-------------------------------------")
    else:
        print("Evaluation Set Analysis FAILED.")
        print("-------------------------------------") 