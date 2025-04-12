# Initial placeholder for TTS data preparation script

import argparse
import json
import logging
import os
import re
import random # <<< Add random for shuffling
from pathlib import Path
from typing import List, Tuple, Dict

# Placeholder for a potential text normalization library
# You might need to install one, e.g., pip install inflect
try:
    import inflect
    p = inflect.engine()
    INFLECT_AVAILABLE = True
except ImportError:
    p = None
    INFLECT_AVAILABLE = False
    print("Warning: 'inflect' library not found. Number-to-words conversion will be basic.")
    print("Consider installing it: pip install inflect")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Text Normalization Functions ---

def normalize_numbers(text: str) -> str:
    """Converts numbers in the text to words using inflect if available."""
    # Remove commas from numbers like 1,000 -> 1000 before conversion
    text = re.sub(r'(\d),(\d)', r'\\1\\2', text)

    if INFLECT_AVAILABLE:
        # Use inflect for robust number-to-words
        def replace_match(match):
            try:
                # Handle potential edge cases or large numbers if needed
                num_str = match.group(0)
                # Prevent inflect from processing very large numbers that might cause errors/long delays
                if len(num_str.split('.')[0]) > 12: # Arbitrary limit (e.g., trillions)
                    logger.warning(f"Skipping number-to-words for large number: {num_str}")
                    return num_str
                return p.number_to_words(num_str)
            except Exception as e:
                logger.warning(f"Inflect failed for '{match.group(0)}': {e}. Keeping original.")
                return match.group(0) # Keep original if inflect fails

        # Find numbers (integers and decimals)
        text = re.sub(r'\b\d+(\.\d+)?\b', replace_match, text)
    else:
        # If inflect is not available, leave numbers as digits.
        pass # Numbers remain as digits

    return text

def normalize_text(text: str) -> str:
    """Applies basic text normalization steps suitable for TTS training data."""
    text = text.lower() # Convert to lowercase

    # Add specific replacements for abbreviations/symbols if needed *before* number conversion
    # Example:
    # text = text.replace("mr.", "mister")
    # text = text.replace("mrs.", "missus")
    # text = text.replace("dr.", "doctor")
    # text = text.replace("st.", "saint") # or "street"
    # text = text.replace("&", " and ")
    # text = text.replace("%": " percent ")
    # text = text.replace("$", " dollars ") # If inflect isn't handling currency

    text = normalize_numbers(text)

    # Remove remaining unwanted characters. Keep letters, digits, spaces, and basic punctuation.
    # Adjust the allowed characters based on the target TTS model requirements.
    # Common practice is to keep .,?!'-
    text = re.sub(r"[^a-z0-9\s.,?!'-]", '', text)

    # Optionally pad punctuation with spaces, required by some models/tokenizers
    # text = re.sub(r'([.,?!'-])', r' \\1 ', text)

    # Collapse multiple spaces to single spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# --- Manifest File Creation Function ---

def create_manifest(data: List[Tuple[str, str]], output_path: Path, delimiter: str = '|') -> None:
    """Creates the manifest file (e.g., metadata.csv) in LJSpeech format."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
        with open(output_path, 'w', encoding='utf-8') as f:
            count = 0
            for wav_path_relative, transcript in data:
                # Ensure paths use forward slashes for consistency across OS
                clean_path = str(wav_path_relative).replace(os.sep, '/')
                f.write(f"{clean_path}{delimiter}{transcript}\n")
                count += 1
        logger.info(f"Successfully wrote {count} entries to manifest file: {output_path}")
    except Exception as e:
        logger.error(f"Failed to write manifest file {output_path}: {e}", exc_info=True)

# --- Directory Processing Functions ---

def find_chunk_directories(input_dir: Path, recursive: bool) -> List[Path]:
    """Finds directories named 'final_tts_chunks' within the input directory."""
    directories_to_process = []
    target_dir_name = "final_tts_chunks"

    if recursive:
        logger.info(f"Recursively searching for '{target_dir_name}' directories under {input_dir}")
        for item in input_dir.rglob(target_dir_name):
             if item.is_dir():
                  directories_to_process.append(item)
    elif input_dir.is_dir() and input_dir.name == target_dir_name:
         logger.info(f"Processing single specified directory: {input_dir}")
         directories_to_process.append(input_dir)
    else:
         # If not recursive and not pointing directly to the target dir, search one level down
         logger.info(f"Checking for '{target_dir_name}' in subdirectories of {input_dir}")
         for item in input_dir.iterdir():
             if item.is_dir() and item.name == target_dir_name:
                 directories_to_process.append(item)
         if not directories_to_process:
             logger.error(f"Input directory '{input_dir}' must contain or be named '{target_dir_name}', or use --recursive.")

    if not directories_to_process:
         logger.warning(f"No '{target_dir_name}' directories found to process in {input_dir}")

    return directories_to_process

def process_chunk_directory(chunk_dir: Path, base_input_dir: Path) -> List[Tuple[Path, str]]:
    """Processes a single directory containing audio chunks and transcripts.
       Returns tuples of (Path object relative to base_input_dir, normalized_transcript).
    """
    prepared_data = []
    logger.info(f"Processing directory: {chunk_dir}")

    wav_files = sorted(list(chunk_dir.glob("*.wav")))
    if not wav_files:
        logger.warning(f"No .wav files found in {chunk_dir}. Skipping.")
        return []

    processed_count = 0
    skipped_count = 0
    for wav_file in wav_files:
        txt_file = chunk_dir / (wav_file.stem + ".txt")
        if not txt_file.exists():
            logger.warning(f"Transcript file not found for {wav_file.name}. Skipping.")
            skipped_count += 1
            continue

        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                original_transcript = f.read().strip()

            if not original_transcript:
                 logger.warning(f"Transcript file {txt_file.name} is empty. Skipping.")
                 skipped_count += 1
                 continue

            normalized_transcript = normalize_text(original_transcript)

            if not normalized_transcript:
                logger.warning(f"Transcript for {wav_file.name} became empty after normalization. Original: '{original_transcript}'. Skipping.")
                skipped_count += 1
                continue

            # Calculate relative path from the *base* input directory
            try:
                # Make path relative to the directory containing the 'final_tts_chunks' dir(s)
                wav_relative_path = wav_file.relative_to(base_input_dir)
                prepared_data.append((wav_relative_path, normalized_transcript))
                processed_count += 1
            except ValueError as e:
                logger.error(f"Could not make {wav_file} relative to base directory {base_input_dir}. Error: {e}. Skipping.")
                skipped_count += 1
                continue

        except Exception as e:
            logger.error(f"Error processing file pair {wav_file.name} / {txt_file.name}: {e}", exc_info=True)
            skipped_count += 1

    logger.info(f"Finished processing {chunk_dir}. Found {processed_count} valid pairs, skipped {skipped_count}.")
    return prepared_data

# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(description="Prepare TTS training data from processed chunks.")
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Base directory containing one or more 'final_tts_chunks' directories."
             " Or point directly to a 'final_tts_chunks' directory."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory where the final manifest file will be saved."
    )
    parser.add_argument(
        "--manifest_name", type=str, default="metadata.csv",
        help="Name of the output manifest file (e.g., metadata.csv, list.txt)."
    )
    parser.add_argument(
        "--delimiter", type=str, default="|",
        help="Delimiter to use in the manifest file (e.g., '|' for LJSpeech)."
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search recursively for 'final_tts_chunks' directories within the input_dir."
    )
    parser.add_argument(
        "--split_ratio", type=float, default=0.1,
        help="Fraction of data to use for the validation set (e.g., 0.1 for 10%). Default: 0.1"
    )

    args = parser.parse_args()
    logger.info("Starting TTS data preparation...")

    input_base_path = Path(args.input_dir).resolve()
    output_base_path = Path(args.output_dir)

    # Find the directories to process
    directories_to_process = find_chunk_directories(input_base_path, args.recursive)
    if not directories_to_process:
        logger.warning("No directories found to process. Exiting.")
        return

    # Determine the effective base path for calculating relative paths in the manifest
    # This should be the common parent directory from which the relative paths start.
    # If recursive, use the provided input_dir.
    # If not recursive and a single chunk dir is given, use its parent.
    # If not recursive and multiple chunk dirs are found directly under input_dir, use input_dir.
    if args.recursive:
        relative_base_dir = input_base_path
    elif len(directories_to_process) == 1 and directories_to_process[0] == input_base_path:
        # Input path points directly to the single chunk dir
        relative_base_dir = input_base_path.parent
    else:
        # Input path is the parent containing one or more chunk dirs
        relative_base_dir = input_base_path

    logger.info(f"Using base directory '{relative_base_dir}' for calculating relative paths in manifest.")

    # Process all found directories
    all_prepared_data = []
    for chunk_dir in directories_to_process:
        all_prepared_data.extend(process_chunk_directory(chunk_dir, relative_base_dir))

    if not all_prepared_data:
        logger.error("No valid audio-transcript pairs found after processing all directories. Cannot create manifest.")
        return

    # <<< Add shuffling and splitting logic >>>
    logger.info(f"Shuffling {len(all_prepared_data)} data points...")
    random.seed(args.seed if hasattr(args, 'seed') else 42) # Use seed if provided, else default
    random.shuffle(all_prepared_data)

    split_index = int(len(all_prepared_data) * (1 - args.split_ratio))
    train_data = all_prepared_data[:split_index]
    eval_data = all_prepared_data[split_index:]

    logger.info(f"Splitting data: {len(train_data)} training samples, {len(eval_data)} validation samples.")

    if not train_data:
        logger.error("No training data after split. Check split ratio and total data size.")
        return
    if not eval_data:
        logger.warning("No evaluation data after split. Consider a smaller split ratio or more data.")
        # Proceeding without eval is possible but not recommended

    # Create the manifest files
    train_manifest_path = output_base_path / f"train_{args.manifest_name}"
    eval_manifest_path = output_base_path / f"eval_{args.manifest_name}"

    create_manifest(train_data, train_manifest_path, args.delimiter)
    if eval_data: # Only create eval manifest if there is eval data
        create_manifest(eval_data, eval_manifest_path, args.delimiter)

    logger.info("--- TTS Data Preparation Summary ---")
    logger.info(f"Processed {len(directories_to_process)} chunk directories.")
    logger.info(f"Total valid audio/text pairs: {len(all_prepared_data)}")
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(eval_data)}")
    logger.info(f"Training manifest created at: {train_manifest_path}")
    if eval_data:
        logger.info(f"Validation manifest created at: {eval_manifest_path}")
    logger.info("----------------------------------")

if __name__ == "__main__":
    main() 