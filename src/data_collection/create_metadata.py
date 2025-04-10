import argparse
# import csv # No longer needed
import json # Import json
from pathlib import Path
import soundfile as sf
import logging
import os
from typing import Tuple

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_interview_dirs(
    segments_dir: Path,
    transcripts_dir: Path,
    speaker_id: str,
    dataset_base_dir: Path, # Main artist directory (e.g., data/olamide)
    metadata_dict: dict # Dictionary to update (passed by reference)
) -> Tuple[int, int]:
    """
    Processes audio-text pairs from a specific interview's directories
    and updates the global metadata dictionary.

    Returns:
        Tuple: (number_of_new_entries, number_of_updated_entries) for this interview.
    """
    new_count = 0
    updated_count = 0
    processed_audio_paths_interview = set()

    if not segments_dir.exists():
        logging.warning(f"Segments directory not found for this interview: {segments_dir}")
        return 0, 0
    if not transcripts_dir.exists():
        logging.warning(f"Transcripts directory not found for this interview: {transcripts_dir}")
        return 0, 0

    transcript_files = list(transcripts_dir.glob("*.txt"))
    if not transcript_files:
        logging.debug(f"No transcripts found in: {transcripts_dir}")
        return 0, 0

    logging.info(f"Processing interview: Found {len(transcript_files)} transcripts in {transcripts_dir.parent.name}...")

    for txt_file in transcript_files:
        wav_filename = f"{txt_file.stem}.wav"
        wav_file = segments_dir / wav_filename

        if not wav_file.exists():
            logging.warning(f"Corresponding audio file not found for transcript {txt_file.name} at {wav_file}. Skipping.")
            continue

        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            transcript = " ".join(transcript.split())
            if not transcript:
                 logging.warning(f"Transcript file {txt_file.name} is empty. Skipping.")
                 continue

            info = sf.info(str(wav_file))
            duration = info.duration

            try:
                relative_audio_path = wav_file.relative_to(dataset_base_dir)
                relative_audio_path_str = str(relative_audio_path).replace(os.sep, '/')
            except ValueError:
                logging.warning(f"Audio file {wav_file} is not under the base dir {dataset_base_dir}. Using absolute path.")
                relative_audio_path_str = str(wav_file.resolve()).replace(os.sep, '/')

            processed_audio_paths_interview.add(relative_audio_path_str)

            # Create new data entry
            new_entry = {
                "audio_filepath": relative_audio_path_str,
                "transcript": transcript,
                "speaker_id": speaker_id,
                "duration_seconds": round(duration, 3)
            }

            # Check if this path existed before in the main dict
            if relative_audio_path_str in metadata_dict:
                if metadata_dict[relative_audio_path_str] != new_entry:
                    updated_count += 1
                # Overwrite anyway to ensure latest data
            else:
                new_count += 1

            # Add or overwrite entry in the main dictionary
            metadata_dict[relative_audio_path_str] = new_entry

        except Exception as e:
            logging.error(f"Error processing pair for {txt_file.name} in {segments_dir.parent.name}: {e}", exc_info=True)

    logging.debug(f"  Processed {len(processed_audio_paths_interview)} pairs for {segments_dir.parent.name}. New: {new_count}, Updated: {updated_count}.")
    return new_count, updated_count


def scan_and_create_metadata(
    artist_dir_str: str,
    output_metadata_file_str: str,
    speaker_id: str
):
    """
    Scans the artist's processed directory for completed interviews and
    creates/updates a master metadata JSONL file.
    """
    artist_dir = Path(artist_dir_str)
    processed_dir = artist_dir / "processed" # Standard path convention
    output_metadata_file = Path(output_metadata_file_str)

    if not processed_dir.exists():
        logging.error(f"Artist processed directory not found: {processed_dir}")
        return

    output_metadata_file.parent.mkdir(parents=True, exist_ok=True)

    # --- Load existing metadata (if file exists) ---
    existing_metadata = {}
    if output_metadata_file.exists():
        logging.info(f"Loading existing metadata from: {output_metadata_file}")
        try:
            with open(output_metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if 'audio_filepath' in entry:
                             existing_metadata[entry['audio_filepath']] = entry
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping invalid JSON line in existing metadata: {line.strip()}")
            logging.info(f"Loaded {len(existing_metadata)} existing entries.")
        except Exception as e:
            logging.error(f"Error loading existing metadata file {output_metadata_file}: {e}. Starting fresh.", exc_info=True)
            existing_metadata = {}
    else:
        logging.info("No existing metadata file found. Creating new one.")

    # --- Scan and process interviews ---
    total_new = 0
    total_updated = 0
    interviews_processed_count = 0

    logging.info(f"Scanning for processed interviews in: {processed_dir}")
    for interview_dir in processed_dir.iterdir():
        if not interview_dir.is_dir():
            continue # Skip files, only look at directories

        # Define expected paths based on convention
        segments_dir = interview_dir / "olamide_segments" / "verified_segments" # Assumes 'olamide_segments' structure
        transcripts_dir = interview_dir / "transcriptions"

        # Check if BOTH required directories exist for this interview
        if segments_dir.exists() and transcripts_dir.exists():
            interviews_processed_count += 1
            # Process this interview, updating the existing_metadata dict
            n_new, n_updated = process_interview_dirs(
                segments_dir,
                transcripts_dir,
                speaker_id,
                artist_dir, # Use artist_dir as the base for relative paths
                existing_metadata # Pass the main dictionary
            )
            total_new += n_new
            total_updated += n_updated
        else:
            logging.debug(f"Skipping directory (missing segments or transcripts): {interview_dir.name}")

    logging.info(f"Scan complete. Found and processed {interviews_processed_count} completed interviews.")
    logging.info(f"Total new entries added: {total_new}, Total entries updated: {total_updated}.")

    # --- Write updated metadata back to file (overwrite) ---
    if existing_metadata:
        try:
            with open(output_metadata_file, 'w', encoding='utf-8') as f:
                # Sort by filepath for consistent ordering
                sorted_keys = sorted(existing_metadata.keys())
                for key in sorted_keys:
                    f.write(json.dumps(existing_metadata[key]) + '\n')
            logging.info(f"Metadata successfully saved ({len(existing_metadata)} total entries) to: {output_metadata_file}")
        except Exception as e:
            logging.error(f"Error writing updated metadata file {output_metadata_file}: {e}", exc_info=True)
    else:
        logging.warning("No entries (existing or new) found to write to metadata file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan artist directory and create/update a master metadata JSONL file.")
    # Change flag from --artist_name to --artist
    parser.add_argument("--artist", required=True, help="Name of the artist (used for directory, output file, and speaker ID).")
    # Removed --artist_name
    # parser.add_argument("--artist_name", required=True, help="Name of the artist (used for directory, output file, and speaker ID).")
    # Removed --artist_dir, --output_file, --speaker_id

    args = parser.parse_args()
    
    # Construct paths and speaker_id based on artist_name
    # Use args.artist now
    artist_name_lower = args.artist.lower()
    artist_directory = Path("data") / artist_name_lower
    output_metadata_file = artist_directory / f"metadata_{artist_name_lower}.jsonl"
    speaker_identifier = artist_name_lower

    scan_and_create_metadata(
        str(artist_directory),      # Pass constructed artist_dir path
        str(output_metadata_file),  # Pass constructed output_file path
        speaker_identifier          # Pass artist name as speaker_id
    ) 