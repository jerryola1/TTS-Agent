import shutil
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Assuming your data is under 'data/olamide/'
# Adjust 'olamide' if your artist name directory is different.
artist_name = "olamide"
artist_base_dir = Path(f"data/{artist_name}")
processed_dir = artist_base_dir / "processed"
manual_segments_dir = artist_base_dir / "manual_segments"

target_dir = Path("alltalk_tts/finetune/put-voice-samples-in-here")

# Directories to search for WAV files
source_dirs_to_search = {
    "processed": ("final_tts_chunks", processed_dir),
    "manual": ("manual_segments", manual_segments_dir)
}
# --- End Configuration ---

def copy_wav_files(source_dir: Path, target_dir: Path):
    """Copies all .wav files from source_dir to target_dir, skipping duplicates."""
    global copied_count, skipped_count
    
    if not source_dir.is_dir():
        logger.warning(f"Source directory not found or not a directory: {source_dir}. Skipping.")
        return
        
    logger.info(f"Processing directory: {source_dir}")
    wav_files = list(source_dir.glob("*.wav"))
    
    if not wav_files:
        logger.info(f"  No .wav files found in {source_dir}")
        return

    for wav_file in wav_files:
        target_file_path = target_dir / wav_file.name
        # Optional: Check if file already exists to avoid duplicates if run multiple times
        if target_file_path.exists():
             logger.info(f"  Skipping, already exists: {wav_file.name}")
             skipped_count += 1
             continue
             
        try:
            shutil.copy(str(wav_file), str(target_dir))
            logger.info(f"  Copied: {wav_file.name}")
            copied_count += 1
        except Exception as e:
            logger.error(f"  Error copying {wav_file.name}: {e}")

# Ensure target directory exists
try:
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target directory ensured: {target_dir.resolve()}")
except OSError as e:
    logger.error(f"Failed to create target directory {target_dir}: {e}")
    exit(1) # Exit if we can't create the target dir

copied_count = 0
skipped_count = 0

logger.info(f"Searching for WAV files to copy to {target_dir}...")

# 1. Process 'final_tts_chunks' directories under the processed directory
if "processed" in source_dirs_to_search:
    dir_name, base_path = source_dirs_to_search["processed"]
    logger.info(f"Searching for '{dir_name}' directories under {base_path}...")
    final_chunk_dirs = base_path.rglob(dir_name)
    found_processed = False
    for chunk_dir in final_chunk_dirs:
        copy_wav_files(chunk_dir, target_dir)
        found_processed = True
    if not found_processed:
         logger.info(f"No '{dir_name}' directories found under {base_path}.")

# 2. Process 'manual_segments' directory directly under the artist base directory
if "manual" in source_dirs_to_search:
    dir_name, base_path = source_dirs_to_search["manual"]
    logger.info(f"Checking for '{dir_name}' directory: {base_path}...")
    copy_wav_files(base_path, target_dir) # Pass the direct path

logger.info("\n--- Summary ---")
logger.info(f"Successfully copied: {copied_count} files.")
logger.info(f"Skipped (already exist): {skipped_count} files.")
logger.info(f"Target directory: {target_dir.resolve()}")