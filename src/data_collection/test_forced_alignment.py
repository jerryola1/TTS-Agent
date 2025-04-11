import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path
import sys
from rich.logging import RichHandler
import json

# Setup logging using RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)
logger = logging.getLogger(__name__)

# --- MFA Configuration (Constants related to environment setup) ---
# These paths point to where MFA and its models are installed/located
# They are not specific to the artist/interview being processed.
MFA_EXECUTABLE_PATH = "/home/jerryola1/miniconda3/envs/mfa/bin/mfa"
MFA_CONDA_ENV_BIN_PATH = str(Path(MFA_EXECUTABLE_PATH).parent)
ACOUSTIC_MODEL_PATH = "/home/jerryola1/Documents/MFA/pretrained_models/acoustic/english_us_arpa.zip"
DICTIONARY_PATH = "/home/jerryola1/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict"
# --- End MFA Configuration ---

def prepare_mfa_input(corpus_directory: Path, audio_dir: Path, transcript_dir: Path):
    """
    Copies audio files and reads plain text from transcript TXT files 
    into the structure MFA expects (.lab files).
    """
    logger.info(f"Preparing MFA input directory: {corpus_directory}")
    corpus_directory.mkdir(parents=True, exist_ok=True) # Ensure it exists

    copied_files = 0
    skipped_files = 0

    # Clean the directory before copying
    for item in corpus_directory.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    logger.debug(f"Cleaned temporary corpus directory: {corpus_directory}")

    for audio_file in audio_dir.glob("*.wav"):
        # Look for the .txt transcript file again
        transcript_file_txt = transcript_dir / (audio_file.stem + ".txt")

        if transcript_file_txt.exists():
            try:
                # Read plain text from TXT file
                with open(transcript_file_txt, 'r', encoding='utf-8') as f:
                    plain_text = f.read().strip() 

                if not plain_text:
                     logger.warning(f"Transcript file {transcript_file_txt.name} is empty. Skipping.")
                     skipped_files += 1
                     continue

                # Copy audio file
                shutil.copy2(audio_file, corpus_directory / audio_file.name)
                
                # Write plain text to .lab file
                output_lab_file = corpus_directory / (audio_file.stem + ".lab")
                with open(output_lab_file, 'w', encoding='utf-8') as lab_f:
                    lab_f.write(plain_text) # Write the text read from the .txt file
                
                copied_files += 1
            except Exception as e:
                 logger.error(f"Error processing transcript {transcript_file_txt.name} or copying files for {audio_file.stem}: {e}")
                 skipped_files += 1
        else:
            # Update warning message to reflect looking for .txt
            logger.warning(f"Transcript TXT file {transcript_file_txt.name} not found for {audio_file.name}. Skipping.")
            skipped_files += 1

    logger.info(f"Prepared {copied_files} audio/transcript pairs for MFA. Skipped {skipped_files}.")
    return copied_files > 0 # Return True if any files were prepared

def run_mfa_align_subprocess(corpus_directory: Path, dictionary_path: str, acoustic_model_path: str, output_directory: Path, num_jobs: int = 4):
    """Internal function to run the MFA subprocess with correct environment."""
    logger.info("Running MFA alignment subprocess...")
    output_directory.mkdir(parents=True, exist_ok=True)

    command = [
        MFA_EXECUTABLE_PATH,
        "align",
        str(corpus_directory),
        dictionary_path,
        acoustic_model_path,
        str(output_directory),
        "--clean",
        "--overwrite",
        "--output_format", "long_textgrid",
        f"--jobs={num_jobs}"
    ]

    logger.info(f"Executing MFA command: {' '.join(command)}")

    try:
        process_env = os.environ.copy()
        original_path = process_env.get('PATH', '')
        process_env['PATH'] = f"{MFA_CONDA_ENV_BIN_PATH}{os.pathsep}{original_path}"
        logger.debug(f"Using modified PATH for subprocess: {process_env['PATH']}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            env=process_env
        )

        # Stream STDOUT
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                cleaned_output = output.strip()
                if cleaned_output and not all(c in '━ ' for c in cleaned_output):
                    logger.info(f"[MFA STDOUT] {cleaned_output}")

        # Stream and collect STDERR
        stderr_output_lines = []
        while True:
            error_line = process.stderr.readline()
            stderr_output_lines.append(error_line)
            if error_line == '' and process.poll() is not None:
                break
            if error_line:
                cleaned_error = error_line.strip()
                if cleaned_error:
                    if "error" in cleaned_error.lower() or "failed" in cleaned_error.lower():
                         logger.warning(f"[MFA STDERR] {cleaned_error}")
                    else:
                         logger.info(f"[MFA STDERR] {cleaned_error}")

        return_code = process.poll()

        if return_code == 0:
            logger.info("MFA alignment subprocess completed successfully.")
            return True
        else:
            logger.error(f"MFA alignment subprocess failed with return code {return_code}.")
            logger.error("Collected STDERR output from MFA:")
            for line in stderr_output_lines:
                if line.strip():
                     logger.error(f"[MFA FAIL] {line.strip()}")
            return False

    except FileNotFoundError:
         logger.error(f"Error: MFA executable not found at '{MFA_EXECUTABLE_PATH}'.")
         logger.error("Verify the absolute path in the script configuration.")
         return False
    except Exception as e:
        logger.error(f"An error occurred while running MFA subprocess: {e}", exc_info=True)
        return False

# --- Main function for Agent integration ---
def run_mfa_alignment_on_directory(
    input_audio_dir: Path,
    input_transcript_dir: Path,
    output_alignment_dir: Path,
    mfa_temp_dir: Path, # Renamed for clarity
    num_jobs: int = 4
) -> bool:
    """
    Runs the complete MFA alignment process for a given directory set.
    This function is intended to be called by the agent.

    Args:
        input_audio_dir: Directory containing input .wav files.
        input_transcript_dir: Directory containing corresponding .txt transcript files.
        output_alignment_dir: Directory where output .TextGrid files will be saved.
        mfa_temp_dir: Base directory to create the temporary MFA corpus subdir within.
        num_jobs: Number of parallel jobs for MFA.

    Returns:
        True if alignment was successful, False otherwise.
    """
    logger.info("--- Starting MFA Alignment Step ---")
    
    # Define the specific temporary corpus directory for this run
    mfa_corpus_subdir = mfa_temp_dir / "mfa_corpus_input"

    # --- Input Validation (Basic) ---
    if not input_audio_dir.is_dir():
        logger.error(f"Input audio directory not found: {input_audio_dir}")
        return False
    if not input_transcript_dir.is_dir():
        logger.error(f"Input transcript directory not found: {input_transcript_dir}")
        return False
    # Ensure the base temp dir exists
    mfa_temp_dir.mkdir(parents=True, exist_ok=True)
    # Ensure MFA executable and models exist (using paths from constants)
    if not Path(MFA_EXECUTABLE_PATH).is_file():
         logger.error(f"MFA executable path not found: {MFA_EXECUTABLE_PATH}")
         return False
    if not Path(ACOUSTIC_MODEL_PATH).is_file():
        logger.error(f"Acoustic model path not found: {ACOUSTIC_MODEL_PATH}")
        return False
    if not Path(DICTIONARY_PATH).is_file():
        logger.error(f"Dictionary path not found: {DICTIONARY_PATH}")
        return False
    # --- End Validation ---

    # 1. Prepare the temporary input directory for MFA
    logger.info(f"Using temporary corpus directory: {mfa_corpus_subdir}")
    if not prepare_mfa_input(mfa_corpus_subdir, input_audio_dir, input_transcript_dir):
        logger.error("Failed to prepare any files for MFA. Check input files.")
        # Optionally clean up the (potentially empty) temp dir
        try:
            if mfa_corpus_subdir.exists():
                 shutil.rmtree(mfa_corpus_subdir)
        except Exception as e:
             logger.warning(f"Could not clean up empty temp dir {mfa_corpus_subdir}: {e}")
        return False

    # 2. Run MFA alignment subprocess
    success = run_mfa_align_subprocess(
        corpus_directory=mfa_corpus_subdir,
        dictionary_path=DICTIONARY_PATH, # Use constant path
        acoustic_model_path=ACOUSTIC_MODEL_PATH, # Use constant path
        output_directory=output_alignment_dir,
        num_jobs=num_jobs
    )

    # 3. Clean up the temporary directory on success or failure
    logger.info(f"Cleaning up temporary directory: {mfa_corpus_subdir}")
    try:
        if mfa_corpus_subdir.exists():
             shutil.rmtree(mfa_corpus_subdir)
             logger.info(f"Successfully removed temporary directory.")
        else:
             logger.info(f"Temporary directory already removed or not created.")
    except Exception as e:
        logger.warning(f"Could not remove temporary directory {mfa_corpus_subdir}: {e}")

    if success:
        logger.info(f"--- MFA Alignment Step Finished Successfully ---")
    else:
        logger.error("--- MFA Alignment Step Failed ---")
        
    return success


# --- Standalone execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Montreal Forced Aligner (MFA) on audio and transcript files.")
    parser.add_argument(
        "--input_audio_dir", type=str, required=True,
        help="Directory containing the input audio segments (.wav files)."
    )
    parser.add_argument(
        "--input_transcript_dir", type=str, required=True,
        help="Directory containing corresponding transcripts (.txt files)."
    )
    parser.add_argument(
        "--output_alignment_dir", type=str, required=True,
        help="Directory where MFA will save the output TextGrid files."
    )
    # Changed default: temporary dir will be created *within* the output dir
    parser.add_argument(
        "--mfa_temp_base_dir", type=str, default=None,
        help="Base directory for MFA temporary files. Defaults to output_alignment_dir/_temp_mfa."
    )
    parser.add_argument(
        "--jobs", type=int, default=4,
        help="Number of parallel jobs for MFA to use."
    )
    # Removed args for overriding MFA paths, rely on constants or env vars if needed later

    args = parser.parse_args()

    input_audio_path = Path(args.input_audio_dir)
    input_transcript_path = Path(args.input_transcript_dir)
    output_alignment_path = Path(args.output_alignment_dir)

    # Determine base directory for temporary files
    if args.mfa_temp_base_dir:
        mfa_temp_base_path = Path(args.mfa_temp_base_dir)
    else:
        # Default to a subdir within the output alignment directory
        mfa_temp_base_path = output_alignment_path / "_temp_mfa"
        logger.info(f"Using default temporary base directory: {mfa_temp_base_path}")

    # Call the main processing function
    run_mfa_alignment_on_directory(
        input_audio_dir=input_audio_path,
        input_transcript_dir=input_transcript_path,
        output_alignment_dir=output_alignment_path,
        mfa_temp_dir=mfa_temp_base_path, # Pass the base temp dir
        num_jobs=args.jobs
    )