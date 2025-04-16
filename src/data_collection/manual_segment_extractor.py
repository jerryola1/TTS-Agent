import argparse
import re
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_time(time_str: str) -> float:
    """Parses a time string (SS, MM:SS, HH:MM:SS) into seconds."""
    parts = time_str.split(':')
    if len(parts) == 1:
        # Assume seconds only
        return float(parts[0])
    elif len(parts) == 2:
        # Assume MM:SS
        return float(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        # Assume HH:MM:SS
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError(f"Invalid time format: {time_str}")

def parse_segment_str(segment_str: str) -> tuple[float, float]:
    """Parses a segment string like 'START-END' into start and end seconds."""
    match = re.match(r'^([\d.:]+)-([\d.:]+)$', segment_str)
    if not match:
        raise ValueError(f"Invalid segment format: '{segment_str}'. Expected format like 'SS-SS', 'MM:SS-MM:SS', 'SS-MM:SS'.")
    
    start_str, end_str = match.groups()
    start_time = parse_time(start_str)
    end_time = parse_time(end_str)

    if start_time >= end_time:
        raise ValueError(f"Start time ({start_time}s) must be less than end time ({end_time}s) in segment '{segment_str}'.")

    return start_time, end_time

def extract_manual_segments(
    input_audio_path: Path,
    artist_name: str,
    segments: list[str],
    output_base_dir: Path = Path("data")
):
    """Extracts specified audio segments and saves them."""
    
    # Validate input audio path
    if not input_audio_path.is_file():
        logger.error(f"Input audio file not found: {input_audio_path}")
        return
        
    # Sanitize artist name for directory creation
    safe_artist_name = artist_name.lower().replace(" ", "_")
    
    # Define and create output directory
    output_dir = output_base_dir / safe_artist_name / "manual_segments"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        return

    # Load audio file
    try:
        logger.info(f"Loading audio file: {input_audio_path}")
        y, sr = librosa.load(str(input_audio_path), sr=None, mono=True) # Load as mono
        duration = librosa.get_duration(y=y, sr=sr)
        logger.info(f"Loaded audio: Duration={duration:.2f}s, Sample Rate={sr}Hz")
    except Exception as e:
        logger.error(f"Failed to load audio file {input_audio_path}: {e}")
        return

    # Process each segment string
    extracted_count = 0
    for i, segment_str in enumerate(segments):
        try:
            start_time, end_time = parse_segment_str(segment_str)
            logger.info(f"Processing segment {i+1}: {segment_str} ({start_time:.2f}s - {end_time:.2f}s)")

            # Check if segment times are within audio duration
            if start_time >= duration:
                logger.warning(f"Segment start time {start_time:.2f}s is beyond audio duration {duration:.2f}s. Skipping segment '{segment_str}'.")
                continue
            if end_time > duration:
                logger.warning(f"Segment end time {end_time:.2f}s is beyond audio duration {duration:.2f}s. Clamping to end.")
                end_time = duration
                if start_time >= end_time: # Re-check after clamping
                    logger.warning(f"Start time now equals or exceeds clamped end time. Skipping segment '{segment_str}'.")
                    continue


            # Convert times to sample indices
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # Extract segment
            segment_audio = y[start_sample:end_sample]

            # Construct output filename
            # Use original stem + manual marker + start/end times
            output_filename = f"{input_audio_path.stem}_manual_{start_time:.2f}-{end_time:.2f}.wav"
            output_path = output_dir / output_filename

            # Save segment
            sf.write(str(output_path), segment_audio, sr)
            logger.info(f"Saved segment {i+1} to: {output_path}")
            extracted_count += 1

        except ValueError as e:
            logger.warning(f"Skipping invalid segment '{segment_str}': {e}")
        except Exception as e:
            logger.error(f"Failed to process segment '{segment_str}': {e}", exc_info=True)

    logger.info(f"Finished processing. Extracted {extracted_count} segments.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manually extract audio segments based on timestamps."
    )
    parser.add_argument(
        "--input_audio", type=str, required=True,
        help="Path to the input audio file (.wav)."
    )
    parser.add_argument(
        "--artist_name", type=str, required=True,
        help="Name of the artist (used for output directory structure)."
    )
    parser.add_argument(
        "--segments", type=str, nargs='+', required=True,
        help="List of segments to extract. Format: 'START-END'. Time can be SS, MM:SS, or HH:MM:SS. Example: '45-51' '1:23-1:35' '55.5-1:12.3'."
    )
    parser.add_argument(
        "--output_base_dir", type=str, default="data",
        help="Base directory for data output (default: 'data')."
    )

    args = parser.parse_args()

    input_path = Path(args.input_audio)
    output_base = Path(args.output_base_dir)

    extract_manual_segments(
        input_audio_path=input_path,
        artist_name=args.artist_name,
        segments=args.segments,
        output_base_dir=output_base
    ) 