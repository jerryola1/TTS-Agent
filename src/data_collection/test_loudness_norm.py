import argparse
import soundfile as sf
import numpy as np
import pyloudnorm as pyln
from pathlib import Path
from typing import Optional
import logging

# Configure basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_audio_loudness(input_path_str: str, output_path_str: str, target_lufs: float = -20.0) -> Optional[Path]:
    """Normalize the loudness of an audio file to a target LUFS level."""
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)

    if not input_path.exists():
        logging.error(f"Input audio file not found: {input_path}")
        return None

    try:
        # Load audio data
        logging.info(f"Loading audio from: {input_path}")
        data, rate = sf.read(str(input_path))
        logging.info(f"Audio loaded: sample rate={rate}, shape={data.shape}")

        # Check for silence or very short audio
        if data.size == 0 or np.max(np.abs(data)) < 1e-6:
            logging.warning(f"Input audio is silent or near-silent: {input_path}. Skipping normalization, copying original.")
            output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
            sf.write(str(output_path), data, rate)
            return output_path
        
        # Measure loudness
        meter = pyln.Meter(rate) # create BS.1770 meter
        loudness = meter.integrated_loudness(data) # measure loudness
        logging.info(f"Measured integrated loudness: {loudness:.2f} LUFS")

        # Calculate gain adjustment
        # Check if loudness is valid (not -inf, which happens for pure silence)
        if not np.isfinite(loudness):
            logging.warning(f"Could not measure valid loudness (possibly silence): {input_path}. Skipping normalization, copying original.")
            output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
            sf.write(str(output_path), data, rate)
            return output_path
            
        delta_lufs = target_lufs - loudness
        gain_linear = 10.0**(delta_lufs / 20.0)
        logging.info(f"Target LUFS: {target_lufs:.2f} LUFS. Gain to apply: {delta_lufs:.2f} dB ({gain_linear:.3f} linear)")

        # Apply gain
        normalized_data = data * gain_linear
        
        # --- Peak Limiting ---
        # Check for clipping and apply limiting to avoid distortion
        max_peak = np.max(np.abs(normalized_data))
        if max_peak > 1.0:
            logging.warning(f"Potential clipping detected after normalization (max peak: {max_peak:.3f}). Applying peak limiting (hard clip at -0.99).")
            # Simple hard clipping (adjust target slightly below 0dBFS)
            peak_target = 0.99 
            normalized_data = np.clip(normalized_data, -peak_target, peak_target)

        # Save normalized audio
        logging.info(f"Saving normalized audio to: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
        sf.write(str(output_path), normalized_data, rate)
        logging.info("Normalization successful.")
        
        return output_path

    except Exception as e:
        logging.error(f"Error during loudness normalization for {input_path}: {e}", exc_info=True) # Log traceback
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize audio loudness to a target LUFS level.")
    parser.add_argument("--audio", required=True, help="Path to the input audio file.")
    parser.add_argument("--output", required=True, help="Path to save the normalized output audio file.")
    parser.add_argument("--target_lufs", type=float, default=-20.0, help="Target loudness level in LUFS (e.g., -20.0)")
    args = parser.parse_args()

    normalized_path = normalize_audio_loudness(args.audio, args.output, args.target_lufs)

    if normalized_path:
        logging.info(f"\nStandalone test successful. Normalized file: {normalized_path}")
    else:
        logging.error("\nStandalone test failed.")
