import argparse
import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement
from pathlib import Path
import soundfile as sf
import numpy as np
import logging
import math

# Configure basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def enhance_audio_chunk(enhance_model, audio_chunk, device):
    """
    Process a chunk of audio with the SpeechBrain enhancement model.
    Pads the chunk with one extra sample to avoid indexing issues.
    Returns the enhanced chunk as a NumPy array.
    """
    # Pad the chunk with one zero sample at the end
    if len(audio_chunk) == 0:
        return audio_chunk
    padded_chunk = np.pad(audio_chunk, (0, 1), mode='constant')
    
    # Convert padded chunk to torch tensor and add batch dimension [1, time]
    audio_tensor = torch.from_numpy(padded_chunk.astype(np.float32)).to(device).unsqueeze(0)
    lengths_tensor = torch.tensor([audio_tensor.shape[1]]).to(device)
    
    with torch.no_grad():
        enhanced_tensor = enhance_model.enhance_batch(audio_tensor.contiguous(), lengths=lengths_tensor)
    
    # Remove batch dimension and convert back to numpy array
    # Remove the extra padded sample to match the original chunk length
    enhanced_np = enhanced_tensor.squeeze(0).cpu().numpy()
    return enhanced_np[:-1]

def process_audio(input_path_str: str, output_path_str: str, chunk_duration: float = 10.0) -> Path:
    """
    Enhance speech in an audio file using SpeechBrain's enhancement model.
    Processes the audio in chunks to avoid memory errors.
    """
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logging.error(f"Input audio file not found: {input_path}")
        return None

    # --- Force CPU Usage ---
    effective_device = "cpu"
    logging.info("Forcing CPU for SpeechBrain enhancement.")

    try:
        # Load the enhancement model onto CPU
        logging.info("Loading SpeechBrain enhancement model (MetricGAN-plus-voicebank) onto CPU...")
        savedir = Path("pretrained_models") / "speechbrain_metricgan-plus-voicebank"
        enhance_model = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir=str(savedir),
            run_opts={"device": effective_device}
        )
        logging.info("Model loaded successfully.")

        # Load the entire audio file
        logging.info(f"Loading audio from: {input_path}")
        data, sr = sf.read(str(input_path), dtype='float32')
        total_samples = len(data)
        logging.info(f"Input audio: {sr} Hz, {total_samples} samples ({total_samples/sr:.2f} seconds)")

        # If stereo, convert to mono
        if data.ndim > 1 and data.shape[1] > 1:
            logging.warning("Input appears stereo, converting to mono for enhancement.")
            data = np.mean(data, axis=1)
        data = data.astype(np.float32)

        # Determine chunk size in samples and number of chunks
        chunk_size = int(chunk_duration * sr)
        num_chunks = math.ceil(total_samples / chunk_size)
        logging.info(f"Splitting audio into {num_chunks} chunk(s) of ~{chunk_duration} seconds each.")

        enhanced_chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_samples)
            logging.info(f"Processing chunk {i+1}/{num_chunks} (samples {start} to {end})...")
            chunk = data[start:end]
            enhanced_chunk = enhance_audio_chunk(enhance_model, chunk, effective_device)
            enhanced_chunks.append(enhanced_chunk)
        
        enhanced_audio = np.concatenate(enhanced_chunks)
        logging.info(f"Enhanced audio shape: {enhanced_audio.shape}")
        logging.info(f"Input RMS: {np.sqrt(np.mean(data**2)):.6f}")
        logging.info(f"Output RMS: {np.sqrt(np.mean(enhanced_audio**2)):.6f}")

        # Save the enhanced full audio
        logging.info(f"Saving enhanced audio to: {output_path}")
        sf.write(str(output_path), enhanced_audio, sr, subtype='PCM_16')
        logging.info("Speech enhancement successful.")

        return output_path

    except Exception as e:
        logging.error(f"Error during SpeechBrain enhancement for {input_path}: {e}", exc_info=True)
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Enhance speech in an audio file by processing in chunks using SpeechBrain's model (CPU)"
    )
    parser.add_argument("--input", required=True, help="Path to input audio file (WAV recommended)")
    parser.add_argument("--output", required=True, help="Path to save the enhanced audio file")
    parser.add_argument("--chunk_duration", type=float, default=10.0,
                        help="Duration (in seconds) of each chunk (default: 10.0 seconds)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = input_path.parent / "speechbrain_enhanced"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_enhanced.wav"

    enhanced_path = process_audio(str(input_path), str(output_path), args.chunk_duration)

    if enhanced_path:
        logging.info(f"\nEnhancement test successful. Enhanced file: {enhanced_path}")
    else:
        logging.error("\nEnhancement test failed.")

if __name__ == "__main__":
    main()
