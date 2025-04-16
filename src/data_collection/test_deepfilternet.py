import argparse
import soundfile as sf
import torch
import numpy as np
import os
from pathlib import Path
from df import enhance, init_df
import df # Import the module itself to check its path
import math

# Print the path of the imported df module
print(f"Importing df module from: {df.__file__}")

def process_audio_in_chunks(input_path, output_path, chunk_duration_s=60, overlap_s=5):
    """Process audio in chunks using DeepFilterNet (Overlap-Add method)."""
    print("\n--- Processing Audio with DeepFilterNet (Chunked Overlap-Add) ---")
    
    # Check CUDA status
    cuda_available = torch.cuda.is_available()
    device = "cpu" # Force CPU usage
    print(f"CUDA available: {cuda_available}")
    print(f"Using device: {device} (Forced CPU for compatibility)")
    
    # Load the audio file
    print(f"\nLoading audio file: {input_path}")
    audio, sr = sf.read(input_path)
    
    print(f"Input audio shape: {audio.shape}")
    print(f"Sample rate: {sr}")
    
    # If stereo, convert to mono
    if audio.ndim > 1:
        print("Converting stereo to mono...")
        audio = np.mean(audio, axis=1)
    
    # Ensure audio is float32 and contiguous
    audio = np.ascontiguousarray(audio.astype(np.float32))
    total_len = len(audio)
    
    # Initialize the DeepFilterNet model
    print("\nLoading DeepFilterNet model...")
    model, df_state, _ = init_df()
    model.eval()

    # Calculate chunk parameters in samples
    chunk_len = int(chunk_duration_s * sr)
    overlap_len = int(overlap_s * sr)
    step = chunk_len - overlap_len
    num_chunks = math.ceil(total_len / step)
    
    print(f"Processing in {num_chunks} chunks of {chunk_duration_s}s with {overlap_s}s overlap ({chunk_len} samples, step {step})...")

    # Output buffer and normalization buffer
    output_audio = np.zeros(total_len, dtype=np.float32)
    norm_factor = np.zeros(total_len, dtype=np.float32)
    
    # Use a Hann window for overlap-add
    window = np.hanning(chunk_len)

    with torch.no_grad():
        for i in range(num_chunks):
            start = i * step
            end = start + chunk_len
            
            # Get the current chunk from original audio
            current_chunk = audio[start:min(end, total_len)]
            current_len = len(current_chunk)

            # Pad chunk if it's shorter than chunk_len (usually the last one)
            if current_len < chunk_len:
                padding_len = chunk_len - current_len
                padded_chunk = np.pad(current_chunk, (0, padding_len), mode='constant')
            else:
                padded_chunk = current_chunk
                padding_len = 0

            print(f"  Processing chunk {i+1}/{num_chunks} (samples {start}-{start+current_len})...")

            # Convert chunk to tensor, ensure contiguous, move to device
            chunk_tensor = torch.from_numpy(padded_chunk).unsqueeze(0) 
            input_tensor = chunk_tensor.to(device).contiguous()

            # Enhance chunk
            enhanced_chunk_tensor = enhance(model, df_state, input_tensor)
            enhanced_padded_chunk = enhanced_chunk_tensor.squeeze(0).cpu().numpy()

            # Apply window
            windowed_enhanced_chunk = enhanced_padded_chunk * window

            # Overlap-Add to output buffer
            output_slice = slice(start, start + chunk_len)
            # Determine the actual length of the slice in the output buffer
            actual_slice_len = len(output_audio[output_slice])
            
            # Add the correctly truncated windowed chunk
            output_audio[output_slice] += windowed_enhanced_chunk[:actual_slice_len]
            
            # Add the correctly truncated window squared to normalization factor
            norm_factor[output_slice] += (window**2)[:actual_slice_len]

    # Normalize the output audio where overlap occurred
    # Avoid division by zero for sections without processing (shouldn't happen here)
    norm_factor[norm_factor < 1e-8] = 1.0  
    output_audio /= norm_factor
    
    # Trim potential silence at the end if output is longer (unlikely with this method, but safe)
    output_audio = output_audio[:total_len]

    # Print output audio statistics
    print(f"\nOutput audio shape: {output_audio.shape}")
    print(f"Input RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    print(f"Output RMS: {np.sqrt(np.mean(output_audio**2)):.6f}")
    
    # Save the enhanced audio
    print(f"\nSaving enhanced audio to: {output_path}")
    sf.write(output_path, output_audio, sr)
    print("Chunked Overlap-Add processing completed successfully!")

def main():
    parser = argparse.ArgumentParser(
        description="Apply DeepFilterNet noise reduction using chunked Overlap-Add."
    )
    parser.add_argument("--audio", required=True, help="Path to input audio file (WAV recommended)")
    parser.add_argument("--chunk_s", type=int, default=60, help="Chunk duration in seconds")
    parser.add_argument("--overlap_s", type=int, default=5, help="Overlap duration in seconds")
    args = parser.parse_args()
    
    input_path = Path(args.audio)
    # Keep output name simple, overwrite previous chunked attempt if exists
    output_dir = input_path.parent / "deepfilternet_cleaned"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_cleaned.wav"
    
    process_audio_in_chunks(str(input_path), str(output_path), args.chunk_s, args.overlap_s)

if __name__ == "__main__":
    main()
