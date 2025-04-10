import os
import argparse
from pathlib import Path
import soundfile as sf
import numpy as np
import librosa
import tensorflow as tf
from typing import Optional
from pydub import AudioSegment
import tempfile
import shutil
from spleeter.separator import Separator

# Initialize Spleeter separator at module level
separator = Separator('spleeter:4stems')

def split_audio_file(audio_path: str, chunk_length_ms: int = 30000) -> list:
    """Split an audio file into smaller chunks."""
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunks.append(chunk)
    return chunks

def process_chunk_with_spleeter(chunk: AudioSegment, output_dir: Path) -> Optional[np.ndarray]:
    """Process a single audio chunk with Spleeter and return only vocals."""
    # Create a unique temporary directory for this chunk
    temp_dir = tempfile.mkdtemp()
    try:
        # Save chunk to temporary file
        temp_audio_path = os.path.join(temp_dir, "chunk.wav")
        chunk.export(temp_audio_path, format='wav')

        # Process with Spleeter
        separator.separate_to_file(temp_audio_path, temp_dir)

        # Load only vocals
        vocals_path = os.path.join(temp_dir, "chunk", "vocals.wav")
        if os.path.exists(vocals_path):
            vocals, _ = librosa.load(vocals_path, sr=None)
            return vocals
        else:
            print(f"Vocals file not found at {vocals_path}")
            return None

    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
        return None
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error cleaning up temporary directory: {str(e)}")

def test_spleeter(audio_path: str, output_dir: Path) -> Optional[Path]:
    """
    Run Spleeter source separation on an audio file and save results.
    Processes the audio in chunks to manage memory usage.
    """
    print("\n--- Running Spleeter Source Separation ---")
    
    # Verify GPU availability
    print("\nChecking GPU availability...")
    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices('GPU'))
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found, using CPU")

    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"Error: Audio file not found at {audio_path}")
        return None

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vocals.wav"

    print(f"\nSeparating sources in {audio_file.name}...")
    print(f"Output will be saved to: {output_path}")
    print("Running source separation (this may take a while)...")

    try:
        # Split audio into chunks
        chunks = split_audio_file(audio_path)
        print(f"Split audio into {len(chunks)} chunks.")
        
        # Process each chunk and write directly to output file
        with sf.SoundFile(str(output_path), 'w', 44100, 2) as outfile:
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i + 1}/{len(chunks)}...")
                vocals = process_chunk_with_spleeter(chunk, output_dir)
                
                if vocals is not None:
                    # Ensure vocals are stereo (2 channels)
                    if len(vocals.shape) == 1:
                        vocals = np.column_stack((vocals, vocals))
                    outfile.write(vocals)
                else:
                    print(f"Failed to process chunk {i + 1}")
        
        print(f"\nSuccessfully processed and saved vocals to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error during Spleeter separation: {e}")
        tf.keras.backend.clear_session()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Spleeter installation and source separation")
    parser.add_argument("--audio", required=True, help="Path to an audio file for testing source separation")
    parser.add_argument("--output", default="data/spleeter_output_test", help="Directory to save separation results")
    args = parser.parse_args()

    output_path = Path(args.output)
    vocals_file_path = test_spleeter(args.audio, output_path)

    if vocals_file_path:
        print(f"\nStandalone test successful. Vocals file located at: {vocals_file_path}")
    else:
        print("\nStandalone test failed.")
