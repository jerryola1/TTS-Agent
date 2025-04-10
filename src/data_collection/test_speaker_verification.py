import torch
import torchaudio
from pathlib import Path
import numpy as np
from speechbrain.pretrained import SpeakerRecognition
import soundfile as sf
import json
import librosa
from typing import Optional

def split_audio(file_path, chunk_duration=30, device=None):
    """Split audio into chunks of specified duration."""
    # Load audio
    y, sr = librosa.load(file_path, sr=None)
    
    # Calculate number of samples per chunk
    samples_per_chunk = int(chunk_duration * sr)
    
    # Split into chunks
    chunks = []
    for i in range(0, len(y), samples_per_chunk):
        chunk = y[i:i + samples_per_chunk]
        if len(chunk) > 0:  # Only add non-empty chunks
            if device:
                chunk = torch.from_numpy(chunk).to(device)
            chunks.append(chunk)
    
    return chunks, sr

def load_audio(file_path, device=None):
    """Load audio file and return waveform and sample rate."""
    # If file is too large (> 1 minute), split it
    duration = librosa.get_duration(path=file_path)
    if duration > 60:
        chunks, sr = split_audio(file_path, device=device)
        return chunks, sr
    else:
        waveform, sample_rate = torchaudio.load(file_path)
        if device:
            waveform = waveform.to(device)
        return [waveform], sample_rate

def create_speaker_embedding(model, audio_paths):
    """Create an average embedding from multiple reference samples."""
    device = next(model.parameters()).device
    model.eval()
    
    embeddings = []
    for path in audio_paths:
        # Load audio and move to device
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.to(device)
        
        # Ensure waveform is in correct shape [batch, channels, time]
        if waveform.dim() == 2:  # [channels, time]
            waveform = waveform.unsqueeze(0)  # Add batch dimension
        
        # Encode the waveform
        with torch.no_grad():
            embedding = model.encode_batch(waveform)
            embeddings.append(embedding.squeeze().cpu().numpy())
    
    # Average the embeddings
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

def verify_speaker(model, reference_embedding, test_audio_path, threshold=0.7):
    """Verify if test audio matches the reference speaker."""
    device = next(model.parameters()).device
    model.eval()
    
    # Load and encode test audio
    waveform, sample_rate = torchaudio.load(test_audio_path)
    waveform = waveform.to(device)
    
    # Ensure waveform is in correct shape [batch, channels, time]
    if waveform.dim() == 2:  # [channels, time]
        waveform = waveform.unsqueeze(0)  # Add batch dimension
    
    # Encode the waveform
    with torch.no_grad():
        test_embedding = model.encode_batch(waveform).squeeze().cpu().numpy()
    
    # Calculate cosine similarity
    similarity = np.dot(reference_embedding, test_embedding) / (
        np.linalg.norm(reference_embedding) * np.linalg.norm(test_embedding)
    )
    
    is_same_speaker = similarity > threshold
    
    return {
        'similarity': float(similarity),
        'is_same_speaker': bool(is_same_speaker),
        'threshold': float(threshold)
    }

def verify_speaker_segments(
    reference_dir: Path,
    test_dir: Path,
    output_file: Path,
    threshold: float = 0.7,
    model: Optional[SpeakerRecognition] = None
):
    """
    Verify speaker identity for all segments in test_dir against reference samples.
    Can optionally use a pre-loaded model.
    """
    # Initialize the speaker recognition model IF NOT PROVIDED
    if model is None:
        print("Loading speaker recognition model...")
        model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
    else:
        print("Using provided speaker recognition model...")

    model.eval()
    # Derive device from the model (either passed in or loaded)
    device = next(model.parameters()).device
    
    # Get reference samples
    reference_files = list(reference_dir.glob("*.wav"))
    print(f"\nFound {len(reference_files)} reference samples")
    
    # Create reference embedding
    print("\nCreating reference embedding...")
    reference_embeddings = []
    with torch.no_grad():
        for path in reference_files:
            # Load audio file
            waveform, sample_rate = torchaudio.load(str(path))
            # Move waveform to the correct device
            waveform = waveform.to(device)
            
            # Use the model's built-in verification method
            embedding = model.encode_batch(waveform)
            reference_embeddings.append(embedding.squeeze().cpu().numpy())
    
    # Average the embeddings
    reference_embedding = np.mean(reference_embeddings, axis=0)
    
    # Test against segments
    test_files = list(test_dir.glob("*.wav"))
    print(f"\nTesting speaker verification on {len(test_files)} segments...")
    
    results = {}
    with torch.no_grad():
        for test_file in test_files:
            print(f"\nTesting {test_file.name}...")
            # Load audio file
            waveform, sample_rate = torchaudio.load(str(test_file))
            # Move waveform to the correct device
            waveform = waveform.to(device)
            
            # Use the model's built-in verification method
            test_embedding = model.encode_batch(waveform)
            test_embedding = test_embedding.squeeze().cpu().numpy()
            
            # Calculate cosine similarity
            similarity = np.dot(reference_embedding, test_embedding) / (
                np.linalg.norm(reference_embedding) * np.linalg.norm(test_embedding)
            )
            
            is_same_speaker = similarity > threshold
            results[test_file.name] = {
                'similarity': float(similarity),
                'is_same_speaker': bool(is_same_speaker),
                'threshold': float(threshold)
            }
            print(f"Similarity: {similarity:.4f}")
            print(f"Same speaker: {is_same_speaker}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return results

if __name__ == "__main__":
    # Example usage with correct paths (will load model internally)
    reference_dir = Path("data/olamide/reference_samples")
    test_dir = Path("data/olamide/processed/The Juice - Olamide/diarized_segments")
    output_file = Path("data/olamide/speaker_verification_results.json")
    # Call without passing model - it will load internally as before
    verify_speaker_segments(reference_dir, test_dir, output_file) 