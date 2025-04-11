import torch
import torchaudio
from pathlib import Path
import numpy as np
from speechbrain.pretrained import SpeakerRecognition
import json
import librosa
import re
from collections import Counter
import shutil
from typing import Optional, Dict, List, Tuple
import logging

# --- Logging Setup (Basic, consider Rich later) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

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
    """Create an average embedding from multiple reference samples, ensuring mono audio."""
    device = next(model.parameters()).device
    model.eval()
    embeddings = []
    with torch.no_grad():
        for path in audio_paths:
            try:
                 waveform, sample_rate = torchaudio.load(path)
                 waveform = waveform.to(device)

                 # --- Ensure MONO audio ---
                 if waveform.shape[0] > 1:
                     logger.debug(f"Reference file {path.name} has multiple channels ({waveform.shape[0]}). Converting to mono.")
                     waveform = torch.mean(waveform, dim=0, keepdim=True)
                 # --- End MONO conversion ---

                 # Ensure shape is [batch, time] for the model
                 if waveform.dim() == 2 and waveform.shape[0] == 1: # If shape is [1, time]
                     waveform = waveform.squeeze(0) # Make it [time]
                 if waveform.dim() == 1: # Add batch dimension if needed
                     waveform = waveform.unsqueeze(0) # -> [1, time]
                 # --- End shape adjustment ---

                 # waveform should now be [1, time]
                 if waveform.dim() != 2 or waveform.shape[0] != 1:
                      logger.warning(f"Unexpected waveform shape {waveform.shape} after processing for {path.name}. Skipping.")
                      continue

                 embedding = model.encode_batch(waveform)
                 embeddings.append(embedding.squeeze().cpu().numpy())
            except Exception as e:
                 logger.warning(f"Could not process reference file {path}: {e}")

    if not embeddings:
         logger.error("No valid reference embeddings could be created.")
         return None
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

def verify_speaker(model, reference_embedding, test_audio_path, threshold=0.7):
    """Verify if test audio matches the reference speaker, ensuring mono audio."""
    device = next(model.parameters()).device
    model.eval()
    try:
        waveform, sample_rate = torchaudio.load(test_audio_path)
        waveform = waveform.to(device)

        # --- Ensure MONO audio ---
        if waveform.shape[0] > 1:
            logger.debug(f"Test file {test_audio_path.name} has multiple channels ({waveform.shape[0]}). Converting to mono.")
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # --- End MONO conversion ---

        # Ensure shape is [batch, time] for the model
        if waveform.dim() == 2 and waveform.shape[0] == 1: # If shape is [1, time]
            waveform = waveform.squeeze(0) # Make it [time]
        if waveform.dim() == 1: # Add batch dimension if needed
            waveform = waveform.unsqueeze(0) # -> [1, time]
        # --- End shape adjustment ---

        if waveform.dim() != 2 or waveform.shape[0] != 1:
             logger.warning(f"Unexpected waveform shape {waveform.shape} after processing for {test_audio_path.name}. Verification skipped.")
             return None # Cannot verify if shape is wrong

        with torch.no_grad():
            test_embedding = model.encode_batch(waveform).squeeze().cpu().numpy()

        # Avoid division by zero if norm is zero (e.g., silent audio)
        norm_ref = np.linalg.norm(reference_embedding)
        norm_test = np.linalg.norm(test_embedding)
        if norm_ref == 0 or norm_test == 0:
             similarity = 0.0
             logger.warning(f"Zero norm embedding encountered for {test_audio_path.name}. Similarity set to 0.")
        else:
            similarity = np.dot(reference_embedding, test_embedding) / (norm_ref * norm_test)

        is_same_speaker = similarity > threshold
        return {
            'similarity': float(similarity),
            'is_same_speaker': bool(is_same_speaker),
            'threshold': float(threshold)
        }
    except Exception as e:
         logger.warning(f"Could not process test file {test_audio_path}: {e}")
         return None

def verify_speaker_segments(
    reference_dir: Path,
    diarized_segments_dir: Path,
    output_dir: Path,
    results_output_file: Path,
    threshold: float = 0.75,
    model: Optional[SpeakerRecognition] = None
) -> List[Path]:
    """
    Verifies speaker identity for segments, applying post-filtering heuristic
    to keep only the most frequent verified speaker ID.

    Args:
        reference_dir: Directory with reference audio files (.wav) for the target speaker.
        diarized_segments_dir: Directory with diarized segments (.wav) to test.
        output_dir: Directory to save the final, filtered verified segments.
        results_output_file: Path to save the JSON results of the verification process.
        threshold: Cosine similarity threshold for verification.
        model: Optional pre-loaded SpeechBrain SpeakerRecognition model.

    Returns:
        A list of Path objects pointing to the finally verified and copied audio segments.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the speaker recognition model IF NOT PROVIDED
    if model is None:
        logger.info("Loading speaker recognition model...")
        try:
            model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
        except Exception as e:
            logger.error(f"Failed to load SpeechBrain model: {e}")
            return []
    else:
        logger.info("Using provided speaker recognition model...")

    model.eval()
    device = next(model.parameters()).device
    logger.info(f"Using device: {device}")

    # --- Create Reference Embedding ---
    reference_files = list(reference_dir.glob("*.wav"))
    if not reference_files:
        logger.error(f"No reference .wav files found in {reference_dir}")
        return []
    logger.info(f"Found {len(reference_files)} reference samples in {reference_dir}")
    logger.info("Creating reference embedding...")
    
    reference_embedding = create_speaker_embedding(model, reference_files)
    if reference_embedding is None:
        logger.error("Failed to create reference embedding.")
        return []
    logger.info("Reference embedding created successfully.")

    # --- Verify All Segments ---
    test_files = list(diarized_segments_dir.glob("*.wav"))
    if not test_files:
        logger.warning(f"No test segments found in {diarized_segments_dir}")
        return []
    logger.info(f"Verifying {len(test_files)} segments from {diarized_segments_dir}...")

    verification_results = {}
    potential_verified_segments = []

    # Regex to extract speaker ID (e.g., SPEAKER_00)
    speaker_id_pattern = re.compile(r"_(SPEAKER_\d+)_")

    for test_file in test_files:
        logger.debug(f"Processing segment: {test_file.name}")
        
        match = speaker_id_pattern.search(test_file.name)
        original_speaker_id = match.group(1) if match else "UNKNOWN_SPEAKER"
        
        result = verify_speaker(model, reference_embedding, test_file, threshold)
        if result is None:
            verification_results[test_file.name] = {"error": "Verification failed", "original_speaker_id": original_speaker_id}
            continue

        verification_results[test_file.name] = result
        verification_results[test_file.name]["original_speaker_id"] = original_speaker_id

        if result['is_same_speaker']:
            logger.info(f"  Segment {test_file.name} PASSED initial threshold (Similarity: {result['similarity']:.4f}, Speaker: {original_speaker_id})")
            potential_verified_segments.append( (test_file, original_speaker_id, result['similarity']) )
        else:
            logger.debug(f"  Segment {test_file.name} FAILED initial threshold (Similarity: {result['similarity']:.4f}, Speaker: {original_speaker_id})")

    # --- Post-filtering Heuristic ---
    if not potential_verified_segments:
        logger.warning("No segments passed the initial verification threshold.")
        # Save results file even if empty
        try:
            with open(results_output_file, 'w') as f:
                json.dump(verification_results, f, indent=2)
            logger.info(f"Verification results (no segments passed) saved to {results_output_file}")
        except Exception as e:
            logger.error(f"Failed to save empty results file {results_output_file}: {e}")
        return []

    logger.info(f"Applying post-filtering: Found {len(potential_verified_segments)} potential segments.")
    
    # Count occurrences of each original speaker ID among potential segments
    speaker_counts = Counter(speaker_id for _, speaker_id, _ in potential_verified_segments)
    logger.info(f"Speaker ID counts among potential segments: {dict(speaker_counts)}")

    if not speaker_counts:
        logger.warning("No speaker IDs found in potential segments.")
        return []

    # Find the speaker ID with the maximum count
    most_frequent_speaker_id = speaker_counts.most_common(1)[0][0]
    max_count = speaker_counts.most_common(1)[0][1]
    logger.info(f"Most frequent verified speaker ID: {most_frequent_speaker_id} (Count: {max_count})")

    # --- Filter and Copy Final Segments ---
    final_verified_paths = []
    for segment_path, speaker_id, similarity in potential_verified_segments:
        if speaker_id == most_frequent_speaker_id:
            dest_path = output_dir / segment_path.name
            try:
                shutil.copy2(segment_path, dest_path)
                final_verified_paths.append(dest_path)
                logger.debug(f"Copied final verified segment: {dest_path.name}")
            except Exception as e:
                logger.error(f"Failed to copy verified segment {segment_path.name} to {output_dir}: {e}")
        else:
            logger.info(f"Discarding segment from different speaker: {segment_path.name} (Speaker: {speaker_id}, Similarity: {similarity:.4f})")

    # --- Save Detailed Results ---
    try:
        # Add the post-filtering decision to the results
        for filename, result_data in verification_results.items():
            if "error" not in result_data:
                original_id = result_data.get("original_speaker_id", "UNKNOWN")
                passed_initial = result_data.get("is_same_speaker", False)
                is_final_speaker = (passed_initial and original_id == most_frequent_speaker_id)
                result_data["is_final_verified_speaker"] = is_final_speaker

        with open(results_output_file, 'w') as f:
            json.dump(verification_results, f, indent=2)
        logger.info(f"Detailed verification results saved to {results_output_file}")
    except Exception as e:
        logger.error(f"Failed to save detailed results file {results_output_file}: {e}")

    if not final_verified_paths:
        logger.warning(f"No segments remained after post-filtering for speaker ID {most_frequent_speaker_id}.")
    else:
        logger.info(f"Finished verification. Copied {len(final_verified_paths)} final segments for speaker {most_frequent_speaker_id} to {output_dir}")

    return final_verified_paths

if __name__ == "__main__":
    # Example usage for standalone testing
    # Ensure these directories exist and contain the necessary files
    reference_dir_test = Path("data/olamide/reference_samples")
    # --- IMPORTANT: Point to a directory with multiple speakers ---
    # Use a directory from a previous step that *hasn't* been verified yet,
    # like the output of diarization/segment extraction (Step 7)
    test_segments_dir_test = Path("data/olamide/processed/The Juice - Olamide/diarized_segments")
    # --- End adjustment ---
    
    final_output_dir_test = Path("data/olamide/processed/The Juice - Olamide/verified_segments_filtered_test")
    results_file_test = Path("data/olamide/speaker_verification_filtered_results_test.json")

    if not test_segments_dir_test.exists():
        print(f"ERROR: Test segments directory not found: {test_segments_dir_test}")
        print("Please ensure the diarized segments exist before running this test.")
    else:
        print(f"Running verification test...")
        print(f"Reference: {reference_dir_test}")
        print(f"Testing:   {test_segments_dir_test}")
        print(f"Output:    {final_output_dir_test}")
        print(f"Results:   {results_file_test}")
        
        # Call the updated function
        final_paths = verify_speaker_segments(
            reference_dir=reference_dir_test,
            diarized_segments_dir=test_segments_dir_test,
            output_dir=final_output_dir_test,
            results_output_file=results_file_test,
            threshold=0.75
        )
        print(f"\nStandalone test finished. Found {len(final_paths)} final verified segments.") 