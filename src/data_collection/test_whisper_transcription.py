import whisper
from pathlib import Path
import torch # Used to check for GPU
import argparse # Import argparse

def transcribe_audio_files(audio_dir: Path, output_dir: Path, model_name: str = "medium"):
    """
    Transcribes all WAV files in a directory using OpenAI Whisper.

    Args:
        audio_dir: Path to the directory containing WAV files.
        output_dir: Path to the directory where transcription files (.txt) will be saved.
        model_name: Name of the Whisper model to use (e.g., "tiny", "base", "small", "medium", "large").
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu" and model_name not in ["tiny", "base", "small"]:
        print(f"Warning: Using '{model_name}' model on CPU can be very slow. Consider 'base' or 'small'.")

    print(f"Loading Whisper model: {model_name}...")
    try:
        model = whisper.load_model(model_name, device=device)
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading Whisper model '{model_name}': {e}")
        print("Please ensure the model name is correct and Whisper is installed correctly ('pip install -U openai-whisper')")
        return # Stop if model loading fails

    audio_files = list(audio_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files in {audio_dir}")

    if not audio_files:
        print("No audio files found to transcribe.")
        return

    for audio_file in audio_files:
        print(f"\nTranscribing {audio_file.name}...")
        try:
            # Transcribe
            # Use fp16=True only if on CUDA and appropriate GPU
            use_fp16 = device == "cuda"
            # Explicitly set language to English
            result = model.transcribe(str(audio_file), fp16=use_fp16, language='en')

            transcription = result["text"].strip()
            print(f"  Transcription: {transcription}")

            # Save transcription to .txt file
            output_txt_file = output_dir / f"{audio_file.stem}.txt"
            with open(output_txt_file, 'w', encoding='utf-8') as f:
                f.write(transcription)
            print(f"  Saved transcription to: {output_txt_file}")

        except Exception as e:
            print(f"Error transcribing {audio_file.name}: {e}")

if __name__ == "__main__":
    # Use argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Transcribe audio files in a directory using Whisper.")
    parser.add_argument("--input_dir", required=True, help="Directory containing the WAV files to transcribe.")
    parser.add_argument("--output_dir", required=True, help="Directory where transcription TXT files will be saved.")
    parser.add_argument("--model", default="medium", 
                        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large"],
                        help="Name of the Whisper model to use.")
    
    args = parser.parse_args()

    input_audio_directory = Path(args.input_dir)
    output_transcription_directory = Path(args.output_dir)
    whisper_model = args.model

    # --- Configuration --- (REMOVED HARDCODED PATHS)
    # # Directory containing the verified audio segments from the previous step
    # input_audio_directory = Path("data/olamide/processed/The Juice - Olamide/olamide_segments/verified_segments")
    # 
    # # Directory where the transcription .txt files will be saved
    # output_transcription_directory = input_audio_directory.parent / "transcriptions" # Save in a sibling folder
    # 
    # # Choose Whisper model size (e.g., "tiny", "base", "small", "medium", "large")
    # # Larger models are generally more accurate but require more resources (GPU recommended for medium+)
    # whisper_model = "medium"
    # --- End Configuration ---

    if not input_audio_directory.exists():
        print(f"Error: Input directory not found: {input_audio_directory}")
        # print("Please ensure the agent has run successfully and produced verified segments.")
    else:
        transcribe_audio_files(input_audio_directory, output_transcription_directory, whisper_model)
        print("\nTranscription process finished.") 