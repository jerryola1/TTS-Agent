import shutil
from pathlib import Path

def organize_segments():
    """Organize speech segments into folders based on their source interview."""
    segments_dir = Path("data/olamide/speech_segments")
    organized_dir = Path("data/olamide/organized_segments")
    
    # Create base directory
    organized_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all WAV files and JSON files
    wav_files = list(segments_dir.glob("*.wav"))
    json_files = list(segments_dir.glob("*.json"))
    
    # Create interview-specific folders and move files
    for wav_file in wav_files:
        # Extract interview name from segment filename
        interview_name = wav_file.name.split("_segment_")[0]
        interview_dir = organized_dir / interview_name
        
        # Create interview directory
        interview_dir.mkdir(exist_ok=True)
        
        # Move WAV file
        shutil.copy2(wav_file, interview_dir / wav_file.name)
    
    # Move corresponding JSON files
    for json_file in json_files:
        interview_name = json_file.stem.replace("_segments", "")
        interview_dir = organized_dir / interview_name
        
        if interview_dir.exists():
            shutil.copy2(json_file, interview_dir / json_file.name)
    
    print("\nOrganized segments by interview:")
    for interview_dir in organized_dir.iterdir():
        if interview_dir.is_dir():
            wav_count = len(list(interview_dir.glob("*.wav")))
            print(f"- {interview_dir.name}: {wav_count} segments")

if __name__ == "__main__":
    organize_segments() 