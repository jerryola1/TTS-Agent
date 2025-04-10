import os
from pathlib import Path

def extract_audio_from_videos():
    """Extract MP3 audio from video files in the raw directory."""
    raw_dir = Path("data/olamide/raw")
    
    # Videos to process
    videos = [
        "EXCLUSIVE INTERVIEW WITH OLAMIDE (Nigerian Entertainment News).mp4",
        "EXCLUSIVES! Special - Olamide & Tony Elumelu (Oil & Gas).mp4",
        "Olamide： 'I May Not Marry My BabyMama, Never Met DaGrin' ｜ The TakeOver with Moet Abebe.mp4"
    ]
    
    for video in videos:
        video_path = raw_dir / video
        if not video_path.exists():
            print(f"Video not found: {video}")
            continue
            
        # Create output filename by replacing .mp4 with .mp3
        audio_path = raw_dir / (video[:-4] + ".mp3")
        
        if audio_path.exists():
            print(f"Audio file already exists: {audio_path}")
            continue
            
        print(f"\nExtracting audio from: {video}")
        cmd = f'ffmpeg -i "{video_path}" -vn -acodec libmp3lame -q:a 2 "{audio_path}"'
        
        try:
            result = os.system(cmd)
            if result == 0:
                print(f"Successfully extracted audio to: {audio_path}")
            else:
                print(f"Failed to extract audio from: {video}")
        except Exception as e:
            print(f"Error processing {video}: {str(e)}")

if __name__ == "__main__":
    extract_audio_from_videos() 