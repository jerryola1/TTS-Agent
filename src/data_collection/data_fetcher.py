import os
import yaml
from pathlib import Path
import json
import yt_dlp
import re
from tqdm import tqdm

class DataFetcher:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_dir = Path(self.config['data']['raw_data_dir'])
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file to track sources
        self.metadata_file = self.raw_data_dir / "metadata.json"
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w') as f:
                json.dump({"sources": []}, f)
    
    def save_metadata(self, source_url, filename, description, content_type="audio"):
        """Save metadata about downloaded files."""
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        metadata["sources"].append({
            "url": source_url,
            "filename": filename,
            "description": description,
            "content_type": content_type,
            "date_downloaded": str(Path(filename).stat().st_mtime)
        })
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def download_youtube_content(self, url, description="", download_video=True):
        """Download content from YouTube video using yt-dlp."""
        try:
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' if download_video else 'bestaudio[ext=m4a]',
                'outtmpl': str(self.raw_data_dir / '%(title)s.%(ext)s'),
                'quiet': False,
                'no_warnings': True,
                'progress': True
            }
            
            # Download the content
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first
                info = ydl.extract_info(url, download=False)
                title = info['title']
                
                # Clean filename
                clean_title = self.clean_filename(title)
                
                if download_video:
                    print(f"Downloading video: {title}")
                    video_path = self.raw_data_dir / f"{clean_title}.mp4"
                    ydl.download([url])
                    self.save_metadata(url, str(video_path), description, "video")
                
                # Download audio separately if video wasn't downloaded or if we want both
                if not download_video or download_video:
                    print(f"Downloading audio: {title}")
                    audio_opts = ydl_opts.copy()
                    audio_opts['format'] = 'bestaudio[ext=m4a]'
                    audio_opts['postprocessors'] = [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                    }]
                    
                    with yt_dlp.YoutubeDL(audio_opts) as ydl_audio:
                        ydl_audio.download([url])
                        audio_path = self.raw_data_dir / f"{clean_title}.mp3"
                        self.save_metadata(url, str(audio_path), description, "audio")
            
            return True
            
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return False
    
    def get_available_sources(self):
        """List all downloaded sources."""
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        return metadata["sources"]

    def clean_filename(self, filename: str) -> str:
        """Clean filename to be filesystem safe."""
        # Remove any URL parameters
        filename = filename.split('?')[0]
        
        # Replace special characters with underscores
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Replace multiple spaces with single space
        filename = re.sub(r'\s+', ' ', filename)
        
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        
        # Remove any remaining special characters
        filename = re.sub(r'[^\w\-_.]', '', filename)
        
        # Ensure filename isn't too long
        if len(filename) > 100:
            filename = filename[:100]
        
        return filename

if __name__ == "__main__":
    fetcher = DataFetcher()
    
    # Example usage:
    # fetcher.download_youtube_content(
    #     "https://www.youtube.com/watch?v=example",
    #     "Olamide interview about music career",
    #     download_video=True
    # ) 