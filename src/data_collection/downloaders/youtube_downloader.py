import logging
from pathlib import Path
from typing import Dict, Optional
import yt_dlp

class YouTubeDownloader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def download(self, url: str, download_dir: Path, metadata: Optional[Dict] = None) -> Optional[Path]:
        """
        Download audio from YouTube URL.
        
        Args:
            url: YouTube URL
            download_dir: Directory to save the downloaded file
            metadata: Optional metadata about the content
            
        Returns:
            Path to the downloaded file if successful, None otherwise
        """
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': str(download_dir / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_audio': True,
            'audio_format': 'mp3',
            'audio_quality': '0',  # best quality
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info
                info = ydl.extract_info(url, download=False)
                
                # Download the audio
                ydl.download([url])
                
                # Get the actual filename
                filename = ydl.prepare_filename(info)
                audio_file = Path(filename).with_suffix('.mp3')
                
                if not audio_file.exists():
                    self.logger.error(f"Downloaded file not found: {audio_file}")
                    return None
                
                self.logger.info(f"Successfully downloaded: {audio_file}")
                return audio_file
                
        except Exception as e:
            self.logger.error(f"Error downloading YouTube video {url}: {str(e)}")
            return None 