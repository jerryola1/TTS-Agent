import logging
from pathlib import Path
from typing import Dict, Optional
import requests
import re

class SoundCloudDownloader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client_id = self._get_client_id()
    
    def _get_client_id(self) -> str:
        """Get a valid SoundCloud client ID."""
        # This is a placeholder. In production, you should:
        # 1. Store the client ID securely (e.g., in environment variables)
        # 2. Implement a proper client ID rotation mechanism
        return "YOUR_SOUNDCLOUD_CLIENT_ID"
    
    def download(self, url: str, download_dir: Path, metadata: Optional[Dict] = None) -> Optional[Path]:
        """
        Download audio from SoundCloud URL.
        
        Args:
            url: SoundCloud URL
            download_dir: Directory to save the downloaded file
            metadata: Optional metadata about the content
            
        Returns:
            Path to the downloaded file if successful, None otherwise
        """
        try:
            # First, get the track info
            track_info = self._get_track_info(url)
            if not track_info:
                return None
            
            # Get the stream URL
            stream_url = self._get_stream_url(track_info)
            if not stream_url:
                return None
            
            # Download the audio
            filename = self._sanitize_filename(track_info['title'])
            output_path = download_dir / f"{filename}.mp3"
            
            response = requests.get(stream_url, stream=True)
            if response.status_code != 200:
                self.logger.error(f"Failed to download from SoundCloud: {response.status_code}")
                return None
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"Successfully downloaded: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error downloading SoundCloud track {url}: {str(e)}")
            return None
    
    def _get_track_info(self, url: str) -> Optional[Dict]:
        """Get track information from SoundCloud API."""
        try:
            # This is a simplified version. In production, you should:
            # 1. Use proper SoundCloud API client
            # 2. Handle rate limiting
            # 3. Implement proper error handling
            api_url = f"https://api.soundcloud.com/resolve?url={url}&client_id={self.client_id}"
            response = requests.get(api_url)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            self.logger.error(f"Error getting track info: {str(e)}")
            return None
    
    def _get_stream_url(self, track_info: Dict) -> Optional[str]:
        """Get the stream URL from track info."""
        try:
            # This is a simplified version. In production, you should:
            # 1. Handle different track types (private, premium, etc.)
            # 2. Implement proper error handling
            stream_url = track_info.get('stream_url')
            if stream_url:
                return f"{stream_url}?client_id={self.client_id}"
            return None
        except Exception as e:
            self.logger.error(f"Error getting stream URL: {str(e)}")
            return None
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to be safe for filesystem."""
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        # Limit length
        return filename[:100] 