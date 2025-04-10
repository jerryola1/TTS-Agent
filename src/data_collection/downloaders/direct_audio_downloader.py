import logging
from pathlib import Path
from typing import Dict, Optional
import requests
import re
from urllib.parse import urlparse

class DirectAudioDownloader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_extensions = {'.mp3', '.wav', '.ogg', '.flac', '.m4a'}
    
    def download(self, url: str, download_dir: Path, metadata: Optional[Dict] = None) -> Optional[Path]:
        """
        Download audio file directly from URL.
        
        Args:
            url: Direct URL to audio file
            download_dir: Directory to save the downloaded file
            metadata: Optional metadata about the content
            
        Returns:
            Path to the downloaded file if successful, None otherwise
        """
        try:
            # Get filename from URL or metadata
            filename = self._get_filename(url, metadata)
            if not filename:
                self.logger.error("Could not determine filename")
                return None
            
            output_path = download_dir / filename
            
            # Check if file already exists
            if output_path.exists():
                self.logger.info(f"File already exists: {output_path}")
                return output_path
            
            # Download the file
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                self.logger.error(f"Failed to download file: {response.status_code}")
                return None
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(ext in content_type for ext in ['audio', 'octet-stream']):
                self.logger.warning(f"Content type '{content_type}' may not be audio")
            
            # Save the file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"Successfully downloaded: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error downloading file {url}: {str(e)}")
            return None
    
    def _get_filename(self, url: str, metadata: Optional[Dict] = None) -> Optional[str]:
        """Get a valid filename from URL or metadata."""
        # Try to get filename from metadata first
        if metadata and 'title' in metadata:
            filename = self._sanitize_filename(metadata['title'])
            # Add extension if not present
            if not any(filename.endswith(ext) for ext in self.supported_extensions):
                filename += '.mp3'  # Default to mp3
            return filename
        
        # Try to get filename from URL
        parsed_url = urlparse(url)
        path = parsed_url.path
        if path:
            filename = path.split('/')[-1]
            # Check if filename has a supported extension
            if any(filename.endswith(ext) for ext in self.supported_extensions):
                return self._sanitize_filename(filename)
        
        # If no valid filename found, generate one
        return self._generate_filename()
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to be safe for filesystem."""
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        # Limit length
        return filename[:100]
    
    def _generate_filename(self) -> str:
        """Generate a unique filename."""
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"audio_{timestamp}.mp3" 