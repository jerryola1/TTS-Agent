import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union
from datetime import datetime
from urllib.parse import urlparse

from .youtube_downloader import YouTubeDownloader
from .soundcloud_downloader import SoundCloudDownloader
from .direct_audio_downloader import DirectAudioDownloader

class DownloadHandler:
    def __init__(self, base_dir: Union[str, Path] = "data", artist_name: Optional[str] = None):
        """
        Initialize the download handler with base directory and optional artist name.
        
        Args:
            base_dir: Base directory for all downloads
            artist_name: Optional artist name to organize downloads
        """
        # --- Set up logger FIRST --- 
        self.logger = logging.getLogger(__name__)
        # --- Logger Setup Done --- 
        
        self.base_dir = Path(base_dir)
        self.artist_name = artist_name
        
        # Set up artist-specific directories
        self.artist_dir = self.base_dir / (artist_name.lower() if artist_name else "unknown_artist")
        self.download_dir = self.artist_dir / "downloads"
        self.db_path = self.artist_dir / "downloads.json"
        
        # Create necessary directories
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database (can now safely log errors if needed)
        self.db = self._load_database()
        
        # Initialize downloaders
        self.downloaders = {
            'youtube': YouTubeDownloader(),
            'soundcloud': SoundCloudDownloader(),
            'direct': DirectAudioDownloader(),
        }
        
        # Set up logging (MOVED TO TOP)
        # self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized download handler for artist: {artist_name or 'unknown'}")
        self.logger.info(f"Download directory: {self.download_dir}")
    
    def download(self, url: str, metadata: Optional[Dict] = None) -> Optional[Path]:
        """
        Download content from the given URL.
        
        Args:
            url: The URL to download from
            metadata: Optional metadata about the content
            
        Returns:
            Path to the downloaded file if successful, None otherwise
        """
        # Check if URL is already processed
        if self._is_processed(url):
            existing_file = self._get_existing_file_path(url)
            if existing_file:
                return existing_file
            # If we get here, the file was missing and was removed from the database
            # We'll continue to download it below
        
        # Determine source type
        source_type = self._detect_source(url)
        if source_type not in self.downloaders:
            self.logger.error(f"Unsupported source type for URL: {url}")
            return None
        
        # Extract or update metadata
        metadata = self._extract_metadata(url, source_type, metadata)
        
        # Download using the appropriate downloader
        downloader = self.downloaders[source_type]
        try:
            file_path = downloader.download(url, self.download_dir, metadata)
            if file_path:
                self._update_database(url, file_path, metadata)
                return file_path
        except Exception as e:
            self.logger.error(f"Error downloading {url}: {str(e)}")
            return None
    
    def _extract_metadata(self, url: str, source_type: str, metadata: Optional[Dict] = None) -> Dict:
        """Extract metadata from the URL or existing metadata."""
        if metadata is None:
            metadata = {}
        
        # Try to extract artist name if not provided
        if not self.artist_name and 'artist' in metadata:
            self.artist_name = metadata['artist']
            self.artist_dir = self.base_dir / self.artist_name.lower()
            self.download_dir = self.artist_dir / "downloads"
            self.download_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Updated artist directory to: {self.artist_dir}")
        
        # Add source information
        metadata['source_type'] = source_type
        metadata['source_url'] = url
        metadata['download_date'] = datetime.now().isoformat()
        
        return metadata
    
    def _detect_source(self, url: str) -> str:
        """Detect the source type from the URL."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        if 'youtube.com' in domain or 'youtu.be' in domain:
            return 'youtube'
        elif 'soundcloud.com' in domain:
            return 'soundcloud'
        # Default to the 'youtube' downloader (which uses yt-dlp)
        # as yt-dlp supports many sites beyond YouTube, including TikTok.
        self.logger.info(f"Unknown domain '{domain}', attempting download with yt-dlp (via 'youtube' downloader key).")
        return 'youtube' # Changed default from 'direct'
    
    def _is_processed(self, url: str) -> bool:
        """Check if the URL has already been processed."""
        return url in self.db.get('processed_urls', {})
    
    def _get_existing_file_path(self, url: str) -> Optional[Path]:
        """Get the path of an already processed file."""
        if url in self.db.get('processed_urls', {}):
            file_path = Path(self.db['processed_urls'][url]['file_path'])
            if file_path.exists():
                return file_path
            else:
                # File is missing, remove from database
                self.logger.warning(f"File {file_path} is missing from disk. Removing from database.")
                del self.db['processed_urls'][url]
                self._save_database()
        return None
    
    def _update_database(self, url: str, file_path: Path, metadata: Dict):
        """Update the database with new download information."""
        if 'processed_urls' not in self.db:
            self.db['processed_urls'] = {}
        
        self.db['processed_urls'][url] = {
            'file_path': str(file_path),
            'metadata': metadata,
            'download_date': datetime.now().isoformat()
        }
        self._save_database()
    
    def _load_database(self) -> Dict:
        """Load the database from file."""
        # Ensure logger exists before attempting to load
        if not hasattr(self, 'logger') or self.logger is None:
             # Fallback if somehow logger wasn't created (shouldn't happen now)
             print("WARNING: Logger not initialized in DownloadHandler before _load_database")
             self.logger = logging.getLogger(__name__)
             
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                # This warning can now safely use self.logger
                self.logger.warning("Database file corrupted, creating new one") 
        return {'processed_urls': {}}
    
    def _save_database(self):
        """Save the database to file."""
        with open(self.db_path, 'w') as f:
            json.dump(self.db, f, indent=4) 