import logging
import argparse
from pathlib import Path
from downloaders import DownloadHandler

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_downloader(artist_name: str):
    """
    Test the downloader functionality with different types of URLs.
    
    Args:
        artist_name: Name of the artist to test with
    """
    # Initialize the download handler for the specified artist
    downloader = DownloadHandler(
        base_dir="data",
        artist_name=artist_name
    )
    
    # Test URLs with metadata
    test_urls = [
        {
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # YouTube test URL
            "type": "youtube",
            "metadata": {
                "title": "Test YouTube Video",
                "artist": artist_name,
                "description": "Test video description"
            }
        },
        {
            "url": "https://soundcloud.com/example-track",  # SoundCloud test URL
            "type": "soundcloud",
            "metadata": {
                "title": "Test SoundCloud Track",
                "artist": artist_name,
                "description": "Test track description"
            }
        },
        {
            "url": "https://example.com/audio.mp3",  # Direct audio test URL
            "type": "direct",
            "metadata": {
                "title": "Test Direct Audio",
                "artist": artist_name,
                "description": "Test audio description"
            }
        }
    ]
    
    # Test each URL
    for test in test_urls:
        print(f"\nTesting {test['type']} downloader with URL: {test['url']}")
        try:
            file_path = downloader.download(test['url'], test['metadata'])
            if file_path:
                print(f"✓ Successfully downloaded to: {file_path}")
                print(f"File exists: {file_path.exists()}")
                print(f"File size: {file_path.stat().st_size / 1024:.2f} KB")
                
                # Verify the file is in the correct artist directory
                assert artist_name.lower() in str(file_path).lower(), "File not in artist directory"
                print(f"✓ File is in correct artist directory ({artist_name})")
            else:
                print("✗ Download failed")
        except Exception as e:
            print(f"✗ Error during download: {str(e)}")
    
    # Test database functionality
    print(f"\nTesting database functionality for {artist_name}:")
    db_path = Path(f"data/{artist_name.lower()}/downloads.json")
    if db_path.exists():
        print(f"✓ Database file exists: {db_path}")
        print(f"Database size: {db_path.stat().st_size / 1024:.2f} KB")
    else:
        print("✗ Database file not found")

def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description='Test the downloader functionality')
    parser.add_argument(
        '--artist',
        type=str,
        required=True,
        help='Name of the artist to test with'
    )
    args = parser.parse_args()
    
    setup_logging()
    test_downloader(args.artist)

if __name__ == "__main__":
    main() 