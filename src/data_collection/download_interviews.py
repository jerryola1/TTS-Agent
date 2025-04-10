import os
import json
from pathlib import Path
import time
from data_fetcher import DataFetcher
from find_sources import SourceFinder

def download_selected_interviews():
    """Download selected high-quality Olamide interviews."""
    # Initialize our tools
    fetcher = DataFetcher()
    finder = SourceFinder()
    
    # Get all available sources
    sources = finder.get_sources()
    
    # List of selected high-quality interviews (based on duration and quality)
    selected_interviews = [
        {
            "title": "The Juice - Olamide",
            "url": "https://www.youtube.com/watch?v=rCUaMC0Ll_8",
            "duration": "PT16M16S"
        },
        {
            "title": "EXCLUSIVES! Special - Olamide & Tony Elumelu",
            "url": "https://www.youtube.com/watch?v=U6TzuIinuBY",
            "duration": "PT14M31S"
        },
        {
            "title": "Olamide: 'I May Not Marry My BabyMama, Never Met DaGrin'",
            "url": "https://www.youtube.com/watch?v=fuSMPgdwAhw",
            "duration": "PT13M2S"
        },
        {
            "title": "EXCLUSIVE INTERVIEW WITH OLAMIDE",
            "url": "https://www.youtube.com/watch?v=lrUQowZxHrw",
            "duration": "PT12M49S"
        },
        {
            "title": "ONE on ONE with Olamide",
            "url": "https://www.youtube.com/watch?v=7U66cWPPUcY",
            "duration": "PT9M45S"
        }
    ]
    
    print(f"Starting download of {len(selected_interviews)} selected interviews...")
    
    for interview in selected_interviews:
        print(f"\nProcessing: {interview['title']}")
        
        # Check if we already have this interview
        existing_sources = fetcher.get_available_sources()
        if any(source['url'] == interview['url'] for source in existing_sources):
            print(f"Interview already downloaded: {interview['title']}")
            continue
        
        # Download both video and audio
        success = fetcher.download_youtube_content(
            url=interview['url'],
            description=f"High-quality Olamide interview: {interview['title']} (Duration: {interview['duration']})",
            download_video=True  # We'll get both video and audio for better options later
        )
        
        if success:
            print(f"Successfully downloaded: {interview['title']}")
        else:
            print(f"Failed to download: {interview['title']}")
        
        # Add a small delay between downloads
        time.sleep(2)
    
    print("\nDownload process completed!")
    print("You can find the downloaded files in the raw data directory.")

if __name__ == "__main__":
    download_selected_interviews() 