import os
import json
from pathlib import Path
from googleapiclient.discovery import build
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class SourceFinder:
    def __init__(self):
        self.sources_file = Path("data/potential_sources.json")
        self.sources_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.sources_file.exists():
            self.save_sources([])
    
    def search_youtube_interviews(self):
        """Search for Olamide interviews on YouTube using the YouTube Data API."""
        print("Searching for Olamide interviews on YouTube...")
        
        # Note: You'll need to set your API key in an environment variable
        api_key = os.getenv('YOUTUBE_API_KEY')
        if not api_key:
            print("Please set your YouTube API key in the YOUTUBE_API_KEY environment variable")
            return []
        
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        search_queries = [
            "Olamide interview",
            "Olamide speaking",
            "Olamide podcast",
            "Olamide talk show",
            "Olamide radio interview"
        ]
        
        found_interviews = []
        
        for query in search_queries:
            print(f"\nSearching for: {query}")
            
            try:
                request = youtube.search().list(
                    part="snippet",
                    q=query,
                    type="video",
                    maxResults=10,
                    relevanceLanguage="en"
                )
                response = request.execute()
                
                for item in response['items']:
                    video_id = item['id']['videoId']
                    
                    # Get video details
                    video_request = youtube.videos().list(
                        part="snippet,contentDetails,statistics",
                        id=video_id
                    )
                    video_response = video_request.execute()
                    
                    if video_response['items']:
                        video = video_response['items'][0]
                        
                        # Skip if it's a music video
                        if "Music" in video['snippet'].get('categoryId', ''):
                            continue
                        
                        interview = {
                            "title": video['snippet']['title'],
                            "url": f"https://www.youtube.com/watch?v={video_id}",
                            "description": video['snippet']['description'][:200] + "...",
                            "duration": video['contentDetails']['duration'],
                            "views": video['statistics'].get('viewCount', 0),
                            "upload_date": video['snippet']['publishedAt'],
                            "type": "interview",
                            "platform": "YouTube"
                        }
                        
                        # Check if interview is already in the list
                        if not any(i['url'] == interview['url'] for i in found_interviews):
                            found_interviews.append(interview)
                            print(f"Found: {interview['title']}")
            
            except Exception as e:
                print(f"Error searching for '{query}': {str(e)}")
                continue
        
        return found_interviews
    
    def save_sources(self, sources):
        """Save the list of sources to the JSON file."""
        with open(self.sources_file, 'w', encoding='utf-8') as f:
            json.dump(sources, f, indent=2, ensure_ascii=False)
    
    def get_sources(self):
        """Get the list of sources from the JSON file."""
        if not self.sources_file.exists():
            return []
        
        with open(self.sources_file, 'r', encoding='utf-8') as f:
            return json.load(f)

if __name__ == "__main__":
    finder = SourceFinder()
    
    # Search for new interviews
    interviews = finder.search_youtube_interviews()
    
    if interviews:
        # Get existing sources and add new ones
        existing_sources = finder.get_sources()
        new_sources = []
        
        for interview in interviews:
            if not any(s['url'] == interview['url'] for s in existing_sources):
                new_sources.append(interview)
        
        if new_sources:
            print(f"\nFound {len(new_sources)} new sources!")
            all_sources = existing_sources + new_sources
            finder.save_sources(all_sources)
            print(f"Total sources now: {len(all_sources)}")
        else:
            print("\nNo new sources found.")
    else:
        print("\nNo interviews found or there was an error with the YouTube API.") 