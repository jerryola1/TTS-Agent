import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import json
from sklearn.cluster import KMeans
import shutil
import logging
from datetime import datetime

class InterviewProcessor:
    def __init__(self, base_dir="data/olamide"):
        """Initialize the interview processor with base directory structure."""
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.analysis_dir = self.base_dir / "analysis"
        
        # Create necessary directories
        for dir_path in [self.processed_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def process_interview(self, interview_file, min_duration=2.0, max_gap=1.0):
        """Process a single interview through the entire pipeline."""
        self.logger.info(f"Starting processing for: {interview_file}")
        
        # Create interview-specific directories
        interview_name = Path(interview_file).stem
        interview_dir = self.processed_dir / interview_name
        interview_dir.mkdir(exist_ok=True)
        
        try:
            # Step 1: Extract speech segments
            segments = self._extract_speech_segments(
                interview_file, 
                interview_dir,
                min_duration,
                max_gap
            )
            
            # Step 2: Separate speakers
            olamide_segments = self._separate_speakers(interview_dir, segments)
            
            # Step 3: Save final results
            self._save_results(interview_dir, olamide_segments)
            
            self.logger.info(f"Successfully processed {interview_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {interview_file}: {str(e)}")
            return False
    
    def _extract_speech_segments(self, interview_file, output_dir, min_duration, max_gap):
        """Extract speech segments from the interview."""
        self.logger.info("Extracting speech segments...")
        
        # Load audio
        audio_path = self.raw_dir / interview_file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Create segments directory
        segments_dir = output_dir / "segments"
        segments_dir.mkdir(exist_ok=True)
        
        # Extract segments (simplified for example - you would add your full logic here)
        segments = []
        # Add your segment extraction logic here
        
        self.logger.info(f"Extracted {len(segments)} segments")
        return segments
    
    def _separate_speakers(self, interview_dir, segments):
        """Separate speakers using voice features."""
        self.logger.info("Separating speakers...")
        
        segments_dir = interview_dir / "segments"
        olamide_dir = interview_dir / "olamide_segments"
        olamide_dir.mkdir(exist_ok=True)
        
        # Extract features and cluster speakers
        features = []
        wav_files = list(segments_dir.glob("*.wav"))
        
        for wav_file in wav_files:
            feat = self._extract_voice_features(wav_file)
            features.append(feat)
        
        # Cluster speakers
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(np.array(features))
        
        # Identify Olamide's cluster
        olamide_segments = []
        # Add your speaker identification logic here
        
        self.logger.info(f"Identified {len(olamide_segments)} segments as Olamide's voice")
        return olamide_segments
    
    def _extract_voice_features(self, audio_path):
        """Extract voice features from an audio segment."""
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        
        # Add other feature extraction logic here
        
        return mfcc_means
    
    def _save_results(self, interview_dir, olamide_segments):
        """Save processing results and metadata."""
        results = {
            'processed_date': datetime.now().isoformat(),
            'olamide_segments': olamide_segments,
            'total_segments': len(olamide_segments)
        }
        
        results_file = interview_dir / "processing_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {results_file}")

def main():
    # Initialize processor
    processor = InterviewProcessor()
    
    # Get list of interviews to process
    interviews = [
        "EXCLUSIVE INTERVIEW WITH OLAMIDE (Nigerian Entertainment News).mp3",
        "Olamide： 'I May Not Marry My BabyMama, Never Met DaGrin' ｜ The TakeOver with Moet Abebe.mp3",
        "ONE on ONE with Olamide.mp3",
        "The Juice - Olamide.mp3"
    ]
    
    # Process each interview
    for interview in interviews:
        processor.process_interview(interview)

if __name__ == "__main__":
    main() 