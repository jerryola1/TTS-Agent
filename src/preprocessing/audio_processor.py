import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

class AudioProcessor:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sample_rate = self.config['audio']['target_sample_rate']
        self.trim_silence = self.config['audio']['trim_silence']
        self.normalize = self.config['audio']['normalize']
        self.max_noise_level = self.config['audio']['max_noise_level']
        
    def load_audio(self, file_path):
        """Load audio file and convert to target sample rate."""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None, None

    def trim_audio(self, audio):
        """Trim silence from audio."""
        if not self.trim_silence:
            return audio
            
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=30)
        return trimmed_audio

    def normalize_audio(self, audio):
        """Normalize audio to [-1, 1] range."""
        if not self.normalize:
            return audio
            
        return librosa.util.normalize(audio)

    def process_file(self, input_path, output_path):
        """Process a single audio file."""
        audio, sr = self.load_audio(input_path)
        if audio is None:
            return False
            
        # Apply processing steps
        audio = self.trim_audio(audio)
        audio = self.normalize_audio(audio)
        
        # Save processed audio
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, sr)
        return True

    def process_directory(self, input_dir, output_dir):
        """Process all audio files in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        audio_files = list(input_dir.glob('*.wav')) + list(input_dir.glob('*.mp3'))
        success_count = 0
        
        for file_path in tqdm(audio_files, desc="Processing audio files"):
            output_path = output_dir / file_path.name
            if self.process_file(str(file_path), str(output_path)):
                success_count += 1
                
        print(f"Processed {success_count}/{len(audio_files)} files successfully")

if __name__ == "__main__":
    processor = AudioProcessor()
    processor.process_directory(
        "data/olamide/raw",
        "data/olamide/processed"
    ) 