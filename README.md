# MusicGPT - Voice Cloning Project

This project aims to create voice cloning models for music artists, starting with Olamide. The system will be able to mimic the artist's voice characteristics, including accent, speaking style, and vocal patterns.

## Project Structure

```
musicGPT/
├── data/
│   └── olamide/
│       ├── raw/          # Original audio files
│       ├── processed/    # Preprocessed audio
│       └── splits/       # Train/val/test splits
├── models/
│   ├── pretrained/      # Base models
│   └── trained/         # Our trained models
├── src/
│   ├── preprocessing/   # Audio preprocessing scripts
│   ├── training/        # Model training scripts
│   └── inference/       # Inference and demo scripts
├── configs/             # Configuration files
├── notebooks/           # Jupyter notebooks for experimentation
└── requirements.txt     # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Collection:
   - Place raw audio files in `data/olamide/raw/`
   - Use preprocessing scripts to clean and prepare data

2. Training:
   - Configure training parameters in `configs/`
   - Run training scripts from `src/training/`

3. Inference:
   - Use inference scripts to generate speech in the artist's voice

## Current Status
- Project initialized
- Basic structure created
- Dependencies defined

## Next Steps
1. Implement audio preprocessing pipeline
2. Set up data collection for Olamide
3. Configure and test base model
4. Begin training process 