from pyannote.audio import Pipeline
import torch
import os

# Set the HuggingFace token as an environment variable
# This way it will be used for all HuggingFace requests
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

print("Testing Pyannote model loading...")
print("Using token from environment variable")

try:
    # Force CUDA usage if available
    if torch.cuda.is_available():
        print("CUDA is available, forcing GPU usage")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device("cuda")
    else:
        print("CUDA is not available, using CPU")
        device = torch.device("cpu")
    
    # First try to load the segmentation model separately
    print("Step 1: Attempting to load the segmentation model...")
    from huggingface_hub import hf_hub_download
    
    try:
        hf_hub_download(
            repo_id="pyannote/segmentation-3.0",
            filename="config.yaml",
            use_auth_token=os.environ["HF_TOKEN"]
        )
        print("Successfully accessed segmentation model")
    except Exception as seg_e:
        print(f"Warning: Could not access segmentation model: {str(seg_e)}")
        print("You may need to accept the license at: https://hf.co/pyannote/segmentation-3.0")
    
    # Now try to load the full pipeline
    print(f"\nStep 2: Attempting to load the full diarization pipeline on {device}...")
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.environ["HF_TOKEN"]
    )
    
    # Move model to device
    pipeline.to(device)
    
    print(f"Success! Pyannote model loaded correctly on {device}.")
    print("You can now use the model for speaker diarization.")
    
except Exception as e:
    print(f"Error loading Pyannote model: {str(e)}")
    print("\nPossible solutions:")
    print("1. Make sure you've accepted ALL required licenses:")
    print("   - https://hf.co/pyannote/speaker-diarization-3.1")
    print("   - https://hf.co/pyannote/segmentation-3.0")
    print("   - https://hf.co/pyannote/embedding") 
    print("2. Check that your token has 'read' permissions")
    print("3. Try manually downloading the models from HuggingFace website")

if __name__ == "__main__":
    # This script just tests loading the model
    pass 