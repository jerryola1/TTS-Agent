import argparse
import logging
import os
from pathlib import Path
import torch
from datetime import datetime

# Rich logging
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

# TTS library imports (ensure TTS is installed: pip install TTS)
try:
    from TTS.config import BaseAudioConfig, BaseDatasetConfig
    from TTS.trainer import Trainer, TrainerArgs
    from TTS.tts.configs.shared_configs import BaseTTSConfig
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.datasets import BaseDataset
    from TTS.tts.models.xtts import Xtts
    from TTS.utils.manage import ModelManager
except ImportError as e:
    print(f"Error importing TTS library: {e}")
    print("Please ensure you have installed it: pip install TTS")
    exit(1)

# --- Rich Logging Setup ---
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)

# --- Main Training Function ---

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune an XTTS model using Coqui TTS library."
    )

    # --- Dataset Arguments ---
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to the directory containing the metadata file (e.g., metadata.csv) and audio files.",
    )
    parser.add_argument(
        "--metadata_file", type=str, default="metadata.csv",
        help="Name of the metadata file within the dataset_path.",
    )

    # --- Model Arguments ---
    parser.add_argument(
        "--model_name", type=str, default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Name or path to the pre-trained XTTS model to fine-tune.",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Directory to save checkpoints, logs, and the final model.",
    )

    # --- Training Arguments ---
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Training batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=5e-6, help="Initial learning rate."
    )
    parser.add_argument(
        "--grad_accum", type=int, default=4,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of DataLoader workers."
    )
    parser.add_argument(
        "--seed", type=int, default=54321, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--save_step", type=int, default=1000, help="Save checkpoint every N steps."
    )
    parser.add_argument(
        "--log_step", type=int, default=100, help="Log training progress every N steps."
    )
    parser.add_argument(
        "--use_gpu", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable GPU usage."
    )

    args = parser.parse_args()

    # --- Prepare Paths and Directory ---
    dataset_path = Path(args.dataset_path)
    metadata_path = dataset_path / args.metadata_file
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"[bold green]Starting XTTS Fine-tuning Process[/bold green]")
    logger.info(f"Dataset Path: [cyan]{dataset_path}[/cyan]")
    logger.info(f"Metadata File: [cyan]{metadata_path}[/cyan]")
    logger.info(f"Output Path: [cyan]{output_path}[/cyan]")
    logger.info(f"Pre-trained Model: [cyan]{args.model_name}[/cyan]")

    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return

    # --- Configuration Setup ---
    logger.info("Setting up configurations...")

    # Audio config (usually okay to keep defaults from the model)
    audio_config = BaseAudioConfig(
        sample_rate=24000, # XTTS default, do not change unless your data is different and you know why
        # Other params usually loaded from model checkpoint
    )

    # Dataset config
    # Using LJSpeech format (wav_path|transcript) as created by prepare_tts_data.py
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train=args.metadata_file, path=str(dataset_path)
    )

    # Model config (Load from pre-trained)
    # We need ModelManager to download/find the model first
    manager = ModelManager()
    model_path, config_path, _ = manager.download_model(args.model_name)

    if config_path is None or not Path(config_path).exists():
         logger.error(f"Could not find config file for model {args.model_name}. Check the model name.")
         # Attempt to load default if path is missing - might fail
         config = XttsConfig()
         # Manually set sample rate as BaseTTSConfig might not have it directly
         config.audio = audio_config
         logger.warning("Could not load model config, using default XttsConfig.")
    else:
        config = XttsConfig()
        config.load_json(config_path)
        config.audio = audio_config # Ensure audio config is updated

    # Set fine-tuning specific parameters
    config.batch_size = args.batch_size
    config.eval_batch_size = max(1, args.batch_size // 2) # Usually smaller for eval
    config.num_loader_workers = args.num_workers
    config.num_eval_loader_workers = args.num_workers
    config.output_path = str(output_path)
    config.epochs = args.epochs
    config.save_step = args.save_step
    config.log_step = args.log_step
    config.save_checkpoints = True
    config.save_all_best = True # Save best model based on eval loss
    config.print_step = 50 # How often to print step progress
    config.plot_step = 100 # How often to plot attention/specs
    config.seed = args.seed
    config.gradient_accumulation_steps = args.grad_accum

    # Set learning rate and optimizer details (can be tuned)
    config.optimizer = "AdamW"
    config.optimizer_params = {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-4}
    config.lr = args.lr
    # Scheduler options can be added here if needed

    # --- Initialize Model ---
    logger.info("Initializing XTTS model...")
    model = Xtts.init_from_config(config)

    # Load pre-trained weights
    if model_path is None or not Path(model_path).exists():
         logger.error(f"Could not find model checkpoint file for {args.model_name}. Check the model name or path.")
         logger.warning("Proceeding with randomly initialized weights - this is NOT fine-tuning!")
    else:
        logger.info(f"Loading weights from: [cyan]{model_path}[/cyan]")
        model.load_checkpoint(config, checkpoint_path=model_path, eval=False, strict=False) # Use strict=False for potential mismatches

    # --- Initialize Trainer ---
    logger.info("Initializing Trainer...")
    trainer_args = TrainerArgs(
         restore_path=None, # Set to a checkpoint path to resume training
         # skip_train_epoch=False # Set to True if resuming and wanting to skip current epoch
         # --- Other TrainerArgs can be set here if needed ---
    )

    # Progress bar setup for Trainer
    # Not directly customizing Trainer's internal bar here, but Rich logger will capture its output
    trainer = Trainer(
        trainer_args,
        config,
        output_path=str(output_path),
        model=model,
        train_samples=None, # Trainer will use the dataset config
        eval_samples=None,  # No eval split defined in this basic setup
    )

    # --- Start Training ---
    logger.info(f"[bold green]Starting Training...[/bold green]")
    try:
        trainer.fit()
        logger.info(f"[bold green]Training Finished Successfully![/bold green]")
    except Exception as e:
        logger.exception(f"[bold red]An error occurred during training:[/bold red] {e}")


if __name__ == "__main__":
    main() 