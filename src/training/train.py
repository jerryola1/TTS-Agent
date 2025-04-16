import argparse
import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.serialization
from datetime import datetime

# Rich logging
from rich.logging import RichHandler
from rich.console import Console

# TTS library imports (ensure TTS is installed: pip install TTS)
try:
    from TTS.config import BaseAudioConfig, BaseDatasetConfig
    # Removed Trainer import
    from TTS.tts.configs.shared_configs import BaseTTSConfig
    from TTS.tts.configs.xtts_config import XttsConfig # Ensure XttsConfig is imported
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs # Needed for safe load
    # Import the main API class
    from TTS.api import TTS
    from TTS.utils.manage import ModelManager
except ImportError as e:
    print(f"Error importing TTS library components: {e}")
    print("Please ensure TTS is installed correctly (pip install \"TTS~=0.21.3\")")
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
        description="Fine-tune an XTTS model using Coqui TTS library (v0.21.x API)."
    )

    # --- Dataset Arguments ---
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to the directory containing the metadata files and audio files.",
    )
    parser.add_argument(
        "--train_metadata_file", type=str, default="train_metadata.csv",
        help="Name of the training metadata file within the dataset_path.",
    )
    parser.add_argument(
        "--eval_metadata_file", type=str, default="eval_metadata.csv",
        help="Name of the evaluation metadata file within the dataset_path.",
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

    # --- Training Arguments (some might be set in config directly) ---
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
    train_metadata_path = dataset_path / args.train_metadata_file
    eval_metadata_path = dataset_path / args.eval_metadata_file
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"[bold green]Starting XTTS Fine-tuning Process (v0.21.x API)[/bold green]")
    logger.info(f"Dataset Path: [cyan]{dataset_path}[/cyan]")
    logger.info(f"Training Metadata: [cyan]{train_metadata_path}[/cyan]")
    logger.info(f"Evaluation Metadata: [cyan]{eval_metadata_path}[/cyan]")
    logger.info(f"Output Path: [cyan]{output_path}[/cyan]")
    logger.info(f"Pre-trained Model: [cyan]{args.model_name}[/cyan]")

    if not train_metadata_path.exists():
        logger.error(f"Training metadata file not found: {train_metadata_path}")
        return
    # Check for eval metadata only if the file name is provided (allows training without eval)
    eval_metadata_exists = False
    if args.eval_metadata_file:
        if eval_metadata_path.exists():
            eval_metadata_exists = True
        else:
            logger.warning(f"Evaluation metadata file not found: {eval_metadata_path}. Proceeding without evaluation.")

    # --- Configuration Setup ---
    logger.info("Setting up configurations...")

    # Audio config (usually okay to keep defaults from the model)
    audio_config = BaseAudioConfig(
        sample_rate=24000, # XTTS default
    )

    # Dataset config
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train=args.train_metadata_file,
        meta_file_val=args.eval_metadata_file if eval_metadata_exists else None,
        path=str(dataset_path)
    )

    # We need ModelManager to download/find the model first
    manager = ModelManager()
    model_path, config_path, _ = manager.download_model(args.model_name)

    # --- Manually resolve config path if manager returns None from cache ---
    if config_path is None and model_path is not None and Path(model_path).is_dir():
        potential_config_path = Path(model_path) / "config.json"
        if potential_config_path.exists():
            logger.info("Manually resolved config path from model directory.")
            config_path = str(potential_config_path)
        else:
             logger.warning(f"config.json not found in model directory: {model_path}")
    # --- End Manual Resolution ---

    # --- Robustly Check Paths and Load Config ---
    if config_path is None or not Path(config_path).exists():
         logger.error(f"Could not find or resolve config.json path: {config_path}")
         logger.error(f"Please check the model name '{args.model_name}' or network connection.")
         return
    else:
        logger.info(f"Found model config at: [cyan]{config_path}[/cyan]")

    if model_path is None or not Path(model_path).is_dir():
         logger.error(f"Could not find or resolve model directory path: {model_path}")
         logger.error(f"Download might have failed for model '{args.model_name}'.")
         return
    else:
         logger.info(f"Model directory found at: [cyan]{model_path}[/cyan]")

    # Load base model config from the verified path
    config = XttsConfig()
    config.load_json(config_path)
    config.audio = audio_config # Ensure audio config is updated

    # --- Update config with training parameters ---
    config.batch_size = args.batch_size
    config.eval_batch_size = max(1, args.batch_size // 2)
    config.num_loader_workers = args.num_workers
    config.num_eval_loader_workers = args.num_workers
    config.output_path = str(output_path)
    config.epochs = args.epochs
    config.save_step = args.save_step
    config.log_step = args.log_step
    config.save_checkpoints = True
    config.save_all_best = True # Save best model based on eval loss
    config.print_step = 50
    config.plot_step = 100
    config.seed = args.seed
    config.gradient_accumulation_steps = args.grad_accum
    config.run_eval = eval_metadata_exists
    config.test_delay_epochs = 5
    config.early_stop_patience = 10
    config.min_eval_loss = 0.01
    config.optimizer = "AdamW"
    config.optimizer_params = {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-4}
    config.lr = args.lr

    # --- Add dataset config to the main config --- >
    config.datasets = [dataset_config]

    # --- Initialize Model using TTS API ---
    # The TTS API class often handles model loading internally based on config
    logger.info("Initializing TTS API...")
    # Ensure correct classes are allowlisted for checkpoint loading via API
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

    tts_api = TTS(model_path=model_path, config_path=config_path, gpu=args.use_gpu)

    # --- Start Training via API ---
    logger.info(f"[bold green]Starting Training via TTS.train_tts()...[/bold green]")
    try:
        # The train_tts method takes the config object
        tts_api.train_tts(config, output_path=str(output_path))
        logger.info(f"[bold green]Training Finished Successfully![/bold green]")
    except Exception as e:
        logger.exception(f"[bold red]An error occurred during training:[/bold red] {e}")


if __name__ == "__main__":
    main() 