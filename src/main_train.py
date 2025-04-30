# src/main_train.py
import logging
import logging.handlers
import argparse
import os
import datetime
import sys # For checking platform, exit
import shutil # For copying latest.log on Windows

from .config import load_config
from .cfr_trainer import CFRTrainer

# Global logger instance (initialized after setup)
logger = logging.getLogger(__name__)

def setup_logging(config, verbose: bool):
    """Configures logging to console, timestamped file, and latest.log link."""
    log_level_str = config.logging.log_level.upper() # Use log_level
    file_log_level = getattr(logging, log_level_str, logging.INFO)
    # Set console level based on verbose flag, but not lower than file level
    console_log_level = file_log_level if verbose else logging.WARNING # Default console to WARNING unless verbose

    log_dir = config.logging.log_dir
    log_prefix = config.logging.log_file_prefix
    # Check if log_dir is valid before creating
    if not log_dir or not isinstance(log_dir, str):
        print(f"ERROR: Invalid log directory '{log_dir}' in config. Logging disabled.")
        return False # Indicate failure

    try:
        os.makedirs(log_dir, exist_ok=True) # Ensure log directory exists
    except OSError as e:
        print(f"ERROR: Could not create log directory '{log_dir}': {e}. Logging disabled.")
        return False # Indicate failure


    # --- Create Timestamped File Name ---
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H%M%S")
    log_filename = os.path.join(log_dir, f"{log_prefix}_{timestamp}.log")

    # --- Formatter ---
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Root Logger Configuration ---
    root_logger = logging.getLogger()
    # Set root logger level to the lowest level used by handlers
    root_logger.setLevel(min(file_log_level, console_log_level))

    # Remove existing handlers to avoid duplicates if script is re-run
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # --- Console Handler ---
    ch = logging.StreamHandler()
    ch.setLevel(console_log_level)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # --- Timestamped File Handler ---
    try:
        fh = logging.FileHandler(log_filename, mode='a') # Append mode
        fh.setLevel(file_log_level)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
        # Use the logger configured *after* adding handlers
        logger.info(f"Command: {' '.join(sys.argv)}")
        logger.info(f"Logging to timestamped file: {log_filename} (Level: {logging.getLevelName(file_log_level)})")
        logger.info(f"Console logging level: {logging.getLevelName(console_log_level)}")
    except Exception as e:
        # Use print as logger might not be fully functional yet
        print(f"ERROR: Could not set up timestamped file logging at {log_filename}: {e}")
        return False # Indicate failure


    # --- Create/Update latest.log ---
    latest_log_path = os.path.join(log_dir, "latest.log")
    try:
        # Use absolute path for symlink target for robustness
        absolute_log_filename = os.path.abspath(log_filename)
        if sys.platform == 'win32':
            # Windows: Use copy (symlinks require admin privileges)
            if os.path.exists(absolute_log_filename): # Ensure source file exists before copying
                 shutil.copy2(absolute_log_filename, latest_log_path)
                 logger.info(f"Copied latest log to: {latest_log_path}")
            else:
                 logger.error(f"Source log file '{absolute_log_filename}' not found for copying to latest.log")
        else:
            # Unix-like: Use symlink
            if os.path.exists(latest_log_path) or os.path.islink(latest_log_path):
                os.remove(latest_log_path)
            if os.path.exists(absolute_log_filename): # Ensure source file exists before linking
                os.symlink(absolute_log_filename, latest_log_path) # Absolute symlink
                logger.info(f"Updated latest.log symlink -> {absolute_log_filename}")
            else:
                 logger.error(f"Source log file '{absolute_log_filename}' not found for creating symlink latest.log")
    except Exception as e:
        logger.error(f"Could not create/update latest.log link/copy: {e}")


    # --- Reduce Verbosity from Libraries ---
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    return True # Indicate success

def main():
    parser = argparse.ArgumentParser(description="Run CFR+ Training for Cambia")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations to run (overrides config if provided)",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load existing agent data before training",
    )
    parser.add_argument(
         "--save_path",
         type=str,
         default=None,
         help="Override save path for agent data (from config)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose console logging (uses level from config)",
    )


    args = parser.parse_args()

    # Load configuration *before* setting up logging
    config = load_config(args.config)
    if not config: # Check if config loading failed
         print("ERROR: Failed to load configuration. Exiting.")
         sys.exit(1)

    # Setup logging using loaded config and verbose flag
    if not setup_logging(config, args.verbose):
         print("ERROR: Failed to set up logging. Exiting.")
         sys.exit(1)
    # Logger is now configured globally

    logger.info("--- Starting Cambia CFR+ Training ---")
    logger.info(f"Configuration loaded from: {args.config}")

    # Override config settings from command line if provided
    if args.iterations is not None:
        config.cfr_training.num_iterations = args.iterations
        logger.info(f"Overriding iterations from command line: {args.iterations}")
    if args.save_path is not None:
         config.persistence.agent_data_save_path = args.save_path
         logger.info(f"Overriding save path from command line: {args.save_path}")


    # Initialize trainer
    try:
         trainer = CFRTrainer(config)
    except Exception as e:
         logger.exception("Failed to initialize CFRTrainer:")
         sys.exit(1)


    # Load existing data if requested
    if args.load:
        if not config.persistence.agent_data_save_path:
             logger.error("Cannot load data: Agent data save path is not configured.")
        else:
             logger.info(f"Attempting to load agent data from: {config.persistence.agent_data_save_path}")
             trainer.load_data()
    else:
        logger.info("Starting training from scratch (no data loaded).")

    # Run training
    try:
        trainer.train() # Uses iterations from config (potentially overridden)
        logger.info("Training completed successfully.")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
        logger.info("Attempting to save current progress...")
        # Validate save path before saving on interrupt
        if not trainer.config.persistence.agent_data_save_path or not os.path.basename(trainer.config.persistence.agent_data_save_path):
            logger.error("Cannot save progress: Agent data save path is invalid or not configured.")
        else:
            try:
                trainer.save_data()
                logger.info("Progress saved.")
            except Exception as save_e:
                logger.error(f"Failed to save progress after interrupt: {save_e}")
    except Exception as e:
        # Use logger.exception to include traceback
        logger.exception("An unexpected error occurred during training:")
        logger.info("Attempting to save current progress before exiting...")
        # Validate save path before saving on error
        if not trainer.config.persistence.agent_data_save_path or not os.path.basename(trainer.config.persistence.agent_data_save_path):
            logger.error("Cannot save progress: Agent data save path is invalid or not configured.")
        else:
            try:
                trainer.save_data()
                logger.info("Progress saved.")
            except Exception as save_e:
                logger.error(f"Failed to save progress after error: {save_e}")


if __name__ == "__main__":
    main()