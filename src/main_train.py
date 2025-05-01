"""src/main_train.py"""

import logging
import logging.handlers
import argparse
import os
import datetime
import sys
from tqdm import tqdm

from .config import load_config
from .cfr_trainer import CFRTrainer

# Global logger instance (initialized after setup)
logger = logging.getLogger(__name__)


# Progress bar handler
class TqdmLoggingHandler(logging.Handler):
    """Passes log records to tqdm.write(), ensuring they don't interfere with the progress bar."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(
                msg, file=sys.stderr
            )  # Write to stderr to avoid interfering with stdout if needed
            self.flush()
        except Exception:
            self.handleError(record)


# Custom Namer for RotatingFileHandler
def log_namer(default_name: str) -> str:
    """
    Custom namer for RotatingFileHandler to achieve {prefix}_{date}_{number}.log format.
    Extracts prefix and date from the base filename structure.
    """
    # default_name will be like /path/to/logs/prefix_run_date/prefix.log.N
    dir_name, base_filename = os.path.split(default_name)
    parts = base_filename.split(".")
    prefix = parts[0]
    num = parts[-1]  # The rotation number

    # Extract date from directory name (assuming format prefix_run_YYYY_MM_DD-HHMMSS)
    run_dir_name = os.path.basename(dir_name)
    try:
        date_part = run_dir_name.split("_run_")[-1]
    except IndexError:
        # Fallback if directory name format is unexpected
        date_part = datetime.datetime.now().strftime("%Y_%m_%d-%H%M%S")

    new_filename = f"{prefix}_{date_part}_{int(num):03d}.log"
    return os.path.join(dir_name, new_filename)


def setup_logging(config, verbose: bool):
    """Configures logging to console, timestamped+chunked file, and latest log dir link."""
    log_level_file_str = config.logging.log_level_file.upper()
    log_level_console_str = config.logging.log_level_console.upper()

    file_log_level = getattr(logging, log_level_file_str, logging.DEBUG)
    default_console_log_level = getattr(logging, log_level_console_str, logging.WARNING)

    # Set console level: Use file level if verbose, otherwise use default console level.
    console_log_level = file_log_level if verbose else default_console_log_level

    main_log_dir = config.logging.log_dir
    log_prefix = config.logging.log_file_prefix

    if not main_log_dir or not isinstance(main_log_dir, str):
        print(
            f"ERROR: Invalid log directory '{main_log_dir}' in config. Logging disabled."
        )
        return None, None  # Indicate failure

    try:
        os.makedirs(main_log_dir, exist_ok=True)
    except OSError as e:
        print(
            f"ERROR: Could not create main log directory '{main_log_dir}': {e}. Logging disabled."
        )
        return None, None  # Indicate failure

    # Create timestamped dir for this run with the new format
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H%M%S")
    # Use the configured log prefix for the directory name
    run_log_dir = os.path.join(main_log_dir, f"{log_prefix}_run_{timestamp}")
    try:
        os.makedirs(run_log_dir, exist_ok=True)
    except OSError as e:
        print(
            f"ERROR: Could not create run-specific log directory '{run_log_dir}': {e}. Logging disabled."
        )
        return None, None

    # Base log file path inside run dir (still uses prefix.log as base for rotator)
    log_filename_base = os.path.join(run_log_dir, f"{log_prefix}.log")

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Root Logger Config
    root_logger = logging.getLogger()
    root_logger.setLevel(
        min(file_log_level, console_log_level)
    )  # Set root level to lowest required

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Display progress bar at base
    # Use Tqdm handler IF we have a TTY, otherwise standard StreamHandler
    if sys.stderr.isatty():
        ch = TqdmLoggingHandler()
    else:
        ch = logging.StreamHandler(sys.stderr)  # Log to stderr
    ch.setLevel(console_log_level)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # File chunking / rotating log file handler
    try:
        # Rotate logs within the timestamped directory
        max_bytes = config.logging.log_max_bytes  # ~10MB
        backup_count = config.logging.log_backup_count  # Number of backup files
        fh = logging.handlers.RotatingFileHandler(
            log_filename_base,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        # Assign the custom namer
        fh.namer = log_namer
        # Note: Rollover occurs *before* the next message is written.
        # The current file will be named prefix.log until it's full.
        # When it rolls over, prefix.log becomes prefix_date_001.log,
        # and a new prefix.log is created. This isn't exactly the requested
        # {prefix}_{date}_{batch}.log format for *all* files including the current one.
        # Achieving that would require a more complex handler or renaming after creation.
        # This setup gives: prefix.log (current), prefix_date_001.log, prefix_date_002.log ...

        fh.setLevel(file_log_level)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

        # Log initial messages *after* handlers are set up
        logger.info("-" * 50)
        logger.info(f"Logging initialized for run: {timestamp}")
        logger.info(f"Log Directory: {run_log_dir}")
        logger.info(
            f"Base Log File: {log_filename_base} (rolls over to {log_prefix}_{timestamp}_NNN.log)"
        )
        logger.info(f"Log Chunk Size: {max_bytes / (1024*1024):.1f} MB")
        logger.info(f"Max Log Chunks: {backup_count}")
        logger.info(f"File Log Level: {logging.getLevelName(file_log_level)}")
        logger.info(f"Console Log Level: {logging.getLevelName(console_log_level)}")
        logger.info(f"Command: {' '.join(sys.argv)}")
        logger.info("-" * 50)

    except Exception as e:
        print(
            f"ERROR: Could not set up rotating file logging at {log_filename_base}: {e}"
        )
        # Remove console handler if file handler failed? Maybe not, console might still be useful.
        return None, None  # Indicate failure

    # Create/Update latest log directory link/copy
    latest_log_link_path = os.path.join(main_log_dir, "latest_run")
    try:
        absolute_run_log_dir = os.path.abspath(run_log_dir)
        if sys.platform == "win32":
            # Windows: Cannot easily "link" directories without special tools/permissions.
            # Create a simple text file pointing to the latest run dir.
            with open(latest_log_link_path + ".txt", "w") as f:
                f.write(f"Latest run directory: {absolute_run_log_dir}\n")
            logger.info(f"Updated latest run marker file: {latest_log_link_path}.txt")
        else:
            # Unix-like: Use symlink to the directory
            if os.path.lexists(latest_log_link_path):  # Use lexists for links
                os.remove(latest_log_link_path)
            os.symlink(
                absolute_run_log_dir, latest_log_link_path, target_is_directory=True
            )
            logger.info(f"Updated latest_run symlink -> {absolute_run_log_dir}")
    except Exception as e:
        logger.error(f"Could not create/update latest_run link/marker: {e}")

    # Reduce Verbosity from Libraries
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Return the run_log_dir for potential use elsewhere (like saving analysis)
    return run_log_dir, timestamp  # Indicate success


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
        help="Override save path for agent data (from config)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose console logging (uses level specified in config's log_level_file)",
    )

    args = parser.parse_args()

    # Load configuration *before* setting up logging
    config = load_config(args.config)
    if not config:
        print("ERROR: Failed to load configuration. Exiting.")
        sys.exit(1)

    # Setup logging
    run_log_dir, run_timestamp = setup_logging(config, args.verbose)
    if not run_log_dir:
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

    # Set the system recursion limit
    if config.system.recursion_limit:
        try:
            sys.setrecursionlimit(config.system.recursion_limit)
            logger.info(f"System recursion limit set to: {config.system.recursion_limit}")
        except Exception as e:
            logger.error(f"Failed to set recursion limit: {e}")

    logger.info(f"Confirm recursion limit: {sys.getrecursionlimit()}")

    # Initialize trainer
    try:
        trainer = CFRTrainer(
            config, run_log_dir=run_log_dir
        )  # Pass run_log_dir to trainer
    except Exception as e:
        logger.exception("Failed to initialize CFRTrainer:")
        sys.exit(1)

    # Load existing data if requested
    if args.load:
        if not config.persistence.agent_data_save_path:
            logger.error("Cannot load data: Agent data save path is not configured.")
        else:
            logger.info(
                f"Attempting to load agent data from: {config.persistence.agent_data_save_path}"
            )
            trainer.load_data()
    else:
        logger.info("Starting training from scratch (no data loaded).")

    # Run training
    try:
        # trainer.train() now handles the tqdm loop internally
        trainer.train()
        logger.info("Training completed successfully.")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
        logger.info("Attempting to save current progress...")
        if not trainer.config.persistence.agent_data_save_path or not os.path.basename(
            trainer.config.persistence.agent_data_save_path
        ):
            logger.error(
                "Cannot save progress: Agent data save path is invalid or not configured."
            )
        else:
            try:
                trainer.save_data()
                logger.info("Progress saved.")
            except Exception as save_e:
                logger.error(f"Failed to save progress after interrupt: {save_e}")
    except Exception as e:
        logger.exception("An unexpected error occurred during training:")
        logger.info("Attempting to save current progress before exiting...")
        if not trainer.config.persistence.agent_data_save_path or not os.path.basename(
            trainer.config.persistence.agent_data_save_path
        ):
            logger.error(
                "Cannot save progress: Agent data save path is invalid or not configured."
            )
        else:
            try:
                trainer.save_data()
                logger.info("Progress saved.")
            except Exception as save_e:
                logger.error(f"Failed to save progress after error: {save_e}")


if __name__ == "__main__":
    main()
