# src/main_train.py
import logging
import logging.handlers
import argparse
import os
import datetime
import sys
import re
import signal
import threading
from tqdm import tqdm

from .serial_rotating_handler import SerialRotatingFileHandler
from .config import load_config
from .cfr_trainer import CFRTrainer, GracefulShutdownException

# Global logger instance (initialized after setup)
logger = logging.getLogger(__name__)

# Shared event for graceful shutdown
shutdown_event = threading.Event()


def handle_sigint(sig, frame):
    """Signal handler for SIGINT (Ctrl+C)."""
    if not shutdown_event.is_set():
        # Do not call logger here: unsafe in signal handler context
        # logger.warning("SIGINT received. Requesting graceful shutdown...")
        print("\nSIGINT received. Requesting graceful shutdown...", file=sys.stderr)
        shutdown_event.set()
        # Optional: Restore default handler if needed, though re-raising KeyboardInterrupt is cleaner
        # signal.signal(signal.SIGINT, signal.SIG_DFL)
    else:
        # logger.warning("Multiple SIGINT received. Shutdown already in progress.")
        print(
            "\nMultiple SIGINT received. Shutdown already in progress.", file=sys.stderr
        )


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


def setup_logging(config, verbose: bool):
    """Configures logging to console and timestamped, serially chunked files within run directories."""
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

    # Create timestamped directory for this run
    run_timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H%M%S")
    run_log_dir = os.path.join(main_log_dir, f"{log_prefix}_run-{run_timestamp}")
    try:
        os.makedirs(run_log_dir, exist_ok=True)
    except OSError as e:
        print(
            f"ERROR: Could not create run-specific log directory '{run_log_dir}': {e}. Logging disabled."
        )
        return None, None

    # Define the base log filename *pattern* for the custom handler (without serial)
    base_log_pattern = os.path.join(run_log_dir, f"{log_prefix}_run-{run_timestamp}")

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

    # Console Handler (using Tqdm handler if possible)
    if sys.stderr.isatty():
        ch = TqdmLoggingHandler()
    else:
        ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(console_log_level)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # Use the Custom Serial Rotating File Handler
    try:
        max_bytes = config.logging.log_max_bytes
        backup_count = config.logging.log_backup_count
        # Pass the pattern, maxBytes, and backupCount
        fh = SerialRotatingFileHandler(
            base_log_pattern,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )

        fh.setLevel(file_log_level)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

        # Log initial messages *after* handlers are set up (using lazy %)
        logger.info("-" * 50)
        logger.info("Logging initialized for run timestamp: %s", run_timestamp)
        logger.info("Run Log Directory: %s", run_log_dir)
        # Get the currently active filename from the handler
        current_log_file = fh.baseFilename
        logger.info(
            "Logging to: %s (rotates to %s_NNN.log inside dir, _001 is oldest)",
            current_log_file,
            base_log_pattern,
        )
        logger.info("Log Chunk Size: %.1f MB", max_bytes / (1024 * 1024))
        logger.info("Max Log Chunks: %d", backup_count)
        logger.info("File Log Level: %s", logging.getLevelName(file_log_level))
        logger.info("Console Log Level: %s", logging.getLevelName(console_log_level))
        logger.info("Command: %s", " ".join(sys.argv))
        logger.info("-" * 50)

    except Exception as e:
        print(
            f"ERROR: Could not set up serial rotating file logging using pattern {base_log_pattern}: {e}"
        )
        # Print traceback for debugging setup issues
        import traceback

        traceback.print_exc()
        return None, None  # Indicate failure

    # Create/Update latest log directory link/copy (pointing to the directory)
    latest_log_link_path = os.path.join(main_log_dir, "latest_run")
    try:
        absolute_run_log_dir = os.path.abspath(run_log_dir)
        if sys.platform == "win32":
            # Windows: Create a simple text file pointing to the latest run dir.
            marker_path = latest_log_link_path + ".txt"
            if os.path.exists(marker_path):
                os.remove(marker_path)
            with open(marker_path, "w", encoding="utf-8") as f:
                f.write(f"Latest run directory: {absolute_run_log_dir}\n")
            logger.info("Updated latest run marker file: %s", marker_path)
        else:
            # Unix-like: Use symlink to the directory
            if os.path.lexists(latest_log_link_path):
                os.remove(latest_log_link_path)
            os.symlink(
                absolute_run_log_dir, latest_log_link_path, target_is_directory=True
            )
            logger.info("Updated latest_run symlink -> %s", absolute_run_log_dir)
    except Exception as e:
        logger.error("Could not create/update latest_run link/marker: %s", e)

    # Reduce Verbosity from Libraries
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Return the run_log_dir
    return run_log_dir, run_timestamp  # Indicate success


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

    # --- Setup Shutdown Handling ---
    # Register signal handler *before* potentially long operations
    signal.signal(signal.SIGINT, handle_sigint)

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
    logger.info("Configuration loaded from: %s", args.config)

    # Override config settings from command line if provided
    if args.iterations is not None:
        config.cfr_training.num_iterations = args.iterations
        logger.info("Overriding iterations from command line: %d", args.iterations)
    if args.save_path is not None:
        config.persistence.agent_data_save_path = args.save_path
        logger.info("Overriding save path from command line: %s", args.save_path)

    # Set the system recursion limit
    if config.system.recursion_limit:
        try:
            sys.setrecursionlimit(config.system.recursion_limit)
            logger.info(
                "System recursion limit set to: %d", config.system.recursion_limit
            )
        except Exception as e:
            logger.error("Failed to set recursion limit: %s", e)

    logger.info("Confirm recursion limit: %d", sys.getrecursionlimit())

    # Initialize trainer, passing the shutdown event
    try:
        trainer = CFRTrainer(
            config,
            run_log_dir=run_log_dir,
            shutdown_event=shutdown_event,  # Pass the event
        )
    except Exception:
        logger.exception("Failed to initialize CFRTrainer:")
        sys.exit(1)

    # Load existing data if requested
    if args.load:
        if not config.persistence.agent_data_save_path:
            logger.error("Cannot load data: Agent data save path is not configured.")
        else:
            logger.info(
                "Attempting to load agent data from: %s",
                config.persistence.agent_data_save_path,
            )
            trainer.load_data()
    else:
        logger.info("Starting training from scratch (no data loaded).")

    # Run training, catching KeyboardInterrupt which is re-raised by the trainer on graceful shutdown
    try:
        trainer.train()
        logger.info("Training completed successfully.")
    except KeyboardInterrupt:
        # This is caught either from a direct interrupt before trainer loop,
        # or after the trainer catches GracefulShutdownException and re-raises KeyboardInterrupt.
        logger.warning("Training interrupted by user (Ctrl+C) or shutdown event.")
        logger.info("Exiting program.")
        # No need to save here, trainer saves *before* raising KeyboardInterrupt
    except GracefulShutdownException:
        # This should ideally be caught inside the trainer's loop now
        logger.error("GracefulShutdownException propagated unexpectedly to main.")
    except Exception:
        logger.exception("An unexpected error occurred during training:")
        logger.info("Attempting to save final progress before exiting due to error...")
        save_path = trainer.config.persistence.agent_data_save_path
        if not save_path or not os.path.basename(save_path):
            logger.error(
                "Cannot save progress: Agent data save path is invalid or not configured."
            )
        else:
            try:
                trainer.save_data()
                logger.info("Progress saved after error.")
            except Exception as save_e:
                logger.error("Failed to save progress after error: %s", save_e)
        sys.exit(1)  # Exit with error code after unexpected exception


if __name__ == "__main__":
    main()
