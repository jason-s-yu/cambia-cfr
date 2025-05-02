# src/main_train.py
import logging
import logging.handlers
import argparse
import os
import datetime
import sys
import signal
import threading
import multiprocessing
from typing import List, Optional
from tqdm import tqdm

from .serial_rotating_handler import SerialRotatingFileHandler
from .config import load_config
from .cfr.trainer import CFRTrainer
from .utils import LogQueue

# Global logger instance (initialized after setup)
logger = logging.getLogger(__name__)

# Shared event for graceful shutdown
shutdown_event = threading.Event()
# Global reference to the queue listener thread
queue_listener: Optional[logging.handlers.QueueListener] = None


def handle_sigint(sig, frame):
    """Signal handler for SIGINT (Ctrl+C)."""
    if not shutdown_event.is_set():
        print("\nSIGINT received. Requesting graceful shutdown...", file=sys.stderr)
        shutdown_event.set()
    else:
        print(
            "\nMultiple SIGINT received. Shutdown already in progress.", file=sys.stderr
        )


class TqdmLoggingHandler(logging.Handler):
    """Passes log records to tqdm.write(), ensuring they don't interfere with the progress bar."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(config, verbose: bool):
    """Configures logging using a QueueListener for multiprocessing."""
    global queue_listener  # Indicate we are modifying the global variable

    log_level_file_str = config.logging.log_level_file.upper()
    log_level_console_str = config.logging.log_level_console.upper()

    file_log_level = getattr(logging, log_level_file_str, logging.DEBUG)
    default_console_log_level = getattr(logging, log_level_console_str, logging.WARNING)
    console_log_level = file_log_level if verbose else default_console_log_level

    main_log_dir = config.logging.log_dir
    log_prefix = config.logging.log_file_prefix

    if not main_log_dir or not isinstance(main_log_dir, str):
        print(f"ERROR: Invalid log directory '{main_log_dir}'. Logging disabled.")
        return None, None, None  # Return three Nones

    # --- Create Directories ---
    try:
        os.makedirs(main_log_dir, exist_ok=True)
        run_timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H%M%S")
        run_log_dir = os.path.join(main_log_dir, f"{log_prefix}_run-{run_timestamp}")
        os.makedirs(run_log_dir, exist_ok=True)
    except OSError as e:
        print(
            f"ERROR: Could not create log directory '{run_log_dir or main_log_dir}': {e}. Logging disabled."
        )
        return None, None, None  # Return three Nones

    # --- Create Handlers (for the main process/listener) ---
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s - %(name)-25s - %(message)s"
    )
    handlers: List[logging.Handler] = []

    # Console Handler
    if sys.stderr.isatty():
        ch = TqdmLoggingHandler()
    else:
        ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(console_log_level)
    ch.setFormatter(formatter)
    handlers.append(ch)

    # File Handler (Serial Rotating)
    try:
        base_log_pattern = os.path.join(run_log_dir, f"{log_prefix}_run-{run_timestamp}")
        max_bytes = config.logging.log_max_bytes
        backup_count = config.logging.log_backup_count
        fh = SerialRotatingFileHandler(
            base_log_pattern,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(file_log_level)
        fh.setFormatter(formatter)
        handlers.append(fh)
        current_log_file = fh.baseFilename  # Get initial file name
    except Exception as e:
        print(f"ERROR: Could not set up serial rotating file logging: {e}")
        # Don't add the file handler if setup fails
        current_log_file = "File logging disabled"

    # --- Setup Queue and Listener ---
    log_queue: LogQueue = multiprocessing.Queue(-1)  # Create the queue
    # Configure the listener to use the handlers created above
    queue_listener = logging.handlers.QueueListener(
        log_queue, *handlers, respect_handler_level=True
    )
    queue_listener.start()

    # --- Configure Root Logger (for the main process) ---
    # The root logger in the main process doesn't need handlers directly,
    # as the listener handles dispatch. But setting the level is good practice.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set root logger to lowest level (DEBUG)
    # Remove any default handlers that might have been added
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    # Add a handler that directs logs *from the main process* to the queue
    main_process_queue_handler = logging.handlers.QueueHandler(log_queue)
    root_logger.addHandler(main_process_queue_handler)

    # --- Initial Log Messages (go through the queue) ---
    logger.info("-" * 50)
    logger.info("Logging initialized via QueueListener for run: %s", run_timestamp)
    logger.info("Run Log Directory: %s", run_log_dir)
    logger.info(
        "Logging to File: %s (Level: %s)",
        current_log_file,
        logging.getLevelName(file_log_level),
    )
    logger.info(
        "Logging to Console: (Level: %s)", logging.getLevelName(console_log_level)
    )
    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("-" * 50)

    # --- Update latest_run link ---
    latest_log_link_path = os.path.join(main_log_dir, "latest_run")
    try:
        absolute_run_log_dir = os.path.abspath(run_log_dir)
        if sys.platform == "win32":
            marker_path = latest_log_link_path + ".txt"
            if os.path.exists(marker_path):
                os.remove(marker_path)
            with open(marker_path, "w", encoding="utf-8") as f:
                f.write(f"Latest run directory: {absolute_run_log_dir}\n")
            logger.info("Updated latest run marker file: %s", marker_path)
        else:
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
    logging.getLogger("PIL").setLevel(logging.WARNING)  # Example

    return run_log_dir, run_timestamp, log_queue  # Return the queue


def main():
    global queue_listener  # Access the global listener variable

    parser = argparse.ArgumentParser(description="Run CFR+ Training for Cambia")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations to run (overrides config)",
    )
    parser.add_argument(
        "--load", action="store_true", help="Load existing agent data before training"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Override save path for agent data"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose console logging"
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_sigint)

    config = load_config(args.config)
    if not config:
        print("ERROR: Failed to load configuration. Exiting.")
        sys.exit(1)

    # Setup logging and get the queue
    run_log_dir, run_timestamp, log_queue = setup_logging(config, args.verbose)
    if not run_log_dir or log_queue is None:
        print("ERROR: Failed to set up logging. Exiting.")
        # Try to stop listener if it somehow started
        if queue_listener:
            queue_listener.stop()
        sys.exit(1)

    # Logger is now configured globally

    logger.info("--- Starting Cambia CFR+ Training ---")
    logger.info("Configuration loaded from: %s", args.config)

    if args.iterations is not None:
        config.cfr_training.num_iterations = args.iterations
        logger.info("Overriding iterations from command line: %d", args.iterations)
    if args.save_path is not None:
        config.persistence.agent_data_save_path = args.save_path
        logger.info("Overriding save path from command line: %s", args.save_path)

    if config.system.recursion_limit:
        try:
            sys.setrecursionlimit(config.system.recursion_limit)
            logger.info("System recursion limit set to: %d", sys.getrecursionlimit())
        except Exception as e:
            logger.error("Failed to set recursion limit: %s", e)

    # Initialize trainer
    try:
        trainer = CFRTrainer(
            config,
            run_log_dir=run_log_dir,
            shutdown_event=shutdown_event,
            log_queue=log_queue,  # Pass the queue
        )
    except Exception:
        logger.exception("Failed to initialize CFRTrainer:")
        if queue_listener:
            queue_listener.stop()  # Stop listener on error
        sys.exit(1)

    # Load data if requested
    if args.load:
        logger.info(
            "Attempting to load agent data from: %s",
            config.persistence.agent_data_save_path,
        )
        trainer.load_data()
    else:
        logger.info("Starting training from scratch.")

    # Run training
    exit_code = 0
    try:
        trainer.train()
        logger.info("Training completed successfully.")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C) or shutdown event.")
        logger.info("Exiting program.")
        # Saving is handled within the trainer's exception handling
    except Exception:
        logger.exception("An unexpected error occurred during training:")
        # Attempt save (handled by trainer's loop already)
        exit_code = 1  # Indicate error exit
    finally:
        # Ensure the queue listener is stopped cleanly
        if queue_listener:
            logger.info("Stopping log queue listener...")
            queue_listener.stop()
            logger.info("Log queue listener stopped.")
        logger.info("--- Cambia CFR+ Training Finished ---")
        sys.exit(exit_code)


if __name__ == "__main__":
    # Set start method for multiprocessing (fork is problematic with threads like QueueListener)
    # 'spawn' is generally safer but might be slower. 'forkserver' is another option.
    # Set this *before* any multiprocessing objects are created.
    start_method_set = False
    try:
        # Use 'forkserver' if available (often better performance than 'spawn')
        # Use 'spawn' on Windows or if 'forkserver' is unavailable/causes issues
        if sys.platform == "win32":
            multiprocessing.set_start_method("spawn", force=True)
            start_method_set = True
        else:
            try:
                multiprocessing.set_start_method("forkserver", force=True)
                start_method_set = True
            except ValueError:
                logging.warning("forkserver start method not available, using spawn.")
                multiprocessing.set_start_method("spawn", force=True)
                start_method_set = True
    except RuntimeError:
        # Context might have already been set if this isn't the main entry point
        logging.debug("Multiprocessing context already set.")
        pass  # Assume it was set correctly elsewhere
    except Exception as e:
        logging.error("Error setting multiprocessing start method: %s", e)

    # Add logging here AFTER attempting to set the start method
    # Note: Logger might not be fully configured yet if setup_logging hasn't run
    # Use basic print or preliminary logging setup if needed
    effective_method = multiprocessing.get_start_method(allow_none=True)
    print(f"INFO: Effective multiprocessing start method: {effective_method}")
    # If using logger, ensure basicConfig is set or queue exists
    # logger.info("Effective multiprocessing start method: %s", effective_method) # Use this if logger is ready

    main()
