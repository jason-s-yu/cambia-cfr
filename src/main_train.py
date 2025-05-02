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
import multiprocessing.managers  # Import managers submodule
import time  # Import time for potential sleep
from typing import List, Optional
from tqdm import tqdm

from .serial_rotating_handler import SerialRotatingFileHandler
from .config import load_config
from .cfr.trainer import CFRTrainer
from .utils import LogQueue
from .cfr.exceptions import GracefulShutdownException  # Import exception

# Global logger instance (initialized after setup)
logger = logging.getLogger(__name__)

# Shared event for graceful shutdown
shutdown_event = threading.Event()
# Global reference to the queue listener thread
queue_listener: Optional[logging.handlers.QueueListener] = None
# Global reference to the main process queue handler
main_process_queue_handler: Optional[logging.handlers.QueueHandler] = None


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
            # Use file=sys.stderr for tqdm to avoid interfering with stdout pipes
            # nolock=True might help slightly with threading, but main issue is elsewhere
            tqdm.write(msg, file=sys.stderr)  # , nolock=True)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(config, verbose: bool, log_queue: LogQueue):
    """Configures logging using a QueueListener for multiprocessing."""
    global queue_listener, main_process_queue_handler  # Add main handler to globals

    log_level_file_str = config.logging.log_level_file.upper()
    log_level_console_str = config.logging.log_level_console.upper()

    file_log_level = getattr(logging, log_level_file_str, logging.DEBUG)
    default_console_log_level = getattr(logging, log_level_console_str, logging.WARNING)
    console_log_level = file_log_level if verbose else default_console_log_level

    main_log_dir = config.logging.log_dir
    log_prefix = config.logging.log_file_prefix

    if not main_log_dir or not isinstance(main_log_dir, str):
        print(f"ERROR: Invalid log directory '{main_log_dir}'. Logging disabled.")
        return None, None  # Return two Nones

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
        return None, None  # Return two Nones

    # --- Create Handlers (for the main process/listener) ---
    # Include process name in the format string
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s - [%(processName)-20s] - %(name)-25s - %(message)s"
    )
    handlers: List[logging.Handler] = []

    # Console Handler
    if sys.stderr.isatty():
        # Use Tqdm handler only if stderr is a TTY
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
        current_log_file = "File logging disabled"

    # --- Setup Queue Listener (uses the passed manager queue) ---
    # Ensure queue_listener is stopped if it exists from a previous failed run
    if queue_listener:
        queue_listener.stop()

    queue_listener = logging.handlers.QueueListener(
        log_queue, *handlers, respect_handler_level=True
    )
    queue_listener.start()

    # --- Configure Root Logger (for the main process) ---
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # Remove any previous handlers (especially important if run multiple times)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    # Add a handler that directs logs *from the main process* to the queue
    # Store reference to this specific handler
    main_process_queue_handler = logging.handlers.QueueHandler(log_queue)
    root_logger.addHandler(main_process_queue_handler)

    # --- Initial Log Messages (go through the queue) ---
    # Get logger instance *after* handler is added
    initial_logger = logging.getLogger(__name__)  # Use a local logger instance here
    initial_logger.info("-" * 50)
    initial_logger.info(
        "Logging initialized via QueueListener for run: %s", run_timestamp
    )
    initial_logger.info("Run Log Directory: %s", run_log_dir)
    initial_logger.info(
        "Logging to File: %s (Level: %s)",
        current_log_file,
        logging.getLevelName(file_log_level),
    )
    initial_logger.info(
        "Logging to Console: (Level: %s)", logging.getLevelName(console_log_level)
    )
    initial_logger.info("Command: %s", " ".join(sys.argv))
    initial_logger.info("-" * 50)

    # --- Update latest_run link ---
    latest_log_link_path = os.path.join(main_log_dir, "latest_run")
    try:
        absolute_run_log_dir = os.path.abspath(run_log_dir)
        if sys.platform == "win32":
            # Use a simple marker file on Windows
            marker_path = latest_log_link_path + ".txt"
            if os.path.exists(marker_path):
                os.remove(marker_path)
            with open(marker_path, "w", encoding="utf-8") as f:
                f.write(f"Latest run directory: {absolute_run_log_dir}\n")
            initial_logger.info("Updated latest run marker file: %s", marker_path)
        else:
            # Use symlink on POSIX systems
            if os.path.lexists(latest_log_link_path):
                os.remove(latest_log_link_path)
            os.symlink(
                absolute_run_log_dir, latest_log_link_path, target_is_directory=True
            )
            initial_logger.info("Updated latest_run symlink -> %s", absolute_run_log_dir)
    except Exception as e:
        initial_logger.error("Could not create/update latest_run link/marker: %s", e)

    # Reduce Verbosity from Libraries
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("joblib").setLevel(logging.WARNING)
    # Silence noisy matplotlib font messages if used later
    logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)

    return run_log_dir, run_timestamp  # Return run dir and timestamp


def main():
    global queue_listener, main_process_queue_handler  # Access global handler ref

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

    # --- Explicitly Manage the Multiprocessing Manager ---
    exit_code = 0
    manager: Optional[multiprocessing.managers.SyncManager] = None
    try:
        # Create and start the manager
        manager = multiprocessing.Manager()
        log_queue: LogQueue = manager.Queue(-1)

        # Setup logging using the manager's queue
        run_log_dir, run_timestamp = setup_logging(config, args.verbose, log_queue)
        if not run_log_dir:
            print("ERROR: Failed to set up logging. Exiting.")
            exit_code = 1
            raise SystemExit(exit_code)  # Use SystemExit to ensure finally block runs

        # Logger is now configured globally
        logger.info("--- Starting Cambia CFR+ Training ---")
        logger.info("Configuration loaded from: %s", args.config)

        # Apply command-line overrides
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
                log_queue=log_queue,  # Pass the manager's queue
            )
        except Exception:
            logger.exception("Failed to initialize CFRTrainer:")
            exit_code = 1
            raise  # Re-raise to be caught by the outer try...except

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
        try:
            trainer.train()
            logger.info("Training completed successfully.")
        except (
            KeyboardInterrupt,
            GracefulShutdownException,
        ):  # Catch both forms of shutdown
            logger.warning("Training interrupted by user (Ctrl+C) or shutdown event.")
            logger.info("Exiting program.")
            # Saving is handled within the trainer's exception handling
        except Exception:
            logger.exception("An unexpected error occurred during training:")
            # Attempt save (handled by trainer's loop already)
            exit_code = 1  # Indicate error exit

    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 1
        # Avoid logging here if logger setup failed
        if logging.getLogger().hasHandlers():
            logger.error("System exit called with code %s.", exit_code)
    except Exception as e:
        # Catch errors during manager setup or trainer init
        print(f"FATAL ERROR during setup: {e}", file=sys.stderr)
        # Attempt to log if logger was partially initialized
        if logging.getLogger().hasHandlers():
            logger.exception("FATAL ERROR during setup:")
        exit_code = 1
    finally:
        # --- Clean Shutdown ---
        print("--- Initiating Final Shutdown Sequence ---", file=sys.stderr)

        root_logger = logging.getLogger()
        log_available = root_logger.hasHandlers() and queue_listener is not None

        # 1. Remove the main process QueueHandler to prevent final logs causing errors
        if main_process_queue_handler:
            print("Removing main process log handler...", file=sys.stderr)
            root_logger.removeHandler(main_process_queue_handler)
            main_process_queue_handler = None  # Clear reference

        # 2. Stop the QueueListener
        if queue_listener:
            print("Stopping log queue listener...", file=sys.stderr)
            try:
                queue_listener.stop()
                print("Log queue listener stopped.", file=sys.stderr)
            except Exception as ql_e:
                print(f"Error stopping queue listener: {ql_e}", file=sys.stderr)
            # Short delay MAYBE helps listener process final items from queue
            # but manager shutdown is the main race condition target
            time.sleep(0.1)

        # 3. Shut down the Manager
        if manager:
            print("Shutting down multiprocessing manager...", file=sys.stderr)
            try:
                manager.shutdown()
                print("Manager shut down.", file=sys.stderr)
            except Exception as mgr_e:
                print(f"Error shutting down manager: {mgr_e}", file=sys.stderr)

        print("--- Cambia CFR+ Training Finished ---", file=sys.stderr)

        # 4. Final logging shutdown
        logging.shutdown()
        sys.exit(exit_code)


if __name__ == "__main__":
    # Set start method for multiprocessing *before* creating Manager or Pool
    start_method_set = False
    try:
        preferred_method = "forkserver" if sys.platform != "win32" else "spawn"
        available_methods = multiprocessing.get_all_start_methods()

        # Check the currently configured method
        current_method = multiprocessing.get_start_method(allow_none=True)

        force_set = current_method is None  # Only force if no method is explicitly set

        if preferred_method in available_methods:
            if force_set or current_method != preferred_method:
                multiprocessing.set_start_method(preferred_method, force=force_set)
                start_method_set = True
                print(f"INFO: Set multiprocessing start method to '{preferred_method}'")
            else:
                print(f"INFO: Multiprocessing start method already '{preferred_method}'")

        elif "spawn" in available_methods:
            if force_set or current_method != "spawn":
                multiprocessing.set_start_method("spawn", force=force_set)
                start_method_set = True
                print(
                    f"INFO: Preferred method '{preferred_method}' not available, using 'spawn'"
                )
            else:
                print("INFO: Multiprocessing start method already 'spawn'")

        elif current_method is None:  # Only error if no method set and no methods work
            # Should not happen on supported platforms
            print("ERROR: Neither 'forkserver' nor 'spawn' start methods available!")

    except RuntimeError as e:
        # Context might have already been set via other means or previous runs
        print(f"DEBUG: Multiprocessing start method likely already set? ({e})")
        pass  # Assume it was set correctly elsewhere
    except Exception as e:
        print(f"ERROR: Error setting multiprocessing start method: {e}")

    effective_method = multiprocessing.get_start_method(allow_none=True)
    print(f"INFO: Effective multiprocessing start method: {effective_method}")

    main()
