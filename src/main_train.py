"""src/main_train.py"""

import logging
import logging.handlers
import argparse
import os
import datetime
import sys
import signal
import threading
import multiprocessing

# Rich imports
from rich.console import Console

# from rich.logging import RichHandler # Using custom handler

from typing import List, Optional, Tuple

from .serial_rotating_handler import SerialRotatingFileHandler
from .config import load_config
from .cfr.trainer import CFRTrainer
from .utils import LogQueue as ProgressQueue
from .cfr.exceptions import GracefulShutdownException
from .live_display import LiveDisplayManager
from .live_log_handler import LiveLogHandler

# Global logger instance
logger = logging.getLogger(__name__)

# Shared event for graceful shutdown
shutdown_event = threading.Event()


def handle_sigint(sig, frame):
    """Signal handler for SIGINT (Ctrl+C)."""
    if not shutdown_event.is_set():
        print("\nSIGINT received. Requesting graceful shutdown...", file=sys.stderr)
        shutdown_event.set()
    else:
        print(
            "\nMultiple SIGINT received. Shutdown already in progress.", file=sys.stderr
        )


def setup_logging(
    config, verbose: bool, live_display_manager: LiveDisplayManager
) -> Optional[Tuple[str, str]]:
    """Configures logging, integrating with the LiveDisplayManager."""
    log_level_file_str = config.logging.log_level_file.upper()
    log_level_console_str = config.logging.log_level_console.upper()

    file_log_level = getattr(logging, log_level_file_str, logging.DEBUG)
    # Console level set low for handler, Rich display/filtering will manage output level if needed
    console_log_level = logging.DEBUG

    main_log_dir = config.logging.log_dir
    log_prefix = config.logging.log_file_prefix

    if not main_log_dir or not isinstance(main_log_dir, str):
        print(f"ERROR: Invalid log directory '{main_log_dir}'. Logging disabled.")
        return None

    # --- Create Run Directory ---
    try:
        run_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
        run_log_dir = os.path.join(main_log_dir, f"{log_prefix}_run_{run_timestamp}")
        os.makedirs(run_log_dir, exist_ok=True)
    except OSError as e:
        print(
            f"ERROR: Could not create log directory '{run_log_dir or main_log_dir}': {e}. Logging disabled."
        )
        return None

    # --- Create Handlers ---
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s - [%(processName)-20s] - %(name)-25s - %(message)s"
    )
    handlers: List[logging.Handler] = []

    # Live Handler
    live_handler = LiveLogHandler(live_display_manager, level=console_log_level)
    handlers.append(live_handler)

    # File Handler
    try:
        main_log_pattern = os.path.join(
            run_log_dir, f"{log_prefix}_run_{run_timestamp}-main"
        )
        max_bytes = config.logging.log_max_bytes
        backup_count = config.logging.log_backup_count
        fh = SerialRotatingFileHandler(
            main_log_pattern,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(file_log_level)
        fh.setFormatter(formatter)
        handlers.append(fh)
        main_log_file = fh.baseFilename
    except Exception as e:
        print(f"ERROR: Could not set up main process file logging: {e}")
        main_log_file = "File logging disabled"

    # --- Configure Root Logger ---
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    for handler in handlers:
        root_logger.addHandler(handler)

    # --- Initial Log Messages ---
    initial_logger = logging.getLogger(__name__)  # Use __name__ logger
    initial_logger.info("-" * 50)
    initial_logger.info("Logging initialized for run: %s", run_timestamp)
    initial_logger.info("Run Log Directory: %s", run_log_dir)
    initial_logger.info(
        "Main Log File: %s (Level: %s)",
        main_log_file,
        logging.getLevelName(file_log_level),
    )
    initial_logger.info(
        "Console Logging via Rich Display Manager (Handler Level: %s)",
        logging.getLevelName(console_log_level),
    )
    initial_logger.info("Command: %s", " ".join(sys.argv))
    initial_logger.info("-" * 50)

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
            initial_logger.info("Updated latest run marker file: %s", marker_path)
        else:
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
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    return run_log_dir, run_timestamp


def main():
    # --- DEBUG PRINT 1 ---
    print("DEBUG: main() started", file=sys.stderr)

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
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose console logging (DEBUG)",
    )
    args = parser.parse_args()

    # --- DEBUG PRINT 2 ---
    print(f"DEBUG: Args parsed: {args}", file=sys.stderr)

    signal.signal(signal.SIGINT, handle_sigint)

    config = load_config(args.config)
    if not config:
        print("ERROR: Failed to load configuration. Exiting.")
        sys.exit(1)

    # --- DEBUG PRINT 3 ---
    print("DEBUG: Config loaded", file=sys.stderr)

    # Initialize Rich Console and LiveDisplayManager
    rich_console = Console(stderr=True, record=False)
    total_iterations_for_display = (
        args.iterations
        if args.iterations is not None
        else config.cfr_training.num_iterations
    )
    live_display_manager = LiveDisplayManager(
        num_workers=config.cfr_training.num_workers,
        total_iterations=total_iterations_for_display,
        console=rich_console,
    )

    # --- DEBUG PRINT 4 ---
    print("DEBUG: LiveDisplayManager initialized", file=sys.stderr)

    # Multiprocessing Manager setup
    exit_code = 0
    manager: Optional[multiprocessing.managers.SyncManager] = None
    progress_queue: Optional[ProgressQueue] = None
    trainer: Optional[CFRTrainer] = None

    try:
        # --- DEBUG PRINT 5 ---
        print("DEBUG: Entering main try block", file=sys.stderr)

        if config.cfr_training.num_workers > 1:
            manager = multiprocessing.Manager()
            progress_queue = manager.Queue(-1)
            print(
                "DEBUG: Multiprocessing manager and queue created", file=sys.stderr
            )  # DEBUG
        else:
            manager = None
            progress_queue = None
            print("DEBUG: Sequential mode - no manager/queue", file=sys.stderr)  # DEBUG

        # Setup logging
        setup_result = setup_logging(config, args.verbose, live_display_manager)
        if not setup_result:
            print("ERROR: Failed to set up logging. Exiting.")
            exit_code = 1
            raise SystemExit(exit_code)
        run_log_dir, run_timestamp = setup_result
        # --- DEBUG PRINT 6 ---
        # Logger might not be fully working if setup failed partially, use print
        print(f"DEBUG: Logging setup complete. Run dir: {run_log_dir}", file=sys.stderr)
        # Now use logger safely
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

        # --- DEBUG PRINT 7 ---
        print("DEBUG: Initializing CFRTrainer...", file=sys.stderr)
        # Initialize trainer
        try:
            trainer = CFRTrainer(
                config=config,
                run_log_dir=run_log_dir,
                run_timestamp=run_timestamp,
                shutdown_event=shutdown_event,
                progress_queue=progress_queue,
                live_display_manager=live_display_manager,
            )
            # --- DEBUG PRINT 8 ---
            print("DEBUG: CFRTrainer initialized successfully.", file=sys.stderr)
        except Exception as trainer_init_e:
            print(
                f"FATAL: Failed to initialize CFRTrainer: {trainer_init_e}",
                file=sys.stderr,
            )  # Use print
            logger.exception("Failed to initialize CFRTrainer:")  # Log exception too
            exit_code = 1
            raise  # Re-raise the exception

        # Load data if requested
        if args.load:
            # --- DEBUG PRINT 9 ---
            print("DEBUG: Loading agent data...", file=sys.stderr)
            logger.info(
                "Attempting to load agent data from: %s",
                config.persistence.agent_data_save_path,
            )
            trainer.load_data()
            if live_display_manager:
                live_display_manager.update_overall_progress(trainer.current_iteration)
                live_display_manager.update_stats(
                    iteration=trainer.current_iteration,
                    infosets=trainer._total_infosets_str,
                    exploitability=trainer._last_exploit_str,
                )
            # --- DEBUG PRINT 10 ---
            print(
                f"DEBUG: Agent data loading complete. Current iteration: {trainer.current_iteration}",
                file=sys.stderr,
            )
        else:
            logger.info("Starting training from scratch.")

        # --- Run training within the Live display context ---
        # --- DEBUG PRINT 11 ---
        print(
            "DEBUG: Calling live_display_manager.run(trainer.train)...", file=sys.stderr
        )
        try:
            if live_display_manager:
                live_display_manager.run(trainer.train)
            else:
                # Fallback if display manager somehow failed init (shouldn't happen)
                print(
                    "WARN: Live display not available, running train directly.",
                    file=sys.stderr,
                )
                trainer.train()

            # --- DEBUG PRINT 12 ---
            # This will only be reached if trainer.train finishes normally
            print("DEBUG: trainer.train() completed.", file=sys.stderr)
            logger.info("Training completed successfully.")
        except (KeyboardInterrupt, GracefulShutdownException) as e:
            # --- DEBUG PRINT 13 ---
            print(
                f"DEBUG: Caught {type(e).__name__} during trainer.train().",
                file=sys.stderr,
            )
            logger.warning(
                "Training interrupted by user (Ctrl+C) or shutdown event (%s).",
                type(e).__name__,
            )
        except Exception as train_exc:
            # --- DEBUG PRINT 14 ---
            print(
                f"DEBUG: Caught Exception {type(train_exc).__name__} during trainer.train().",
                file=sys.stderr,
            )
            logger.exception("An unexpected error occurred during training:")
            exit_code = 1

    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 1
        # --- DEBUG PRINT 15 ---
        print(f"DEBUG: Caught SystemExit with code {exit_code}.", file=sys.stderr)
        if logging.getLogger().hasHandlers():
            logger.error("System exit called with code %s.", exit_code)
    except Exception as e:
        # --- DEBUG PRINT 16 ---
        print(
            f"DEBUG: Caught Exception {type(e).__name__} in outer try block.",
            file=sys.stderr,
        )
        rich_console.print(f"[bold red]FATAL ERROR during setup:[/bold red] {e}")
        rich_console.print_exception(show_locals=True)
        if logging.getLogger().hasHandlers():
            logger.exception("FATAL ERROR during setup:")
        exit_code = 1
    finally:
        # --- DEBUG PRINT 17 ---
        print("DEBUG: Entering main finally block.", file=sys.stderr)
        # --- Clean Shutdown ---
        rich_console.print("--- Initiating Final Shutdown Sequence ---")

        if live_display_manager and live_display_manager.live:
            rich_console.print("Stopping Rich Live display...")
            live_display_manager.stop()

        if manager:
            rich_console.print(
                "Shutting down multiprocessing manager (for progress queue)..."
            )
            try:
                manager.shutdown()
            except Exception as mgr_e:
                rich_console.print(f"[red]Error shutting down manager:[/red] {mgr_e}")
            else:
                rich_console.print("Manager shut down.")  # Print success only if no error

        if trainer:
            rich_console.print("Writing run summary...")
            trainer._write_run_summary()

        rich_console.print("--- Cambia CFR+ Training Finished ---")
        logging.shutdown()
        # --- DEBUG PRINT 18 ---
        print(f"DEBUG: Exiting with code {exit_code}.", file=sys.stderr)
        sys.exit(exit_code)


if __name__ == "__main__":
    # Multiprocessing start method setup remains same
    start_method_set = False
    try:
        preferred_method = "forkserver" if sys.platform != "win32" else "spawn"
        available_methods = multiprocessing.get_all_start_methods()
        current_method = multiprocessing.get_start_method(allow_none=True)
        force_set = current_method is None

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
        elif current_method is None:
            print("ERROR: Neither 'forkserver' nor 'spawn' start methods available!")
    except RuntimeError as e:
        print(f"DEBUG: Multiprocessing start method likely already set? ({e})")
    except Exception as e:
        print(f"ERROR: Error setting multiprocessing start method: {e}")

    effective_method = multiprocessing.get_start_method(allow_none=True)
    print(f"INFO: Effective multiprocessing start method: {effective_method}")

    main()
