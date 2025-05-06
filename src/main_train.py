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
# Global reference to LiveDisplayManager for SIGINT handler
live_display_manager_global: Optional[LiveDisplayManager] = None


def handle_sigint(sig, frame):
    """Signal handler for SIGINT (Ctrl+C)."""
    global live_display_manager_global
    if not shutdown_event.is_set():
        print("\nSIGINT received. Requesting graceful shutdown...", file=sys.stderr)
        # Immediately stop the Rich display to prevent CPU churn
        if live_display_manager_global and live_display_manager_global.live:
            print("Stopping Rich Live display due to SIGINT...", file=sys.stderr)
            try:
                live_display_manager_global.stop()
            except Exception as e:
                print(f"Error stopping Rich Live display in SIGINT: {e}", file=sys.stderr)
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
    global live_display_manager_global

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
    signal.signal(signal.SIGINT, handle_sigint)

    config = load_config(args.config)
    if not config:
        print("ERROR: Failed to load configuration. Exiting.")
        sys.exit(1)

    rich_console = Console(stderr=True, record=False)
    total_iterations_for_display = (
        args.iterations
        if args.iterations is not None
        else config.cfr_training.num_iterations
    )
    live_display_manager_global = LiveDisplayManager(
        num_workers=config.cfr_training.num_workers,
        total_iterations=total_iterations_for_display,
        console=rich_console,
    )

    exit_code = 0
    manager: Optional[multiprocessing.managers.SyncManager] = None
    progress_queue: Optional[ProgressQueue] = None
    trainer: Optional[CFRTrainer] = None
    # Define target_iterations_in_config here for access in finally block
    target_iterations_in_config = config.cfr_training.num_iterations
    if args.iterations is not None:  # Override if CLI arg is present
        target_iterations_in_config = args.iterations

    try:
        if config.cfr_training.num_workers > 1:
            manager = multiprocessing.Manager()
            progress_queue = manager.Queue(-1)
        else:
            manager = None
            progress_queue = None

        setup_result = setup_logging(config, args.verbose, live_display_manager_global)
        if not setup_result:
            print("ERROR: Failed to set up logging. Exiting.")
            exit_code = 1
            raise SystemExit(exit_code)
        run_log_dir, run_timestamp = setup_result

        logger.info("--- Starting Cambia CFR+ Training ---")
        logger.info("Configuration loaded from: %s", args.config)

        if args.iterations is not None:
            config.cfr_training.num_iterations = (
                args.iterations
            )  # This updates the config object
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
        try:
            trainer = CFRTrainer(
                config=config,
                run_log_dir=run_log_dir,
                run_timestamp=run_timestamp,
                shutdown_event=shutdown_event,
                progress_queue=progress_queue,
                live_display_manager=live_display_manager_global,
            )
        except Exception as trainer_init_e:
            print(
                f"FATAL: Failed to initialize CFRTrainer: {trainer_init_e}",
                file=sys.stderr,
            )
            logger.exception("Failed to initialize CFRTrainer:")
            exit_code = 1
            raise

        if args.load:
            logger.info(
                "Attempting to load agent data from: %s",
                config.persistence.agent_data_save_path,
            )
            trainer.load_data()
            if live_display_manager_global:
                live_display_manager_global.update_overall_progress(
                    trainer.current_iteration
                )
                live_display_manager_global.update_stats(
                    iteration=trainer.current_iteration,
                    infosets=trainer._total_infosets_str,
                    exploitability=trainer._last_exploit_str,
                )
        else:
            logger.info("Starting training from scratch.")
        try:
            if live_display_manager_global:
                live_display_manager_global.run(trainer.train)
            else:
                print(
                    "WARN: Live display not available, running train directly.",
                    file=sys.stderr,
                )
                trainer.train()
            logger.info("Training completed successfully.")

        except (KeyboardInterrupt, GracefulShutdownException) as e:
            logger.warning(
                "Training interrupted by user (Ctrl+C) or shutdown event (%s).",
                type(e).__name__,
            )
            if not shutdown_event.is_set():
                shutdown_event.set()
        except Exception as train_exc:
            logger.exception("An unexpected error occurred during training:")
            exit_code = 1

    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 1
        if logging.getLogger().hasHandlers():
            logger.error("System exit called with code %s.", exit_code)
    except Exception as e:
        rich_console.print(f"[bold red]FATAL ERROR during setup:[/bold red] {e}")
        rich_console.print_exception(show_locals=True)
        if logging.getLogger().hasHandlers():
            logger.exception("FATAL ERROR during setup:")
        exit_code = 1
    finally:
        rich_console.print("--- Initiating Final Shutdown Sequence ---")

        if live_display_manager_global and live_display_manager_global.live:
            rich_console.print("Ensuring Rich Live display is stopped...")
            try:
                live_display_manager_global.stop()
            except Exception as e:
                rich_console.print(
                    f"[red]Error stopping Rich display in finally: {e}[/red]"
                )

        if manager:
            rich_console.print(
                "Shutting down multiprocessing manager (for progress queue)..."
            )
            try:
                manager.shutdown()
            except Exception as mgr_e:
                rich_console.print(f"[red]Error shutting down manager:[/red] {mgr_e}")
            else:
                rich_console.print("Manager shut down.")

        if trainer:
            # Check if training completed its full course or was interrupted.
            # trainer.train() now handles its own summary writing on normal completion or SIGINT.
            # This call is a fallback if trainer.train() exited very early due to an exception
            # before it could write its own summary.
            # If trainer.current_iteration reached the target, train() should have written summary.
            # The main reason for this check is if trainer.train() itself failed to complete its try-finally.
            training_fully_completed = (
                trainer.current_iteration >= target_iterations_in_config
            )

            if not training_fully_completed or shutdown_event.is_set():
                # If interrupted or did not complete all target iterations, ensure summary is attempted.
                # trainer.train()'s _perform_emergency_save or normal completion's _write_run_summary should ideally cover this.
                # This is a safeguard.
                if hasattr(trainer, "_write_run_summary") and callable(
                    trainer._write_run_summary
                ):
                    rich_console.print(
                        "Writing run summary (main_train.py finally block)..."
                    )
                    trainer._write_run_summary()
                else:
                    rich_console.print(
                        "Trainer object or summary writer not available in main_train.py finally."
                    )
        else:
            # If trainer itself failed to initialize
            rich_console.print(
                "Trainer object not initialized. Cannot write summary from main_train.py."
            )

        rich_console.print("--- Cambia CFR+ Training Finished ---")
        logging.shutdown()
        sys.exit(exit_code)


if __name__ == "__main__":
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
                # print(f"INFO: Set multiprocessing start method to '{preferred_method}'")
        elif "spawn" in available_methods:
            if force_set or current_method != "spawn":
                multiprocessing.set_start_method("spawn", force=force_set)
                start_method_set = True
                # print(f"INFO: Preferred method '{preferred_method}' not available, using 'spawn'")
        elif current_method is None:
            print(
                "ERROR: Neither 'forkserver' nor 'spawn' start methods available and none set!"
            )
    except RuntimeError:
        pass
    except Exception as e:
        print(f"ERROR: Error setting multiprocessing start method: {e}")

    # effective_method = multiprocessing.get_start_method(allow_none=True)
    # print(f"INFO: Effective multiprocessing start method: {effective_method}")

    main()
