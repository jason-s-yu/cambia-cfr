"""src/main_train.py"""

import logging
import argparse
import os
import datetime
import sys
import signal
import threading
import multiprocessing
import queue
import traceback

from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from rich.console import Console

from .serial_rotating_handler import SerialRotatingFileHandler
from .config import Config, load_config
from .cfr.exceptions import GracefulShutdownException
from .live_display import LiveDisplayManager
from .live_log_handler import LiveLogHandler
from .log_archiver import LogArchiver
from .cfr.trainer import CFRTrainer
from .utils import LogQueue as ProgressQueue

# Use TYPE_CHECKING for imports only needed for type hints
if TYPE_CHECKING:
    from .serial_rotating_handler import (
        SerialRotatingFileHandler,
    )
    from .utils import LogQueue as ProgressQueue
    from .live_display import LiveDisplayManager
    from .log_archiver import LogArchiver
    from .cfr.trainer import CFRTrainer


# Global logger instance
logger = logging.getLogger(__name__)

# Shared event for graceful shutdown
shutdown_event = threading.Event()
# Global references for SIGINT handler and final shutdown
live_display_manager_global: Optional["LiveDisplayManager"] = None
log_archiver_global: Optional["LogArchiver"] = None
archive_queue_global: Optional[Union[queue.Queue, "multiprocessing.Queue"]] = (
    None
)


def handle_sigint(sig, frame):
    """Signal handler for SIGINT (Ctrl+C)."""
    global live_display_manager_global, log_archiver_global
    if not shutdown_event.is_set():
        # Use print for initial signal handling as logging might be unreliable during shutdown
        print("\nSIGINT received. Requesting graceful shutdown...", file=sys.stderr)
        # Immediately stop the Rich display if it exists and is running
        if live_display_manager_global and getattr(
            live_display_manager_global, "live", None
        ):
            print("Stopping Rich Live display due to SIGINT...", file=sys.stderr)
            try:
                live_display_manager_global.stop()
            except Exception as e_stop_rich:
                print(
                    f"Error stopping Rich Live display in SIGINT: {e_stop_rich}",
                    file=sys.stderr,
                )

        # Signal main shutdown event FIRST
        shutdown_event.set()

        # Signal LogArchiver to stop (it will process queue on its own thread)
        # No need to call stop() here, finally block handles it.
        # if log_archiver_global:
        #     print("Signaling LogArchiver to stop...", file=sys.stderr)
        # log_archiver_global.stop() # Moved to finally
    else:
        print(
            "\nMultiple SIGINT received. Shutdown already in progress.", file=sys.stderr
        )


def setup_logging(
    config: Config,
    verbose: bool,  # Keep verbose flag? Currently sets console level below.
    live_display_manager: "LiveDisplayManager",
    archive_q: Optional[Union[queue.Queue, "multiprocessing.Queue"]],
) -> Optional[Tuple[str, str]]:
    """Configures logging, integrating with LiveDisplayManager and archive queue."""
    try:
        log_level_file_str = config.logging.log_level_file.upper()
        log_level_console_str = config.logging.log_level_console.upper()

        file_log_level_value = getattr(logging, log_level_file_str, logging.DEBUG)
        # Effective console level controls what the LiveLogHandler forwards
        effective_console_log_level_value = getattr(
            logging, log_level_console_str, logging.ERROR
        )

        main_log_dir = config.logging.log_dir
        log_prefix = config.logging.log_file_prefix

        if not main_log_dir or not isinstance(main_log_dir, str):
            print(f"ERROR: Invalid log directory '{main_log_dir}'. Logging disabled.")
            return None

        # Create Run Directory
        run_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
        run_log_dir = os.path.join(main_log_dir, f"{log_prefix}_run_{run_timestamp}")
        os.makedirs(run_log_dir, exist_ok=True)  # Raises OSError on failure

        # Create Handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)-8s - [%(processName)-20s] - %(name)-25s - %(message)s"
        )
        handlers: List[logging.Handler] = []

        # Live Handler (forwards based on its level)
        live_handler = LiveLogHandler(
            live_display_manager, level=effective_console_log_level_value
        )
        handlers.append(live_handler)

        # File Handler
        main_log_pattern = os.path.join(
            run_log_dir, f"{log_prefix}_run_{run_timestamp}-main"
        )
        fh = SerialRotatingFileHandler(
            main_log_pattern,
            maxBytes=config.logging.log_max_bytes,
            backupCount=config.logging.log_backup_count,
            encoding="utf-8",
            archive_queue=archive_q,
            logging_config_snapshot=config.logging,
        )
        fh.setLevel(file_log_level_value)
        fh.setFormatter(formatter)
        handlers.append(fh)
        main_log_file = fh.baseFilename

        # Configure Root Logger
        root_logger = logging.getLogger()
        # Remove existing handlers first
        for handler in root_logger.handlers[:]:
            try:
                root_logger.removeHandler(handler)
                if hasattr(handler, "close"):
                    handler.close()
            except Exception:
                pass  # Ignore errors closing old handlers
        # Set level and add new handlers
        root_logger.setLevel(logging.DEBUG)  # Set root low, handlers control output
        for handler in handlers:
            root_logger.addHandler(handler)

        # Initial Log Messages (use root logger now it's configured)
        logging.info("-" * 50)
        logging.info("Logging initialized for run: %s", run_timestamp)
        logging.info("Run Log Directory: %s", run_log_dir)
        logging.info(
            "Main Log File: %s (Level: %s)",
            main_log_file,
            logging.getLevelName(file_log_level_value),
        )
        logging.info(
            "Console via Rich (Handler Level: %s, Rich Filter Level: %s)",
            logging.getLevelName(live_handler.level),
            logging.getLevelName(
                getattr(live_display_manager, "console_log_level_value", logging.ERROR)
            ),  # Get effective level used by Rich
        )
        logging.info("Command: %s", " ".join(sys.argv))
        logging.info("-" * 50)

        # Update latest_run link/marker
        latest_log_link_path = os.path.join(main_log_dir, "latest_run")
        try:
            absolute_run_log_dir = os.path.abspath(run_log_dir)
            if sys.platform == "win32":
                marker_path = latest_log_link_path + ".txt"
                if os.path.exists(marker_path):
                    os.remove(marker_path)
                with open(marker_path, "w", encoding="utf-8") as f_marker:
                    f_marker.write(f"Latest run directory: {absolute_run_log_dir}\n")
                logging.info("Updated latest run marker file: %s", marker_path)
            else:
                if os.path.lexists(latest_log_link_path):
                    os.remove(latest_log_link_path)
                os.symlink(
                    absolute_run_log_dir, latest_log_link_path, target_is_directory=True
                )
                logging.info("Updated latest_run symlink -> %s", absolute_run_log_dir)
        except OSError as e_link:
            logging.error("Could not create/update latest_run link/marker: %s", e_link)
        except Exception as e_link_other:  # Catch other potential errors
            logging.error(
                "Unexpected error updating latest_run link/marker: %s",
                e_link_other,
                exc_info=True,
            )

        # Reduce Verbosity from Libraries
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("joblib").setLevel(logging.WARNING)
        # Reduce matplotlib font manager noise if it occurs
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

        return run_log_dir, run_timestamp

    # Use more specific exception types if setup fails critically
    except (OSError, IOError, ValueError, AttributeError) as e_setup:
        print(f"FATAL: Logging setup failed: {e_setup}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None
    except Exception as e_setup_other:  # Catch any other unexpected errors
        print(
            f"FATAL: Unexpected error during logging setup: {e_setup_other}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return None


def main():
    global live_display_manager_global, log_archiver_global, archive_queue_global

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
    args = parser.parse_args()

    # Set up signal handler early
    signal.signal(signal.SIGINT, handle_sigint)

    # Load config first, needed for logging setup
    config = load_config(args.config)
    if not config:
        print("ERROR: Failed to load configuration. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Initialize Rich Console and Display Manager (needed for logging handler)
    rich_console = Console(stderr=True, record=False)
    total_iterations_for_display = (
        args.iterations
        if args.iterations is not None
        else getattr(config.cfr_training, "num_iterations", 0)
    )
    console_log_level_str = getattr(config.logging, "log_level_console", "ERROR").upper()
    console_log_level_value_for_display = getattr(
        logging, console_log_level_str, logging.ERROR
    )

    # Basic check on num_workers before creating display manager
    num_workers_for_display = getattr(config.cfr_training, "num_workers", 1)
    if not isinstance(num_workers_for_display, int) or num_workers_for_display < 0:
        print(
            f"ERROR: Invalid num_workers configured: {num_workers_for_display}. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    live_display_manager_global = LiveDisplayManager(
        num_workers=num_workers_for_display,
        total_iterations=total_iterations_for_display,
        console=rich_console,
        console_log_level_value=console_log_level_value_for_display,
    )

    exit_code = 0
    manager: Optional[multiprocessing.managers.SyncManager] = None
    progress_queue: Optional[ProgressQueue] = None
    trainer: Optional[CFRTrainer] = None
    # Target iterations respects command-line override
    target_iterations = (
        args.iterations
        if args.iterations is not None
        else getattr(config.cfr_training, "num_iterations", 0)
    )

    # Determine if multiprocessing Manager is needed for archive queue
    needs_mp_manager_for_archive = num_workers_for_display > 1 and getattr(
        config.logging, "log_archive_enabled", False
    )

    try:
        # Set up multiprocessing manager and queues if needed
        if num_workers_for_display > 1:
            manager = multiprocessing.Manager()
            progress_queue = manager.Queue(-1)
            if needs_mp_manager_for_archive:
                archive_queue_global = manager.Queue(-1)
            else:
                archive_queue_global = queue.Queue(
                    -1
                )  # Use threading queue if archiver runs in main process thread
        else:  # Single worker or no archiving
            progress_queue = (
                None  # Single worker updates directly if needed, or no progress queue
            )
            archive_queue_global = queue.Queue(
                -1
            )  # Use threading queue if archiving enabled

        # Start LogArchiver thread if enabled
        if getattr(config.logging, "log_archive_enabled", False):
            if (
                archive_queue_global is None
            ):  # Should not happen if logic above is correct
                print(
                    "ERROR: Log archiving enabled but archive queue not initialized.",
                    file=sys.stderr,
                )
                sys.exit(1)
            log_archiver_global = LogArchiver(
                config, archive_queue_global, ""
            )  # run_log_dir set later
            log_archiver_global.start()

        # Setup logging (requires config, display manager, archive queue)
        # Pass False for verbose flag, let config control levels
        setup_result = setup_logging(
            config, False, live_display_manager_global, archive_queue_global
        )
        if not setup_result:
            print("ERROR: Failed to set up logging. Exiting.", file=sys.stderr)
            exit_code = 1
            raise SystemExit(exit_code)  # Use SystemExit for controlled exit
        run_log_dir, run_timestamp = setup_result

        # Update LogArchiver with the actual run_log_dir
        if log_archiver_global:
            log_archiver_global.run_log_dir = run_log_dir
            logging.info("LogArchiver run_log_dir updated to: %s", run_log_dir)

        logging.info("--- Starting Cambia CFR+ Training ---")
        logging.info("Configuration loaded from: %s", args.config)

        # Override config based on command line args AFTER initial logging setup
        if args.iterations is not None:
            config.cfr_training.num_iterations = args.iterations
        if args.save_path is not None:
            config.persistence.agent_data_save_path = args.save_path
        if args.iterations is not None or args.save_path is not None:
            logging.info(
                "Applied command-line overrides (Iterations: %s, Save Path: %s)",
                args.iterations,
                args.save_path,
            )

        # Set recursion limit
        if getattr(config.system, "recursion_limit", 0) > 0:
            try:
                sys.setrecursionlimit(config.system.recursion_limit)
                logging.info("System recursion limit set to: %d", sys.getrecursionlimit())
            except (ValueError, RecursionError) as e_recur:  # Catch specific errors
                logging.error(
                    "Failed to set recursion limit to %d: %s",
                    config.system.recursion_limit,
                    e_recur,
                )

        # Initialize Trainer
        try:
            trainer = CFRTrainer(
                config=config,
                run_log_dir=run_log_dir,
                run_timestamp=run_timestamp,
                shutdown_event=shutdown_event,
                progress_queue=progress_queue,
                live_display_manager=live_display_manager_global,
                archive_queue=archive_queue_global,
            )
            # Pass global archiver ref to trainer if it exists
            if trainer and log_archiver_global:
                trainer.log_archiver_global_ref = log_archiver_global
        except Exception as trainer_init_e:  # Catch specific init errors if possible
            # Log exception before raising
            logging.exception(
                "FATAL: Failed to initialize CFRTrainer: %s", trainer_init_e
            )
            # Print to stderr as logging might be compromised
            print(
                f"FATAL: Failed to initialize CFRTrainer: {trainer_init_e}",
                file=sys.stderr,
            )
            exit_code = 1
            raise  # Re-raise after logging

        # Load data if requested
        if args.load:
            logger.info(
                "Attempting to load agent data from: %s",
                config.persistence.agent_data_save_path,
            )
            if hasattr(trainer, "load_data") and callable(trainer.load_data):
                trainer.load_data()
                if live_display_manager_global:  # Update display with loaded state
                    live_display_manager_global.update_overall_progress(
                        trainer.current_iteration
                    )
                    live_display_manager_global.update_stats(
                        iteration=trainer.current_iteration,
                        infosets=trainer._total_infosets_str,
                        exploitability=trainer._last_exploit_str,
                    )
            else:
                logger.error("Trainer object missing load_data method.")
        else:
            logger.info("Starting training from scratch.")

        # Initial log size update
        if log_archiver_global and live_display_manager_global:
            try:
                current_size, archived_size = (
                    log_archiver_global.get_total_log_size_info()
                )
                live_display_manager_global.update_log_summary_display(
                    current_size, archived_size
                )
            except Exception as e_log_size_init:
                logger.error("Error during initial log size display: %s", e_log_size_init)

        # --- Main Training Execution ---
        training_completed_normally = False
        try:
            if live_display_manager_global:
                # Run the trainer's train method within the Live context
                live_display_manager_global.run(trainer.train)
            else:
                # Run directly if no display manager
                print(
                    "WARN: Live display not available, running train directly.",
                    file=sys.stderr,
                )
                if hasattr(trainer, "train") and callable(trainer.train):
                    trainer.train()  # Blocking call
                else:
                    logger.error("Trainer object missing train method.")
                    raise RuntimeError("Trainer cannot be executed.")
            # If train completes without exception, mark as normal completion
            training_completed_normally = True
            logger.info("Training completed successfully.")

        except (
            GracefulShutdownException
        ) as shutdown_exc:  # Catch specific shutdown signal
            logger.warning("Training interrupted by shutdown signal: %s.", shutdown_exc)
            # Emergency save handled by trainer's internal logic or finally block
            exit_code = 0  # Indicate graceful exit
        except KeyboardInterrupt:  # Should be caught by signal handler, but as fallback
            logger.warning(
                "KeyboardInterrupt caught directly in main. Requesting shutdown."
            )
            if not shutdown_event.is_set():
                shutdown_event.set()
            # Emergency save handled by finally block
            exit_code = 0
        except Exception as train_exc:  # Catch other unexpected training errors
            logger.exception("An unexpected error occurred during training:")
            # Emergency save handled by finally block
            exit_code = 1  # Indicate error exit

    except SystemExit as sys_exit:  # Catch explicit sys.exit calls
        exit_code = sys_exit.code if isinstance(sys_exit.code, int) else 1
        # Log only if logger is still functional
        if logging.getLogger().hasHandlers():
            logger.error("System exit called with code %s.", exit_code)
        else:
            print(f"System exit called with code {exit_code}.", file=sys.stderr)
    except (
        Exception
    ) as setup_exc:  # Catch errors during setup phase (before main training try block)
        # Use print as logging might not be set up
        print(f"\n--- FATAL ERROR during setup ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        rich_console.print(f"[bold red]FATAL ERROR during setup:[/bold red] {setup_exc}")
        # Log if possible
        if logging.getLogger().hasHandlers():
            logger.exception("FATAL ERROR during setup:")
        exit_code = 1

    # --- Final Shutdown Sequence ---
    finally:
        is_shutting_down_gracefully = (
            shutdown_event.is_set() or training_completed_normally
        )
        shutdown_reason = (
            "Normal Completion"
            if training_completed_normally
            else (
                "Graceful Shutdown" if is_shutting_down_gracefully else "Error/Exception"
            )
        )
        rich_console.print(
            f"\n--- Initiating Final Shutdown Sequence ({shutdown_reason}) ---"
        )

        # Final log size update attempt
        if log_archiver_global and live_display_manager_global:
            try:
                current_size, archived_size = (
                    log_archiver_global.get_total_log_size_info()
                )
                # Check methods exist before calling
                if hasattr(live_display_manager_global, "update_log_summary_display"):
                    live_display_manager_global.update_log_summary_display(
                        current_size, archived_size
                    )
                # Refresh if live display might still be active
                if getattr(live_display_manager_global, "live", None):
                    if hasattr(live_display_manager_global, "refresh"):
                        live_display_manager_global.refresh()
            except Exception as final_log_size_e:
                rich_console.print(
                    f"[yellow]Could not perform final log size update: {final_log_size_e}[/yellow]"
                )

        # Ensure Rich Live display is stopped
        if live_display_manager_global and getattr(
            live_display_manager_global, "live", None
        ):
            rich_console.print("Ensuring Rich Live display is stopped...")
            try:
                if hasattr(live_display_manager_global, "stop"):
                    live_display_manager_global.stop()
            except Exception as e_stop_rich_final:
                rich_console.print(
                    f"[red]Error stopping Rich display in finally: {e_stop_rich_final}[/red]"
                )

        # Perform final save/summary via trainer if needed (e.g., normal exit)
        # Emergency save should have been handled in exception blocks or by trainer internally
        if trainer and training_completed_normally and not is_shutting_down_gracefully:
            # If completed normally without shutdown signal, perform final save/summary
            rich_console.print("Performing final save and summary...")
            try:
                if hasattr(trainer, "save_data") and callable(trainer.save_data):
                    trainer.save_data()
                if hasattr(trainer, "_write_run_summary") and callable(
                    trainer._write_run_summary
                ):
                    trainer._write_run_summary()
            except Exception as e_final_save:
                rich_console.print(
                    f"[red]Error during final save/summary: {e_final_save}[/red]"
                )
        elif not training_completed_normally and trainer:
            # If exited due to error/shutdown, summary should have been written by _perform_emergency_save
            rich_console.print(
                "Shutdown/Error occurred, final summary likely written by emergency handler."
            )
        elif not trainer:
            rich_console.print(
                "Trainer object not initialized. Cannot perform final save/summary."
            )

        # Stop LogArchiver (allow time to process queue)
        if log_archiver_global:
            rich_console.print("Stopping LogArchiver...")
            try:
                if hasattr(log_archiver_global, "stop"):
                    log_archiver_global.stop(timeout=10.0)
                    rich_console.print("LogArchiver stopped.")
                else:
                    rich_console.print(
                        "[yellow]LogArchiver missing stop method.[/yellow]"
                    )
            except Exception as e_stop_archiver:
                rich_console.print(
                    f"[red]Error stopping LogArchiver: {e_stop_archiver}[/red]"
                )

        # Shutdown multiprocessing manager (if used)
        if manager:
            rich_console.print("Shutting down multiprocessing manager...")
            try:
                manager.shutdown()
            except Exception as mgr_e:
                rich_console.print(f"[red]Error shutting down manager:[/red] {mgr_e}")
            else:
                rich_console.print("Manager shut down.")

        rich_console.print("--- Cambia CFR+ Training Finished ---")

        # Final logging shutdown
        logging.shutdown()
        sys.exit(exit_code)


if __name__ == "__main__":
    # Set multiprocessing start method (important for stability, especially on non-Windows)
    start_method_set = False
    try:
        preferred_method = "forkserver" if sys.platform != "win32" else "spawn"
        available_methods = multiprocessing.get_all_start_methods()
        current_method = multiprocessing.get_start_method(allow_none=True)

        method_to_set = None
        if preferred_method in available_methods:
            method_to_set = preferred_method
        elif "spawn" in available_methods:
            method_to_set = "spawn"
        # Add 'fork' as a fallback if others aren't available (less safe with threads)
        elif "fork" in available_methods and sys.platform != "win32":
            method_to_set = "fork"

        if method_to_set and (current_method is None or current_method != method_to_set):
            # Force only if no method is currently set
            force_set = current_method is None
            multiprocessing.set_start_method(method_to_set, force=force_set)
            start_method_set = True
            print(
                f"INFO: Multiprocessing start method set to '{method_to_set}'.",
                file=sys.stderr,
            )
        elif current_method:
            print(
                f"INFO: Multiprocessing start method already set to '{current_method}'.",
                file=sys.stderr,
            )
        elif not method_to_set:
            print(
                "ERROR: No suitable multiprocessing start method found!", file=sys.stderr
            )

    except RuntimeError:  # Context already set
        # print("DEBUG: Multiprocessing context already set.", file=sys.stderr)
        pass
    except Exception as e_mp_start:
        print(
            f"ERROR: Setting multiprocessing start method failed: {e_mp_start}",
            file=sys.stderr,
        )

    # Execute main function
    main()
