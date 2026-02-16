"""src/main_train.py"""

import logging
import argparse
import os
import datetime
import sys
import signal
import threading
import multiprocessing
import multiprocessing.managers
import queue
import traceback
from dataclasses import dataclass

from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from rich.console import Console

from .serial_rotating_handler import SerialRotatingFileHandler
from .config import Config, load_config
from .cfr.exceptions import GracefulShutdownException, ConfigParseError, ConfigValidationError
from .live_display import LiveDisplayManager
from .live_log_handler import LiveLogHandler
from .log_archiver import LogArchiver
from .cfr.trainer import CFRTrainer
from .cfr.deep_trainer import DeepCFRTrainer, DeepCFRConfig
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
    from .cfr.deep_trainer import DeepCFRTrainer


# Global logger instance
logger = logging.getLogger(__name__)

# Shared event for graceful shutdown
shutdown_event = threading.Event()
# Global references for SIGINT handler and final shutdown
live_display_manager_global: Optional["LiveDisplayManager"] = None
log_archiver_global: Optional["LogArchiver"] = None
archive_queue_global: Optional[Union[queue.Queue, "multiprocessing.Queue"]] = None


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


@dataclass
class TrainingInfrastructure:
    """Container for training infrastructure components."""
    live_display_manager: LiveDisplayManager
    progress_queue: Optional[ProgressQueue]
    archive_queue: Optional[Union[queue.Queue, "multiprocessing.Queue"]]
    manager: Optional[multiprocessing.managers.SyncManager]
    log_archiver: Optional[LogArchiver]
    run_log_dir: str
    run_timestamp: str


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
            except Exception:  # JUSTIFIED: Cleaning up old handlers during reconfiguration; errors don't affect new setup
                logger.debug("Error closing old handler during cleanup")  # Use logger if available, else silent
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
        except Exception as e_link_other:  # JUSTIFIED: Symlink creation is convenience feature; must not crash logging setup
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
    except Exception as e_setup_other:  # JUSTIFIED: Top-level setup handler; logging not yet functional; print to stderr
        print(
            f"FATAL: Unexpected error during logging setup: {e_setup_other}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return None


def create_infrastructure(
    config: Config,
    total_iterations: int,
    console_log_level: Optional[int] = None,
) -> TrainingInfrastructure:
    """
    Create all training infrastructure: display, queues, archiver, logging.

    Args:
        config: Configuration object
        total_iterations: Total iterations for progress display
        console_log_level: Optional override for console log level

    Returns:
        TrainingInfrastructure with all components initialized
    """
    global live_display_manager_global, log_archiver_global, archive_queue_global

    # Initialize Rich Console and Display Manager
    rich_console = Console(stderr=True, record=False)
    console_log_level_str = getattr(config.logging, "log_level_console", "ERROR").upper()
    console_log_level_value = console_log_level or getattr(
        logging, console_log_level_str, logging.ERROR
    )

    # Get num_workers from config
    num_workers = getattr(config.cfr_training, "num_workers", 1)
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError(f"Invalid num_workers configured: {num_workers}")

    # Create LiveDisplayManager
    live_display_manager = LiveDisplayManager(
        num_workers=num_workers,
        total_iterations=total_iterations,
        console=rich_console,
        console_log_level_value=console_log_level_value,
    )
    live_display_manager_global = live_display_manager

    # Create queues and manager based on configuration
    manager: Optional[multiprocessing.managers.SyncManager] = None
    progress_queue: Optional[ProgressQueue] = None
    archive_queue: Optional[Union[queue.Queue, "multiprocessing.Queue"]] = None

    needs_manager_for_queues = num_workers > 1
    is_archiving_enabled = getattr(config.logging, "log_archive_enabled", False)

    if needs_manager_for_queues:
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue(-1)
        if is_archiving_enabled:
            archive_queue = manager.Queue(-1)
    else:
        progress_queue = None
        if is_archiving_enabled:
            archive_queue = queue.Queue(-1)

    archive_queue_global = archive_queue

    # Start LogArchiver thread if enabled
    log_archiver: Optional[LogArchiver] = None
    if is_archiving_enabled:
        if archive_queue is None:
            raise RuntimeError("Log archiving enabled but archive queue is None")
        log_archiver = LogArchiver(config, archive_queue, "")
        log_archiver.start()

    log_archiver_global = log_archiver

    # Setup logging
    setup_result = setup_logging(config, False, live_display_manager, archive_queue)
    if not setup_result:
        raise RuntimeError("Failed to set up logging")

    run_log_dir, run_timestamp = setup_result

    # Update LogArchiver with the actual run_log_dir
    if log_archiver:
        log_archiver.run_log_dir = run_log_dir
        logging.info("LogArchiver run_log_dir updated to: %s", run_log_dir)

    return TrainingInfrastructure(
        live_display_manager=live_display_manager,
        progress_queue=progress_queue,
        archive_queue=archive_queue,
        manager=manager,
        log_archiver=log_archiver,
        run_log_dir=run_log_dir,
        run_timestamp=run_timestamp,
    )


def shutdown_infrastructure(
    infra: TrainingInfrastructure,
    trainer: Optional[Union[CFRTrainer, DeepCFRTrainer]],
    training_completed_normally: bool,
):
    """
    Shutdown infrastructure components and perform final cleanup.

    Args:
        infra: TrainingInfrastructure instance
        trainer: Trainer instance (CFRTrainer or DeepCFRTrainer)
        training_completed_normally: True if training completed without errors
    """
    rich_console = Console(stderr=True, record=False)
    is_shutting_down_gracefully = shutdown_event.is_set() or training_completed_normally
    shutdown_reason = (
        "Normal Completion"
        if training_completed_normally
        else ("Graceful Shutdown" if is_shutting_down_gracefully else "Error/Exception")
    )
    rich_console.print(
        f"\n--- Initiating Final Shutdown Sequence ({shutdown_reason}) ---"
    )

    # Final log size update attempt
    if infra.log_archiver and infra.live_display_manager:
        try:
            current_size, archived_size = infra.log_archiver.get_total_log_size_info()
            if hasattr(infra.live_display_manager, "update_log_summary_display"):
                infra.live_display_manager.update_log_summary_display(
                    current_size, archived_size
                )
            if getattr(infra.live_display_manager, "live", None):
                if hasattr(infra.live_display_manager, "refresh"):
                    infra.live_display_manager.refresh()
        except Exception as final_log_size_e:
            rich_console.print(
                f"[yellow]Could not perform final log size update: {final_log_size_e}[/yellow]"
            )

    # Ensure Rich Live display is stopped
    if infra.live_display_manager and getattr(infra.live_display_manager, "live", None):
        rich_console.print("Ensuring Rich Live display is stopped...")
        try:
            if hasattr(infra.live_display_manager, "stop"):
                infra.live_display_manager.stop()
        except Exception as e_stop_rich_final:
            rich_console.print(
                f"[red]Error stopping Rich display in finally: {e_stop_rich_final}[/red]"
            )

    # Perform final save/summary via trainer if needed
    if trainer and training_completed_normally:
        if not shutdown_event.is_set():
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
        else:
            rich_console.print(
                "Shutdown occurred during final steps. Skipping final save/summary."
            )
    elif not training_completed_normally and trainer:
        rich_console.print(
            "Shutdown/Error occurred, final summary likely written by emergency handler."
        )
    elif not trainer:
        rich_console.print(
            "Trainer object not initialized. Cannot perform final save/summary."
        )

    # Stop LogArchiver
    if infra.log_archiver:
        rich_console.print("Stopping LogArchiver...")
        try:
            if hasattr(infra.log_archiver, "stop"):
                infra.log_archiver.stop(timeout=10.0)
                rich_console.print("LogArchiver stopped.")
            else:
                rich_console.print("[yellow]LogArchiver missing stop method.[/yellow]")
        except Exception as e_stop_archiver:
            rich_console.print(
                f"[red]Error stopping LogArchiver: {e_stop_archiver}[/red]"
            )

    # Shutdown multiprocessing manager
    if infra.manager:
        rich_console.print("Shutting down multiprocessing manager...")
        try:
            infra.manager.shutdown()
        except Exception as mgr_e:
            rich_console.print(f"[red]Error shutting down manager:[/red] {mgr_e}")
        else:
            rich_console.print("Manager shut down.")

    rich_console.print("--- Cambia CFR+ Training Finished ---")
    logging.shutdown()


def run_tabular_training(
    config: Config,
    infra: TrainingInfrastructure,
    iterations: Optional[int] = None,
    load: bool = False,
    save_path: Optional[str] = None,
) -> int:
    """
    Run tabular CFR+ training.

    Args:
        config: Configuration object
        infra: TrainingInfrastructure instance
        iterations: Override number of iterations (None = use config)
        load: Load existing agent data before training
        save_path: Override save path for agent data

    Returns:
        Exit code (0 for success, 1 for error)
    """
    exit_code = 0
    trainer: Optional[CFRTrainer] = None
    training_completed_normally = False

    try:
        logging.info("--- Starting Cambia CFR+ Training ---")

        # Apply CLI overrides
        if iterations is not None:
            config.cfr_training.num_iterations = iterations
        if save_path is not None:
            config.persistence.agent_data_save_path = save_path
        if iterations is not None or save_path is not None:
            logging.info(
                "Applied command-line overrides (Iterations: %s, Save Path: %s)",
                iterations,
                save_path,
            )

        # Set recursion limit
        if getattr(config.system, "recursion_limit", 0) > 0:
            try:
                sys.setrecursionlimit(config.system.recursion_limit)
                logging.info("System recursion limit set to: %d", sys.getrecursionlimit())
            except (ValueError, RecursionError) as e_recur:
                logging.error(
                    "Failed to set recursion limit to %d: %s",
                    config.system.recursion_limit,
                    e_recur,
                )

        # Initialize Trainer
        trainer = CFRTrainer(
            config=config,
            run_log_dir=infra.run_log_dir,
            run_timestamp=infra.run_timestamp,
            shutdown_event=shutdown_event,
            progress_queue=infra.progress_queue,
            live_display_manager=infra.live_display_manager,
            archive_queue=infra.archive_queue,
        )
        if trainer and infra.log_archiver:
            trainer.log_archiver_global_ref = infra.log_archiver

        # Load data if requested
        if load:
            logger.info(
                "Attempting to load agent data from: %s",
                config.persistence.agent_data_save_path,
            )
            if hasattr(trainer, "load_data") and callable(trainer.load_data):
                trainer.load_data()
                if infra.live_display_manager:
                    infra.live_display_manager.update_overall_progress(
                        trainer.current_iteration
                    )
                    infra.live_display_manager.update_stats(
                        iteration=trainer.current_iteration,
                        infosets=trainer._total_infosets_str,
                        exploitability=trainer._last_exploit_str,
                    )
            else:
                logger.error("Trainer object missing load_data method.")
        else:
            logger.info("Starting training from scratch.")

        # Initial log size update
        if infra.log_archiver and infra.live_display_manager:
            try:
                current_size, archived_size = infra.log_archiver.get_total_log_size_info()
                infra.live_display_manager.update_log_summary_display(
                    current_size, archived_size
                )
            except Exception as e_log_size_init:
                logger.debug("Error during initial log size display: %s", e_log_size_init)

        # Main Training Execution
        if infra.live_display_manager:
            infra.live_display_manager.run(trainer.train)
        else:
            if hasattr(trainer, "train") and callable(trainer.train):
                trainer.train()
            else:
                logger.error("Trainer object missing train method.")
                raise RuntimeError("Trainer cannot be executed.")

        training_completed_normally = True
        logger.info("Training completed successfully.")

    except GracefulShutdownException as shutdown_exc:
        logger.warning("Training interrupted by shutdown signal: %s.", shutdown_exc)
        exit_code = 0
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt caught directly in training.")
        if not shutdown_event.is_set():
            shutdown_event.set()
        exit_code = 0
    except Exception as train_exc:
        logger.exception("An unexpected error occurred during training:")
        exit_code = 1
    finally:
        shutdown_infrastructure(infra, trainer, training_completed_normally)

    return exit_code


def run_deep_training(
    config: Config,
    dcfr_config: DeepCFRConfig,
    infra: TrainingInfrastructure,
    steps: Optional[int] = None,
    checkpoint: Optional[str] = None,
    save_path: Optional[str] = None,
) -> int:
    """
    Run Deep CFR training.

    Args:
        config: Configuration object
        dcfr_config: DeepCFRConfig instance
        infra: TrainingInfrastructure instance
        steps: Number of training steps (None = use config default)
        checkpoint: Path to checkpoint to resume from
        save_path: Override save path for checkpoints

    Returns:
        Exit code (0 for success, 1 for error)
    """
    exit_code = 0
    trainer: Optional[DeepCFRTrainer] = None
    training_completed_normally = False

    try:
        logging.info("--- Starting Cambia Deep CFR Training ---")

        # Apply CLI overrides
        if save_path is not None:
            config.persistence.agent_data_save_path = save_path
            logging.info("Applied command-line override: Save Path = %s", save_path)

        # Set recursion limit
        if getattr(config.system, "recursion_limit", 0) > 0:
            try:
                sys.setrecursionlimit(config.system.recursion_limit)
                logging.info("System recursion limit set to: %d", sys.getrecursionlimit())
            except (ValueError, RecursionError) as e_recur:
                logging.error(
                    "Failed to set recursion limit to %d: %s",
                    config.system.recursion_limit,
                    e_recur,
                )

        # Initialize DeepCFRTrainer
        trainer = DeepCFRTrainer(
            config=config,
            deep_cfr_config=dcfr_config,
            run_log_dir=infra.run_log_dir,
            run_timestamp=infra.run_timestamp,
            shutdown_event=shutdown_event,
            progress_queue=infra.progress_queue,
            live_display_manager=infra.live_display_manager,
            archive_queue=infra.archive_queue,
        )
        if trainer and infra.log_archiver:
            trainer.log_archiver_global_ref = infra.log_archiver

        # Load checkpoint if requested
        if checkpoint:
            logger.info("Attempting to load checkpoint from: %s", checkpoint)
            trainer.load_checkpoint(checkpoint)
        else:
            logger.info("Starting training from scratch.")

        # Initial log size update
        if infra.log_archiver and infra.live_display_manager:
            try:
                current_size, archived_size = infra.log_archiver.get_total_log_size_info()
                infra.live_display_manager.update_log_summary_display(
                    current_size, archived_size
                )
            except Exception as e_log_size_init:
                logger.debug("Error during initial log size display: %s", e_log_size_init)

        # Main Training Execution
        if infra.live_display_manager:
            infra.live_display_manager.run(
                lambda: trainer.train(num_training_steps=steps)
            )
        else:
            if hasattr(trainer, "train") and callable(trainer.train):
                trainer.train(num_training_steps=steps)
            else:
                logger.error("Trainer object missing train method.")
                raise RuntimeError("Trainer cannot be executed.")

        training_completed_normally = True
        logger.info("Deep CFR training completed successfully.")

    except GracefulShutdownException as shutdown_exc:
        logger.warning("Training interrupted by shutdown signal: %s.", shutdown_exc)
        exit_code = 0
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt caught directly in deep training.")
        if not shutdown_event.is_set():
            shutdown_event.set()
        exit_code = 0
    except Exception as train_exc:
        logger.exception("An unexpected error occurred during deep training:")
        exit_code = 1
    finally:
        shutdown_infrastructure(infra, trainer, training_completed_normally)

    return exit_code


def get_checkpoint_info(filepath: str) -> dict:
    """
    Load and return checkpoint metadata.

    Args:
        filepath: Path to checkpoint file (.pt for Deep CFR, .joblib for tabular)

    Returns:
        Dictionary with checkpoint metadata
    """
    import torch

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    if filepath.endswith(".pt"):
        # Deep CFR checkpoint
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        return {
            "type": "deep_cfr",
            "training_step": checkpoint.get("training_step", 0),
            "total_traversals": checkpoint.get("total_traversals", 0),
            "current_iteration": checkpoint.get("current_iteration", 0),
            "advantage_buffer_path": checkpoint.get("advantage_buffer_path", ""),
            "strategy_buffer_path": checkpoint.get("strategy_buffer_path", ""),
            "advantage_loss_history": checkpoint.get("advantage_loss_history", []),
            "strategy_loss_history": checkpoint.get("strategy_loss_history", []),
            "config": checkpoint.get("dcfr_config", {}),
        }
    elif filepath.endswith(".joblib"):
        # Tabular CFR checkpoint
        from .cfr import persistence
        data = persistence.load_agent_data(filepath)
        return {
            "type": "tabular_cfr",
            "iteration": data.get("iteration", 0),
            "num_infosets": len(data.get("regret_sum", {})),
            "exploitability_history": data.get("exploitability_history", []),
        }
    else:
        raise ValueError(f"Unknown checkpoint format: {filepath}")


def main():
    """Main entry point for backward compatibility."""
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

    # Load config
    config = load_config(args.config)
    if not config:
        print("ERROR: Failed to load configuration. Exiting.", file=sys.stderr)
        sys.exit(1)

    logging.info("Configuration loaded from: %s", args.config)

    # Determine total iterations for display
    total_iterations = (
        args.iterations
        if args.iterations is not None
        else getattr(config.cfr_training, "num_iterations", 0)
    )

    try:
        # Create infrastructure
        infra = create_infrastructure(config, total_iterations)

        # Run tabular training
        exit_code = run_tabular_training(
            config, infra, iterations=args.iterations, load=args.load, save_path=args.save_path
        )

        sys.exit(exit_code)

    except Exception as e:
        print(f"FATAL: Error during setup or training: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


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

    except RuntimeError:  # JUSTIFIED: Context already set is expected condition; silent is appropriate
        # print("DEBUG: Multiprocessing context already set.", file=sys.stderr)
        logger.debug("Multiprocessing context already set")
    except Exception as e_mp_start:  # JUSTIFIED: Multiprocessing setup before logging; must print to stderr
        print(
            f"ERROR: Setting multiprocessing start method failed: {e_mp_start}",
            file=sys.stderr,
        )

    # Execute main function
    main()
