"""
src/cfr/deep_trainer.py

Deep CFR Trainer — replaces the tabular CFRTrainer for neural network-based training.

Training loop:
1. Every K traversals, distribute advantage network weights to workers
2. Workers run external sampling traversals, return ReservoirSamples
3. Append samples to advantage buffer (Mv) and strategy buffer (Mpi)
4. Train advantage network on Mv with weighted MSE loss: (t^alpha) * MSE(pred, target)
5. Train strategy network on Mpi similarly

Uses multiprocessing Pool pattern similar to training_loop_mixin.py.
"""

import logging
import multiprocessing
import multiprocessing.pool
import os
import threading
import time
import queue
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..config import Config
from ..constants import NUM_PLAYERS
from ..encoding import INPUT_DIM, NUM_ACTIONS
from ..networks import AdvantageNetwork, StrategyNetwork
from ..reservoir import ReservoirBuffer, ReservoirSample
from ..utils import LogQueue as ProgressQueue
from ..live_display import LiveDisplayManager
from ..log_archiver import LogArchiver

from .deep_worker import run_deep_cfr_worker, DeepCFRWorkerResult
from .exceptions import GracefulShutdownException, CheckpointSaveError, CheckpointLoadError

logger = logging.getLogger(__name__)


@dataclass
class DeepCFRConfig:
    """Configuration for Deep CFR training."""
    # Network architecture
    input_dim: int = INPUT_DIM
    hidden_dim: int = 256
    output_dim: int = NUM_ACTIONS
    dropout: float = 0.1

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 2048
    train_steps_per_iteration: int = 4000
    alpha: float = 1.5  # Weighting exponent for t^alpha loss

    # Traversals per training step
    traversals_per_step: int = 1000

    # Buffer capacity
    advantage_buffer_capacity: int = 2_000_000
    strategy_buffer_capacity: int = 2_000_000

    # Checkpointing
    save_interval: int = 10  # Save every N training steps (not traversals)

    # Use GPU for training if available
    use_gpu: bool = False

    @classmethod
    def from_yaml_config(cls, config: "Config", **overrides) -> "DeepCFRConfig":
        """Construct DeepCFRConfig from Config.deep_cfr, applying CLI overrides."""
        deep_cfg = config.deep_cfr
        kwargs = {
            "hidden_dim": deep_cfg.hidden_dim,
            "dropout": deep_cfg.dropout,
            "learning_rate": deep_cfg.learning_rate,
            "batch_size": deep_cfg.batch_size,
            "train_steps_per_iteration": deep_cfg.train_steps_per_iteration,
            "alpha": deep_cfg.alpha,
            "traversals_per_step": deep_cfg.traversals_per_step,
            "advantage_buffer_capacity": deep_cfg.advantage_buffer_capacity,
            "strategy_buffer_capacity": deep_cfg.strategy_buffer_capacity,
            "save_interval": deep_cfg.save_interval,
            "use_gpu": deep_cfg.use_gpu,
        }
        # Apply CLI overrides (only non-None values)
        for key, value in overrides.items():
            if value is not None:
                kwargs[key] = value
        return cls(**kwargs)


class DeepCFRTrainer:
    """
    Orchestrates Deep CFR training with external sampling workers.

    Replaces the tabular CFRTrainer. Uses:
    - AdvantageNetwork for regret prediction
    - StrategyNetwork for average strategy
    - ReservoirBuffers for training sample storage
    - Multiprocessing pool for parallel traversals
    """

    def __init__(
        self,
        config: Config,
        deep_cfr_config: Optional[DeepCFRConfig] = None,
        run_log_dir: Optional[str] = None,
        run_timestamp: Optional[str] = None,
        shutdown_event: Optional[threading.Event] = None,
        progress_queue: Optional[ProgressQueue] = None,
        live_display_manager: Optional[LiveDisplayManager] = None,
        archive_queue: Optional[Union[queue.Queue, "multiprocessing.Queue"]] = None,
    ):
        self.config = config
        self.dcfr_config = deep_cfr_config or DeepCFRConfig()
        self.run_log_dir = run_log_dir
        self.run_timestamp = run_timestamp
        self.shutdown_event = shutdown_event or threading.Event()
        self.progress_queue = progress_queue
        self.live_display_manager = live_display_manager
        self.archive_queue = archive_queue
        self.log_archiver_global_ref: Optional[LogArchiver] = None

        # Device selection
        if self.dcfr_config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info("Deep CFR Trainer using device: %s", self.device)

        # Networks
        self.advantage_net = AdvantageNetwork(
            input_dim=self.dcfr_config.input_dim,
            hidden_dim=self.dcfr_config.hidden_dim,
            output_dim=self.dcfr_config.output_dim,
            dropout=self.dcfr_config.dropout,
        ).to(self.device)

        self.strategy_net = StrategyNetwork(
            input_dim=self.dcfr_config.input_dim,
            hidden_dim=self.dcfr_config.hidden_dim,
            output_dim=self.dcfr_config.output_dim,
            dropout=self.dcfr_config.dropout,
        ).to(self.device)

        # Optimizers
        self.advantage_optimizer = optim.Adam(
            self.advantage_net.parameters(), lr=self.dcfr_config.learning_rate
        )
        self.strategy_optimizer = optim.Adam(
            self.strategy_net.parameters(), lr=self.dcfr_config.learning_rate
        )

        # Reservoir buffers
        self.advantage_buffer = ReservoirBuffer(
            capacity=self.dcfr_config.advantage_buffer_capacity
        )
        self.strategy_buffer = ReservoirBuffer(
            capacity=self.dcfr_config.strategy_buffer_capacity
        )

        # Training state
        self.current_iteration = 0
        self.total_traversals = 0
        self.training_step = 0

        # Tracking
        self.advantage_loss_history: List[Tuple[int, float]] = []
        self.strategy_loss_history: List[Tuple[int, float]] = []

        logger.info(
            "DeepCFRTrainer initialized. Advantage net params: %d, Strategy net params: %d",
            sum(p.numel() for p in self.advantage_net.parameters()),
            sum(p.numel() for p in self.strategy_net.parameters()),
        )

    def _get_network_weights_for_workers(self) -> Dict[str, Any]:
        """
        Serialize advantage network weights for distribution to workers.
        Returns numpy arrays for pickle-friendly multiprocessing transfer.
        """
        state_dict = self.advantage_net.state_dict()
        return {k: v.cpu().numpy() for k, v in state_dict.items()}

    def _get_network_config(self) -> Dict[str, int]:
        """Returns network configuration dict for workers."""
        return {
            "input_dim": self.dcfr_config.input_dim,
            "hidden_dim": self.dcfr_config.hidden_dim,
            "output_dim": self.dcfr_config.output_dim,
        }

    def _train_network(
        self,
        network: nn.Module,
        optimizer: optim.Optimizer,
        buffer: ReservoirBuffer,
        alpha: float,
        num_steps: int,
        network_name: str,
    ) -> float:
        """
        Train a network on reservoir buffer samples using weighted MSE loss.

        Loss = (t^alpha) * MSE(network(features), target)
        where t is the iteration number stored in each sample.

        Returns average loss over all training steps.
        """
        if len(buffer) == 0:
            logger.warning("Cannot train %s: buffer is empty.", network_name)
            return 0.0

        network.train()
        total_loss = 0.0
        actual_steps = 0
        batch_size = self.dcfr_config.batch_size

        for step in range(num_steps):
            if self.shutdown_event.is_set():
                logger.warning("Shutdown detected during %s training.", network_name)
                break

            batch = buffer.sample_batch(batch_size)
            if not batch:
                break

            # Collate batch
            features_batch = np.stack([s.features for s in batch])
            targets_batch = np.stack([s.target for s in batch])
            masks_batch = np.stack([s.action_mask for s in batch])
            iterations_batch = np.array([s.iteration for s in batch], dtype=np.float32)

            # Convert to tensors
            features_t = torch.from_numpy(features_batch).float().to(self.device)
            targets_t = torch.from_numpy(targets_batch).float().to(self.device)
            masks_t = torch.from_numpy(masks_batch).bool().to(self.device)
            iterations_t = torch.from_numpy(iterations_batch).float().to(self.device)

            # Compute iteration weights: (t + 1)^alpha to avoid 0^alpha for iteration 0
            weights = (iterations_t + 1.0).pow(alpha)
            # Normalize weights to prevent loss magnitude from growing with iterations
            weights = weights / weights.mean()

            # Forward pass
            predictions = network(features_t, masks_t)

            # Weighted MSE loss: weight per sample, masked to legal actions only
            # Only compute loss on legal actions
            masked_preds = predictions * masks_t.float()
            masked_targets = targets_t * masks_t.float()

            # Per-sample MSE (mean over action dim)
            num_legal = masks_t.float().sum(dim=1).clamp(min=1.0)
            per_sample_mse = ((masked_preds - masked_targets) ** 2).sum(dim=1) / num_legal

            # Apply iteration weights
            weighted_loss = (weights * per_sample_mse).mean()

            # Backward pass
            optimizer.zero_grad()
            weighted_loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += weighted_loss.item()
            actual_steps += 1

        avg_loss = total_loss / max(actual_steps, 1)
        logger.info(
            "%s training: %d steps, avg loss: %.6f (buffer size: %d)",
            network_name, actual_steps, avg_loss, len(buffer),
        )
        return avg_loss

    def _shutdown_pool(self, pool: Optional[multiprocessing.pool.Pool]):
        """Gracefully shuts down the multiprocessing pool."""
        if pool:
            pool_running = getattr(pool, "_state", -1) == multiprocessing.pool.RUN
            if pool_running:
                logger.info("Terminating worker pool...")
                try:
                    pool.terminate()
                    time.sleep(0.5)
                    pool.join()
                    logger.info("Worker pool terminated and joined.")
                except ValueError:
                    logger.warning("Pool already closed.")
                except Exception as e:
                    logger.error("Error during pool shutdown: %s", e, exc_info=True)
            else:
                logger.info("Worker pool already terminated.")

    def train(self, num_training_steps: Optional[int] = None):
        """
        Main Deep CFR training loop.

        Each training step:
        1. Distribute advantage network weights to workers
        2. Run K traversals in parallel
        3. Collect samples into reservoir buffers
        4. Train advantage network on advantage buffer
        5. Train strategy network on strategy buffer
        """
        cfr_config = getattr(self.config, "cfr_training", None)
        if not cfr_config:
            logger.critical("CFRTrainingConfig not found. Cannot train.")
            return

        total_steps = num_training_steps or getattr(cfr_config, "num_iterations", 100)
        num_workers = getattr(cfr_config, "num_workers", 1)
        save_interval = self.dcfr_config.save_interval
        traversals_per_step = self.dcfr_config.traversals_per_step
        display = self.live_display_manager

        # Each "training step" consists of K traversals + network training
        # The total number of traversals per step is:
        #   traversals_per_step (distributed across workers)
        # Workers per batch = num_workers, so we need ceil(K / num_workers) batches

        start_step = self.training_step + 1
        end_step = self.training_step + total_steps

        logger.info(
            "Starting Deep CFR training from step %d to %d (%d workers, %d traversals/step).",
            start_step, end_step, num_workers, traversals_per_step,
        )

        pool: Optional[multiprocessing.pool.Pool] = None

        try:
            for step in range(start_step, end_step + 1):
                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before training step")

                step_start_time = time.time()
                self.training_step = step

                if display and hasattr(display, "update_main_process_status"):
                    display.update_main_process_status(f"Step {step}: Running traversals...")

                # Get current network weights for workers
                network_weights = self._get_network_weights_for_workers()
                network_config = self._get_network_config()

                # Run traversals in batches
                step_advantage_samples: List[ReservoirSample] = []
                step_strategy_samples: List[ReservoirSample] = []
                traversals_done = 0
                total_nodes = 0

                while traversals_done < traversals_per_step:
                    if self.shutdown_event.is_set():
                        raise GracefulShutdownException("Shutdown during traversals")

                    batch_size = min(num_workers, traversals_per_step - traversals_done)

                    # Prepare worker args
                    worker_args_list = []
                    for i in range(batch_size):
                        iter_num = self.total_traversals + traversals_done + i
                        worker_args_list.append((
                            iter_num,
                            self.config,
                            network_weights,
                            network_config,
                            self.progress_queue,
                            self.archive_queue,
                            i,  # worker_id
                            self.run_log_dir or "logs",
                            self.run_timestamp or "unknown",
                        ))

                    # Execute workers
                    if num_workers == 1 or batch_size == 1:
                        # Sequential
                        results = []
                        for args in worker_args_list:
                            if self.shutdown_event.is_set():
                                raise GracefulShutdownException("Shutdown during sequential worker")
                            result = run_deep_cfr_worker(args)
                            results.append(result)
                    else:
                        # Parallel
                        if not pool:
                            logger.info("Creating worker pool (size %d)...", num_workers)
                            pool = multiprocessing.Pool(processes=num_workers)

                        try:
                            async_results = pool.map_async(
                                run_deep_cfr_worker, worker_args_list
                            )
                            while not async_results.ready():
                                if self.shutdown_event.is_set():
                                    self._shutdown_pool(pool)
                                    pool = None
                                    raise GracefulShutdownException(
                                        "Shutdown during parallel workers"
                                    )
                                async_results.wait(timeout=0.5)
                            results = async_results.get()
                        except (GracefulShutdownException, KeyboardInterrupt):
                            self._shutdown_pool(pool)
                            pool = None
                            raise
                        except Exception as e_pool:
                            logger.exception("Worker pool error: %s", e_pool)
                            self._shutdown_pool(pool)
                            pool = None
                            raise

                    # Collect samples from results
                    for result in results:
                        if isinstance(result, DeepCFRWorkerResult):
                            step_advantage_samples.extend(result.advantage_samples)
                            step_strategy_samples.extend(result.strategy_samples)
                            total_nodes += result.stats.nodes_visited
                            if result.stats.error_count > 0:
                                logger.warning(
                                    "Worker reported %d errors.", result.stats.error_count
                                )

                    traversals_done += batch_size

                self.total_traversals += traversals_done

                # Add samples to reservoir buffers
                for sample in step_advantage_samples:
                    self.advantage_buffer.add(sample)
                for sample in step_strategy_samples:
                    self.strategy_buffer.add(sample)

                logger.info(
                    "Step %d: %d traversals, %d advantage samples, %d strategy samples, %d nodes",
                    step, traversals_done, len(step_advantage_samples),
                    len(step_strategy_samples), total_nodes,
                )

                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before network training")

                # Train advantage network
                if display and hasattr(display, "update_main_process_status"):
                    display.update_main_process_status(f"Step {step}: Training advantage net...")

                adv_loss = self._train_network(
                    self.advantage_net, self.advantage_optimizer,
                    self.advantage_buffer, self.dcfr_config.alpha,
                    self.dcfr_config.train_steps_per_iteration,
                    "AdvantageNetwork",
                )
                self.advantage_loss_history.append((step, adv_loss))

                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before strategy training")

                # Train strategy network
                if display and hasattr(display, "update_main_process_status"):
                    display.update_main_process_status(f"Step {step}: Training strategy net...")

                strat_loss = self._train_network(
                    self.strategy_net, self.strategy_optimizer,
                    self.strategy_buffer, self.dcfr_config.alpha,
                    self.dcfr_config.train_steps_per_iteration,
                    "StrategyNetwork",
                )
                self.strategy_loss_history.append((step, strat_loss))

                step_time = time.time() - step_start_time

                # Update display
                if display and hasattr(display, "update_stats"):
                    display.update_stats(
                        iteration=step,
                        infosets=f"Adv:{len(self.advantage_buffer)} Str:{len(self.strategy_buffer)}",
                        exploitability=f"AdvL:{adv_loss:.4f} StrL:{strat_loss:.4f}",
                        last_iter_time=step_time,
                    )
                if display and hasattr(display, "update_main_process_status"):
                    display.update_main_process_status("Idle / Waiting...")

                logger.info(
                    "Step %d complete in %.2fs. Adv loss: %.6f, Strat loss: %.6f. "
                    "Buffers: Adv=%d, Strat=%d. Total traversals: %d",
                    step, step_time, adv_loss, strat_loss,
                    len(self.advantage_buffer), len(self.strategy_buffer),
                    self.total_traversals,
                )

                # Periodic save
                if save_interval > 0 and step % save_interval == 0:
                    self.save_checkpoint()

            logger.info("Deep CFR training completed %d steps.", total_steps)

        except (GracefulShutdownException, KeyboardInterrupt) as e:
            logger.warning("Shutdown during training: %s. Saving checkpoint...", type(e).__name__)
            self._shutdown_pool(pool)
            pool = None
            self.save_checkpoint()
            raise GracefulShutdownException("Shutdown processed in Deep CFR trainer") from e

        except Exception as e:
            logger.exception("Unhandled error in Deep CFR training loop.")
            self._shutdown_pool(pool)
            pool = None
            self.save_checkpoint()
            raise

        # Final save
        if not self.shutdown_event.is_set():
            self._shutdown_pool(pool)
            self.save_checkpoint()

    def save_checkpoint(self, filepath: Optional[str] = None):
        """
        Save training state: network weights, optimizer state, buffers, iteration count.

        Saves the main checkpoint as a .pt file and reservoir buffers as .npz files
        alongside it.

        Raises:
            CheckpointSaveError: If saving the checkpoint or buffers fails.
        """
        path = filepath or getattr(
            self.config.persistence, "agent_data_save_path",
            "strategy/deep_cfr_checkpoint.pt",
        )

        try:
            # Ensure directory exists
            checkpoint_dir = os.path.dirname(path) if os.path.dirname(path) else "."
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Derive buffer file paths from the main checkpoint path
            base_path = os.path.splitext(path)[0]
            adv_buffer_path = f"{base_path}_advantage_buffer"
            strat_buffer_path = f"{base_path}_strategy_buffer"

            checkpoint = {
                "advantage_net_state_dict": self.advantage_net.state_dict(),
                "strategy_net_state_dict": self.strategy_net.state_dict(),
                "advantage_optimizer_state_dict": self.advantage_optimizer.state_dict(),
                "strategy_optimizer_state_dict": self.strategy_optimizer.state_dict(),
                "training_step": self.training_step,
                "total_traversals": self.total_traversals,
                "current_iteration": self.current_iteration,
                "dcfr_config": {
                    "input_dim": self.dcfr_config.input_dim,
                    "hidden_dim": self.dcfr_config.hidden_dim,
                    "output_dim": self.dcfr_config.output_dim,
                    "dropout": self.dcfr_config.dropout,
                    "learning_rate": self.dcfr_config.learning_rate,
                    "batch_size": self.dcfr_config.batch_size,
                    "train_steps_per_iteration": self.dcfr_config.train_steps_per_iteration,
                    "alpha": self.dcfr_config.alpha,
                    "traversals_per_step": self.dcfr_config.traversals_per_step,
                    "advantage_buffer_capacity": self.dcfr_config.advantage_buffer_capacity,
                    "strategy_buffer_capacity": self.dcfr_config.strategy_buffer_capacity,
                },
                "advantage_loss_history": self.advantage_loss_history,
                "strategy_loss_history": self.strategy_loss_history,
                "advantage_buffer_path": adv_buffer_path,
                "strategy_buffer_path": strat_buffer_path,
            }

            torch.save(checkpoint, path)
            self.advantage_buffer.save(adv_buffer_path)
            self.strategy_buffer.save(strat_buffer_path)
            logger.info(
                "Checkpoint saved to %s (step %d, %d traversals).",
                path, self.training_step, self.total_traversals,
            )
        except (OSError, IOError, PermissionError) as e:
            logger.error("Failed to save checkpoint to %s: %s", path, e)
            raise CheckpointSaveError(f"Failed to save checkpoint to {path}: {e}") from e
        except Exception as e:
            logger.error("Unexpected error saving checkpoint to %s: %s", path, e)
            raise CheckpointSaveError(f"Unexpected error saving checkpoint to {path}: {e}") from e

    def load_checkpoint(self, filepath: Optional[str] = None):
        """
        Load training state from checkpoint.

        Raises:
            CheckpointLoadError: If loading the checkpoint or buffers fails.
        """
        path = filepath or getattr(
            self.config.persistence, "agent_data_save_path",
            "strategy/deep_cfr_checkpoint.pt",
        )

        if not os.path.exists(path):
            logger.info("No checkpoint found at %s. Starting fresh.", path)
            return

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            self.advantage_net.load_state_dict(checkpoint["advantage_net_state_dict"])
            self.strategy_net.load_state_dict(checkpoint["strategy_net_state_dict"])
            self.advantage_optimizer.load_state_dict(checkpoint["advantage_optimizer_state_dict"])
            self.strategy_optimizer.load_state_dict(checkpoint["strategy_optimizer_state_dict"])

            # Load reservoir buffers from their saved .npz files
            adv_buffer_path = checkpoint.get("advantage_buffer_path")
            strat_buffer_path = checkpoint.get("strategy_buffer_path")

            if adv_buffer_path:
                self.advantage_buffer = ReservoirBuffer(
                    capacity=self.dcfr_config.advantage_buffer_capacity
                )
                self.advantage_buffer.load(adv_buffer_path)

            if strat_buffer_path:
                self.strategy_buffer = ReservoirBuffer(
                    capacity=self.dcfr_config.strategy_buffer_capacity
                )
                self.strategy_buffer.load(strat_buffer_path)

            self.training_step = checkpoint.get("training_step", 0)
            self.total_traversals = checkpoint.get("total_traversals", 0)
            self.current_iteration = checkpoint.get("current_iteration", 0)
            self.advantage_loss_history = checkpoint.get("advantage_loss_history", [])
            self.strategy_loss_history = checkpoint.get("strategy_loss_history", [])

            logger.info(
                "Checkpoint loaded from %s. Resuming at step %d (%d traversals). "
                "Buffers: Adv=%d, Strat=%d.",
                path, self.training_step, self.total_traversals,
                len(self.advantage_buffer), len(self.strategy_buffer),
            )
        except FileNotFoundError as e:
            logger.error("Checkpoint file not found: %s", path)
            raise CheckpointLoadError(f"Checkpoint file not found: {path}") from e
        except (KeyError, ValueError) as e:
            logger.error("Corrupted or incompatible checkpoint file %s: %s", path, e)
            raise CheckpointLoadError(f"Corrupted or incompatible checkpoint file {path}: {e}") from e
        except (OSError, IOError) as e:
            logger.error("Failed to load checkpoint from %s: %s", path, e)
            raise CheckpointLoadError(f"Failed to load checkpoint from {path}: {e}") from e
        except Exception as e:
            logger.error("Unexpected error loading checkpoint from %s: %s", path, e)
            raise CheckpointLoadError(f"Unexpected error loading checkpoint from {path}: {e}") from e

    def get_strategy_network(self) -> StrategyNetwork:
        """Returns the trained strategy network for deployment/evaluation."""
        return self.strategy_net

    def get_advantage_network(self) -> AdvantageNetwork:
        """Returns the trained advantage network."""
        return self.advantage_net
