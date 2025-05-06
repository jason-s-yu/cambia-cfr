"""
src/cfr/trainer.py

Main CFR Trainer class, composing functionality from mixins.
"""

import logging
import threading
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np

from ..analysis_tools import AnalysisTools
from ..config import Config
from ..constants import NUM_PLAYERS
from ..utils import PolicyDict, ReachProbDict, LogQueue as ProgressQueue
from ..live_display import LiveDisplayManager

from .data_manager_mixin import CFRDataManagerMixin
from .recursion_mixin import CFRRecursionMixin
from .training_loop_mixin import CFRTrainingLoopMixin

logger = logging.getLogger(__name__)


# Ensure Mixin classes are defined before use or imported correctly
class CFRTrainer(CFRDataManagerMixin, CFRRecursionMixin, CFRTrainingLoopMixin):
    """
    Orchestrates CFR+ training for Cambia using a mixin-based architecture.
    """

    def __init__(
        self,
        config: Config,
        run_log_dir: Optional[str] = None,
        run_timestamp: Optional[str] = None,
        shutdown_event: Optional[threading.Event] = None,
        progress_queue: Optional[ProgressQueue] = None,
        live_display_manager: Optional[LiveDisplayManager] = None,
    ):
        """
        Initializes the CFRTrainer.

        Args:
            config: Configuration object.
            run_log_dir: Directory for logs specific to this run.
            run_timestamp: Timestamp string for this run.
            shutdown_event: Threading event to signal graceful shutdown.
            progress_queue: Multiprocessing queue for worker progress updates.
            live_display_manager: Manager for the Rich live display.
        """
        self.config = config
        self.num_players = NUM_PLAYERS
        self.progress_queue = progress_queue
        self.live_display_manager = live_display_manager

        # Initialize data structures (managed primarily by CFRDataManagerMixin)
        self.regret_sum: PolicyDict = defaultdict(lambda: np.array([], dtype=np.float64))
        self.strategy_sum: PolicyDict = defaultdict(
            lambda: np.array([], dtype=np.float64)
        )
        self.reach_prob_sum: ReachProbDict = defaultdict(float)
        self.average_strategy: Optional[PolicyDict] = None

        # Training state
        self.current_iteration = 0
        self.exploitability_results: List[Tuple[int, float]] = []

        # Analysis and Logging
        analysis_log_dir = run_log_dir if run_log_dir else config.logging.log_dir
        analysis_log_prefix = config.logging.log_file_prefix
        self.analysis = AnalysisTools(config, analysis_log_dir, analysis_log_prefix)
        self.run_log_dir = run_log_dir
        self.run_timestamp = run_timestamp

        # Internal state for progress display/debugging (used by display manager now)
        self.max_depth_this_iter = 0
        self._last_exploit_str = "N/A"
        self._total_infosets_str = "0"

        # Shutdown handling
        self.shutdown_event = shutdown_event or threading.Event()

        logger.info("CFRTrainer initialized with %d players.", self.num_players)
        logger.debug("Config loaded: %s", self.config)
        logger.debug("Run Log Directory: %s", self.run_log_dir)
        logger.debug("Run Timestamp: %s", self.run_timestamp)
        logger.debug(
            "Live Display Manager: %s",
            "Provided" if self.live_display_manager else "None",
        )
