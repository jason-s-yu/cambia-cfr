"""
src/cfr/deep_worker.py

Implements the Deep CFR worker process using External Sampling MCCFR.

Key differences from worker.py (tabular outcome sampling):
- External Sampling: enumerate ALL actions at traverser nodes, sample ONE at opponent/chance nodes
- No importance sampling correction (exact regrets from enumeration)
- Returns ReservoirSamples instead of regret/strategy dict updates
- Uses neural network for strategy computation (AdvantageNetwork -> ReLU -> normalize)
- Encodes infosets via encode_infoset() from encoding.py
"""

import logging
import os
import queue
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..agent_state import AgentState
from ..config import Config
from .exceptions import (
    GameStateError,
    ActionApplicationError,
    UndoFailureError,
    AgentStateError,
    ObservationUpdateError,
    EncodingError,
    NetworkError,
    TraversalError,
)
from ..constants import (
    NUM_PLAYERS,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionAbilityPeekOtherSelect,
    ActionAbilityPeekOwnSelect,
    ActionDiscard,
    ActionSnapOpponentMove,
    DecisionContext,
)
from ..game.engine import CambiaGameState
from ..serial_rotating_handler import SerialRotatingFileHandler
from ..abstraction import get_card_bucket
from ..encoding import encode_infoset, encode_action_mask, action_to_index, NUM_ACTIONS
from ..networks import AdvantageNetwork, get_strategy_from_advantages
from ..reservoir import ReservoirSample
from ..utils import WorkerStats, SimulationNodeData

# Re-use observation helpers from worker.py
from .worker import _create_observation, _filter_observation

logger = logging.getLogger(__name__)

# Progress update interval (nodes)
PROGRESS_UPDATE_NODE_INTERVAL = 2500


@dataclass
class DeepCFRWorkerResult:
    """Results from a single Deep CFR worker traversal."""

    advantage_samples: List[ReservoirSample] = field(default_factory=list)
    strategy_samples: List[ReservoirSample] = field(default_factory=list)
    stats: WorkerStats = field(default_factory=WorkerStats)
    simulation_nodes: List[SimulationNodeData] = field(default_factory=list)
    final_utility: Optional[List[float]] = None


def _get_strategy_from_network(
    network_weights: Dict[str, torch.Tensor],
    features: np.ndarray,
    action_mask: np.ndarray,
    network_config: Dict[str, int],
) -> np.ndarray:
    """
    Compute strategy from advantage network weights.

    Loads weights into a temporary AdvantageNetwork, runs forward pass,
    then applies ReLU + normalize (regret matching on predicted advantages).

    Returns numpy array of shape (NUM_ACTIONS,) with strategy probabilities.

    Raises:
        NetworkError: If network inference fails
    """
    try:
        input_dim = network_config.get("input_dim", features.shape[0])
        hidden_dim = network_config.get("hidden_dim", 256)
        output_dim = network_config.get("output_dim", NUM_ACTIONS)

        net = AdvantageNetwork(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
        )
        net.load_state_dict(network_weights)
        net.eval()

        with torch.no_grad():
            features_tensor = torch.from_numpy(features).float().unsqueeze(0)
            mask_tensor = torch.from_numpy(action_mask).bool().unsqueeze(0)
            # get_strategy_from_advantages expects torch tensors, returns torch tensor
            raw_advantages = net(features_tensor, mask_tensor).squeeze(0)
            strategy_tensor = get_strategy_from_advantages(
                raw_advantages.unsqueeze(0), mask_tensor
            )
            return strategy_tensor.squeeze(0).numpy()
    except Exception as e:
        raise NetworkError(f"Network inference failed: {e}") from e


def _deep_traverse(
    game_state: CambiaGameState,
    agent_states: List[AgentState],
    updating_player: int,
    network_weights: Optional[Dict[str, torch.Tensor]],
    iteration: int,
    config: Config,
    network_config: Dict[str, int],
    advantage_samples: List[ReservoirSample],
    strategy_samples: List[ReservoirSample],
    depth: int,
    worker_stats: WorkerStats,
    progress_queue: Optional[queue.Queue],
    worker_id: int,
    min_depth_after_bottom_out_tracker: List[float],
    has_bottomed_out_tracker: List[bool],
    simulation_nodes: List[SimulationNodeData],
) -> np.ndarray:
    """
    Recursive External Sampling traversal for Deep CFR.

    At traverser's node: enumerate ALL legal actions, recurse on each, compute exact regrets.
    At opponent's node: sample ONE action from strategy (network), recurse.
    At chance node: sample ONE outcome, recurse.

    Returns utility vector for both players.
    """
    logger_t = logging.getLogger(__name__)

    worker_stats.nodes_visited += 1
    worker_stats.max_depth = max(worker_stats.max_depth, depth)

    if has_bottomed_out_tracker[0]:
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )

    # Progress update
    if progress_queue and (
        worker_stats.nodes_visited % PROGRESS_UPDATE_NODE_INTERVAL == 0
    ):
        try:
            progress_update = (
                worker_id,
                depth,
                worker_stats.max_depth,
                worker_stats.nodes_visited,
                (
                    int(min_depth_after_bottom_out_tracker[0])
                    if min_depth_after_bottom_out_tracker[0] != float("inf")
                    else 0
                ),
            )
            progress_queue.put_nowait(progress_update)
        except queue.Full:
            pass
        except Exception as pq_e:
            logger_t.error("W%d D%d: Error putting progress: %s", worker_id, depth, pq_e)
            worker_stats.error_count += 1

    # Terminal check
    if game_state.is_terminal():
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        return np.array(
            [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
        )

    # Depth limit check
    if depth >= config.system.recursion_limit:
        logger_t.error("W%d D%d: Max recursion depth reached.", worker_id, depth)
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Determine decision context
    if game_state.snap_phase_active:
        current_context = DecisionContext.SNAP_DECISION
    elif game_state.pending_action:
        pending = game_state.pending_action
        if isinstance(pending, ActionDiscard):
            current_context = DecisionContext.POST_DRAW
        elif isinstance(
            pending,
            (
                ActionAbilityPeekOwnSelect,
                ActionAbilityPeekOtherSelect,
                ActionAbilityBlindSwapSelect,
                ActionAbilityKingLookSelect,
                ActionAbilityKingSwapDecision,
            ),
        ):
            current_context = DecisionContext.ABILITY_SELECT
        elif isinstance(pending, ActionSnapOpponentMove):
            current_context = DecisionContext.SNAP_MOVE
        else:
            logger_t.warning(
                "W%d D%d: Unknown pending action type (%s).",
                worker_id,
                depth,
                type(pending).__name__,
            )
            worker_stats.warning_count += 1
            current_context = DecisionContext.START_TURN
    else:
        current_context = DecisionContext.START_TURN

    player = game_state.get_acting_player()
    if player == -1:
        logger_t.error("W%d D%d: Could not determine acting player.", worker_id, depth)
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    opponent = 1 - player

    # Get legal actions
    try:
        legal_actions_set = game_state.get_legal_actions()
        legal_actions = sorted(list(legal_actions_set), key=repr)
    except GameStateError as e_legal:
        logger_t.warning(
            "W%d D%d: Game state error getting legal actions: %s",
            worker_id,
            depth,
            e_legal,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)
    except Exception as e_legal:  # JUSTIFIED: worker resilience - workers must not crash the training pool
        logger_t.error(
            "W%d D%d: Error getting legal actions: %s",
            worker_id,
            depth,
            e_legal,
            exc_info=True,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    num_actions = len(legal_actions)
    if num_actions == 0:
        if not game_state.is_terminal():
            logger_t.error(
                "W%d D%d: No legal actions but non-terminal!", worker_id, depth
            )
            worker_stats.error_count += 1
            return np.zeros(NUM_PLAYERS, dtype=np.float64)
        else:
            has_bottomed_out_tracker[0] = True
            min_depth_after_bottom_out_tracker[0] = min(
                min_depth_after_bottom_out_tracker[0], float(depth)
            )
            return np.array(
                [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
            )

    # Encode infoset and action mask
    current_agent_state = agent_states[player]

    # Get drawn card bucket for POST_DRAW encoding
    drawn_card_bucket = None
    if current_context == DecisionContext.POST_DRAW:
        drawn_card_obj = game_state.pending_action_data.get("drawn_card")
        if drawn_card_obj is not None:
            drawn_card_bucket = get_card_bucket(drawn_card_obj)

    try:
        features = encode_infoset(
            current_agent_state, current_context, drawn_card_bucket=drawn_card_bucket
        )
        action_mask = encode_action_mask(legal_actions)
    except (EncodingError, AgentStateError) as e_encode:
        logger_t.warning(
            "W%d D%d: Encoding/agent state error for infoset: %s",
            worker_id,
            depth,
            e_encode,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)
    except Exception as e_encode:  # JUSTIFIED: worker resilience - workers must not crash the training pool
        logger_t.error(
            "W%d D%d: Error encoding infoset/mask: %s",
            worker_id,
            depth,
            e_encode,
            exc_info=True,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Compute strategy from advantage network
    if network_weights is not None:
        try:
            strategy = _get_strategy_from_network(
                network_weights, features, action_mask, network_config
            )
        except NetworkError as e_net:
            logger_t.warning(
                "W%d D%d: Network inference error: %s. Using uniform.",
                worker_id,
                depth,
                e_net,
            )
            worker_stats.warning_count += 1
            strategy = np.ones(num_actions, dtype=np.float64) / num_actions
        except Exception as e_net:  # JUSTIFIED: worker resilience - fallback to uniform strategy on unexpected errors
            logger_t.warning(
                "W%d D%d: Network inference failed: %s. Using uniform.",
                worker_id,
                depth,
                e_net,
            )
            worker_stats.warning_count += 1
            strategy = np.ones(num_actions, dtype=np.float64) / num_actions
    else:
        # No network weights yet (first iteration) - use uniform
        strategy = np.ones(num_actions, dtype=np.float64) / num_actions

    # Map network strategy (NUM_ACTIONS) to local strategy (num_actions)
    # The strategy from get_strategy_from_advantages is already over legal actions only
    # if action_mask was used correctly. However, the network outputs NUM_ACTIONS dims.
    # We need to extract only the legal action probabilities.
    if len(strategy) == NUM_ACTIONS:
        local_strategy = np.zeros(num_actions, dtype=np.float64)
        for a_idx, action in enumerate(legal_actions):
            global_idx = action_to_index(action)
            local_strategy[a_idx] = strategy[global_idx]
        total = local_strategy.sum()
        if total > 1e-9:
            local_strategy /= total
        else:
            local_strategy = np.ones(num_actions, dtype=np.float64) / num_actions
        strategy = local_strategy

    # Ensure strategy length matches
    if len(strategy) != num_actions:
        logger_t.warning(
            "W%d D%d: Strategy len %d != num_actions %d. Using uniform.",
            worker_id,
            depth,
            len(strategy),
            num_actions,
        )
        worker_stats.warning_count += 1
        strategy = np.ones(num_actions, dtype=np.float64) / num_actions

    # --- External Sampling Logic ---
    if player == updating_player:
        # TRAVERSER'S NODE: enumerate ALL legal actions
        action_values = np.zeros((num_actions, NUM_PLAYERS), dtype=np.float64)

        for a_idx, action in enumerate(legal_actions):
            apply_success = False
            try:
                state_delta, undo_info = game_state.apply_action(action)
                if callable(undo_info):
                    apply_success = True
                else:
                    logger_t.error(
                        "W%d D%d: apply_action for %s returned invalid undo.",
                        worker_id,
                        depth,
                        action,
                    )
                    worker_stats.error_count += 1
            except ActionApplicationError as e_apply:
                logger_t.warning(
                    "W%d D%d: Action application error for %s: %s",
                    worker_id,
                    depth,
                    action,
                    e_apply,
                )
                worker_stats.error_count += 1
            except Exception as e_apply:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                logger_t.error(
                    "W%d D%d: Error applying action %s: %s",
                    worker_id,
                    depth,
                    action,
                    e_apply,
                    exc_info=True,
                )
                worker_stats.error_count += 1

            if apply_success:
                # Create observation and update agent states
                observation = _create_observation(
                    None, action, game_state, player, game_state.snap_results_log
                )
                next_agent_states = []
                agent_update_failed = False

                if observation is None:
                    logger_t.error(
                        "W%d D%d: Failed to create observation after %s.",
                        worker_id,
                        depth,
                        action,
                    )
                    worker_stats.error_count += 1
                    agent_update_failed = True
                    try:
                        undo_info()
                    except UndoFailureError:
                        pass
                    except Exception:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                        pass
                else:
                    try:
                        for agent_idx, agent_state in enumerate(agent_states):
                            cloned_agent = agent_state.clone()
                            player_specific_obs = _filter_observation(
                                observation, agent_idx
                            )
                            cloned_agent.update(player_specific_obs)
                            next_agent_states.append(cloned_agent)
                    except (AgentStateError, ObservationUpdateError) as e_update:
                        logger_t.warning(
                            "W%d D%d: Agent state update error after %s: %s",
                            worker_id,
                            depth,
                            action,
                            e_update,
                        )
                        worker_stats.error_count += 1
                        agent_update_failed = True
                        try:
                            undo_info()
                        except UndoFailureError:
                            pass
                        except Exception:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                            pass
                    except Exception as e_update:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                        logger_t.error(
                            "W%d D%d: Error updating agents after %s: %s",
                            worker_id,
                            depth,
                            action,
                            e_update,
                            exc_info=True,
                        )
                        worker_stats.error_count += 1
                        agent_update_failed = True
                        try:
                            undo_info()
                        except UndoFailureError:
                            pass
                        except Exception:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                            pass

                if not agent_update_failed:
                    try:
                        action_values[a_idx] = _deep_traverse(
                            game_state,
                            next_agent_states,
                            updating_player,
                            network_weights,
                            iteration,
                            config,
                            network_config,
                            advantage_samples,
                            strategy_samples,
                            depth + 1,
                            worker_stats,
                            progress_queue,
                            worker_id,
                            min_depth_after_bottom_out_tracker,
                            has_bottomed_out_tracker,
                            simulation_nodes,
                        )
                    except TraversalError as e_recurse:
                        logger_t.warning(
                            "W%d D%d: Traversal error in recursion after %s: %s",
                            worker_id,
                            depth,
                            action,
                            e_recurse,
                        )
                        worker_stats.error_count += 1
                    except Exception as e_recurse:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                        logger_t.error(
                            "W%d D%d: Recursion error after %s: %s",
                            worker_id,
                            depth,
                            action,
                            e_recurse,
                            exc_info=True,
                        )
                        worker_stats.error_count += 1

                    # Undo after recursion
                    try:
                        undo_info()
                    except UndoFailureError as e_undo:
                        logger_t.error(
                            "W%d D%d: Undo failure for %s: %s. State corrupt.",
                            worker_id,
                            depth,
                            action,
                            e_undo,
                            exc_info=True,
                        )
                        worker_stats.error_count += 1
                        return np.zeros(NUM_PLAYERS, dtype=np.float64)
                    except Exception as e_undo:  # JUSTIFIED: worker resilience - must not crash on undo, state likely corrupt
                        logger_t.error(
                            "W%d D%d: Error undoing %s: %s. State corrupt.",
                            worker_id,
                            depth,
                            action,
                            e_undo,
                            exc_info=True,
                        )
                        worker_stats.error_count += 1
                        return np.zeros(NUM_PLAYERS, dtype=np.float64)

        # Compute exact counterfactual values
        # node_value = sum over actions: strategy[a] * action_values[a]
        node_value = strategy @ action_values  # shape: (NUM_PLAYERS,)

        # Compute regrets: regret(a) = v(a)[player] - node_value[player]
        regrets = action_values[:, player] - node_value[player]

        # Build full-size regret target vector (NUM_ACTIONS)
        regret_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a_idx, action in enumerate(legal_actions):
            global_idx = action_to_index(action)
            regret_target[global_idx] = regrets[a_idx]

        # Store advantage sample
        advantage_samples.append(
            ReservoirSample(
                features=features.astype(np.float32),
                target=regret_target,
                action_mask=action_mask.astype(np.bool_),
                iteration=iteration,
            )
        )

        return node_value

    else:
        # OPPONENT'S NODE: sample ONE action from strategy
        # Store strategy sample for this infoset
        strategy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a_idx, action in enumerate(legal_actions):
            global_idx = action_to_index(action)
            strategy_target[global_idx] = strategy[a_idx]

        strategy_samples.append(
            ReservoirSample(
                features=features.astype(np.float32),
                target=strategy_target,
                action_mask=action_mask.astype(np.bool_),
                iteration=iteration,
            )
        )

        # Sample one action
        if np.sum(strategy) > 1e-9:
            try:
                chosen_idx = np.random.choice(num_actions, p=strategy)
            except ValueError:
                chosen_idx = np.random.choice(num_actions)
                worker_stats.warning_count += 1
        else:
            chosen_idx = np.random.choice(num_actions)
            worker_stats.warning_count += 1

        chosen_action = legal_actions[chosen_idx]

        # Apply action, recurse, undo
        apply_success = False
        node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)

        try:
            state_delta, undo_info = game_state.apply_action(chosen_action)
            if callable(undo_info):
                apply_success = True
            else:
                logger_t.error(
                    "W%d D%d: apply_action for sampled %s returned invalid undo.",
                    worker_id,
                    depth,
                    chosen_action,
                )
                worker_stats.error_count += 1
        except ActionApplicationError as e_apply:
            logger_t.warning(
                "W%d D%d: Action application error for sampled %s: %s",
                worker_id,
                depth,
                chosen_action,
                e_apply,
            )
            worker_stats.error_count += 1
        except Exception as e_apply:  # JUSTIFIED: worker resilience - workers must not crash the training pool
            logger_t.error(
                "W%d D%d: Error applying sampled %s: %s",
                worker_id,
                depth,
                chosen_action,
                e_apply,
                exc_info=True,
            )
            worker_stats.error_count += 1

        if apply_success:
            observation = _create_observation(
                None, chosen_action, game_state, player, game_state.snap_results_log
            )
            next_agent_states = []
            agent_update_failed = False

            if observation is None:
                logger_t.error(
                    "W%d D%d: Failed to create observation after sampled %s.",
                    worker_id,
                    depth,
                    chosen_action,
                )
                worker_stats.error_count += 1
                agent_update_failed = True
                try:
                    undo_info()
                except UndoFailureError:
                    pass
                except Exception:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                    pass
            else:
                try:
                    for agent_idx, agent_state in enumerate(agent_states):
                        cloned_agent = agent_state.clone()
                        player_specific_obs = _filter_observation(observation, agent_idx)
                        cloned_agent.update(player_specific_obs)
                        next_agent_states.append(cloned_agent)
                except (AgentStateError, ObservationUpdateError) as e_update:
                    logger_t.warning(
                        "W%d D%d: Agent state update error after sampled %s: %s",
                        worker_id,
                        depth,
                        chosen_action,
                        e_update,
                    )
                    worker_stats.error_count += 1
                    agent_update_failed = True
                    try:
                        undo_info()
                    except UndoFailureError:
                        pass
                    except Exception:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                        pass
                except Exception as e_update:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                    logger_t.error(
                        "W%d D%d: Error updating agents after sampled %s: %s",
                        worker_id,
                        depth,
                        chosen_action,
                        e_update,
                        exc_info=True,
                    )
                    worker_stats.error_count += 1
                    agent_update_failed = True
                    try:
                        undo_info()
                    except UndoFailureError:
                        pass
                    except Exception:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                        pass

            if not agent_update_failed:
                try:
                    node_value = _deep_traverse(
                        game_state,
                        next_agent_states,
                        updating_player,
                        network_weights,
                        iteration,
                        config,
                        network_config,
                        advantage_samples,
                        strategy_samples,
                        depth + 1,
                        worker_stats,
                        progress_queue,
                        worker_id,
                        min_depth_after_bottom_out_tracker,
                        has_bottomed_out_tracker,
                        simulation_nodes,
                    )
                except TraversalError as e_recurse:
                    logger_t.warning(
                        "W%d D%d: Traversal error in recursion after sampled %s: %s",
                        worker_id,
                        depth,
                        chosen_action,
                        e_recurse,
                    )
                    worker_stats.error_count += 1
                except Exception as e_recurse:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                    logger_t.error(
                        "W%d D%d: Recursion error after sampled %s: %s",
                        worker_id,
                        depth,
                        chosen_action,
                        e_recurse,
                        exc_info=True,
                    )
                    worker_stats.error_count += 1

                try:
                    undo_info()
                except UndoFailureError as e_undo:
                    logger_t.error(
                        "W%d D%d: Undo failure for sampled %s: %s. State corrupt.",
                        worker_id,
                        depth,
                        chosen_action,
                        e_undo,
                        exc_info=True,
                    )
                    worker_stats.error_count += 1
                    return np.zeros(NUM_PLAYERS, dtype=np.float64)
                except Exception as e_undo:  # JUSTIFIED: worker resilience - must not crash on undo, state likely corrupt
                    logger_t.error(
                        "W%d D%d: Error undoing sampled %s: %s. State corrupt.",
                        worker_id,
                        depth,
                        chosen_action,
                        e_undo,
                        exc_info=True,
                    )
                    worker_stats.error_count += 1
                    return np.zeros(NUM_PLAYERS, dtype=np.float64)

        return node_value


def run_deep_cfr_worker(
    worker_args: Tuple[
        int,  # iteration
        Config,
        Optional[Dict[str, Any]],  # network_weights (serialized state_dict)
        Dict[str, int],  # network_config
        Optional[queue.Queue],  # progress_queue
        Optional[Any],  # archive_queue
        int,  # worker_id
        str,  # run_log_dir
        str,  # run_timestamp
    ],
) -> Optional[DeepCFRWorkerResult]:
    """
    Top-level function executed by each Deep CFR worker process.
    Sets up logging, initializes game, runs external sampling traversal,
    returns advantage and strategy samples.
    """
    logger_instance: Optional[logging.Logger] = None
    worker_stats = WorkerStats()
    (
        iteration,
        config,
        network_weights_serialized,
        network_config,
        progress_queue,
        archive_queue,
        worker_id,
        run_log_dir,
        run_timestamp,
    ) = worker_args

    worker_stats.worker_id = worker_id
    simulation_nodes_this_sim: List[SimulationNodeData] = []
    advantage_samples: List[ReservoirSample] = []
    strategy_samples: List[ReservoirSample] = []

    # --- Logging setup (same pattern as tabular worker) ---
    worker_root_logger = logging.getLogger()
    try:
        for handler in worker_root_logger.handlers[:]:
            worker_root_logger.removeHandler(handler)
            if hasattr(handler, "close"):
                try:
                    handler.close()
                except Exception:
                    pass

        worker_root_logger.setLevel(logging.DEBUG)
        null_handler = logging.NullHandler()
        worker_root_logger.addHandler(null_handler)
        worker_root_logger.propagate = False

        worker_log_dir = os.path.join(run_log_dir, f"w{worker_id}")
        os.makedirs(worker_log_dir, exist_ok=True)
        log_pattern = os.path.join(
            worker_log_dir,
            f"{config.logging.log_file_prefix}_run_{run_timestamp}-w{worker_id}",
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)-8s - [%(processName)-20s] - %(name)-25s - %(message)s"
        )
        file_handler = SerialRotatingFileHandler(
            filename_pattern=log_pattern,
            maxBytes=config.logging.log_max_bytes,
            backupCount=config.logging.log_backup_count,
            encoding="utf-8",
            archive_queue=archive_queue,
            logging_config_snapshot=config.logging,
        )
        worker_log_level_str = config.logging.get_worker_log_level(
            worker_id, config.cfr_training.num_workers
        )
        file_log_level = getattr(logging, worker_log_level_str.upper(), logging.DEBUG)
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)
        worker_root_logger.addHandler(file_handler)

        logger_instance = logging.getLogger(__name__)
        logger_instance.info(
            "Deep CFR Worker %d logging initialized (dir: %s).", worker_id, worker_log_dir
        )
    except Exception as log_setup_e:
        print(
            f"!!! CRITICAL Error setting up logging W{worker_id}: {log_setup_e} !!!",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        worker_stats.error_count += 1
        if not worker_root_logger.hasHandlers():
            worker_root_logger.addHandler(logging.NullHandler())
        logger_instance = logging.getLogger(__name__)

    # --- Main simulation logic ---
    try:
        # Deserialize network weights to torch tensors
        network_weights: Optional[Dict[str, torch.Tensor]] = None
        if network_weights_serialized is not None:
            try:
                network_weights = {
                    k: torch.tensor(v) if isinstance(v, np.ndarray) else v
                    for k, v in network_weights_serialized.items()
                }
            except Exception as e_deserialize:
                if logger_instance:
                    logger_instance.warning(
                        "W%d: Failed to deserialize network weights: %s. Using uniform strategy.",
                        worker_id,
                        e_deserialize,
                    )
                worker_stats.warning_count += 1
                network_weights = None

        # Initialize game state
        try:
            game_state = CambiaGameState(house_rules=config.cambia_rules)
        except GameStateError as game_init_e:
            if logger_instance:
                logger_instance.warning(
                    "W%d Iter %d: Game state initialization error: %s",
                    worker_id,
                    iteration,
                    game_init_e,
                )
            worker_stats.error_count += 1
            return DeepCFRWorkerResult(
                stats=worker_stats,
                simulation_nodes=simulation_nodes_this_sim,
            )
        except Exception as game_init_e:  # JUSTIFIED: worker resilience - workers must not crash the training pool
            if logger_instance:
                logger_instance.error(
                    "W%d Iter %d: Failed GameState init: %s",
                    worker_id,
                    iteration,
                    game_init_e,
                    exc_info=True,
                )
            worker_stats.error_count += 1
            return DeepCFRWorkerResult(
                stats=worker_stats,
                simulation_nodes=simulation_nodes_this_sim,
            )

        # Initialize agent states
        initial_agent_states = []
        if not game_state.is_terminal():
            try:
                initial_obs = _create_observation(None, None, game_state, -1, [])
                if initial_obs is None:
                    raise ValueError("Failed to create initial observation.")

                initial_hands = [list(p.hand) for p in game_state.players]
                initial_peeks = [p.initial_peek_indices for p in game_state.players]
                for i in range(NUM_PLAYERS):
                    agent = AgentState(
                        player_id=i,
                        opponent_id=1 - i,
                        memory_level=config.agent_params.memory_level,
                        time_decay_turns=config.agent_params.time_decay_turns,
                        initial_hand_size=len(initial_hands[i]),
                        config=config,
                    )
                    agent.initialize(initial_obs, initial_hands[i], initial_peeks[i])
                    initial_agent_states.append(agent)
            except (AgentStateError, ObservationUpdateError, EncodingError) as agent_init_e:
                if logger_instance:
                    logger_instance.warning(
                        "W%d Iter %d: Agent state initialization error: %s",
                        worker_id,
                        iteration,
                        agent_init_e,
                    )
                worker_stats.error_count += 1
                return DeepCFRWorkerResult(
                    stats=worker_stats,
                    simulation_nodes=simulation_nodes_this_sim,
                )
            except Exception as agent_init_e:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                if logger_instance:
                    logger_instance.error(
                        "W%d Iter %d: Failed AgentStates init: %s",
                        worker_id,
                        iteration,
                        agent_init_e,
                        exc_info=True,
                    )
                worker_stats.error_count += 1
                return DeepCFRWorkerResult(
                    stats=worker_stats,
                    simulation_nodes=simulation_nodes_this_sim,
                )
        else:
            if logger_instance:
                logger_instance.warning(
                    "W%d Iter %d: Game terminal at init.",
                    worker_id,
                    iteration,
                )
            final_utility = np.array(
                [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
            )
            return DeepCFRWorkerResult(
                stats=worker_stats,
                simulation_nodes=simulation_nodes_this_sim,
                final_utility=final_utility.tolist(),
            )

        if len(initial_agent_states) != NUM_PLAYERS:
            if logger_instance:
                logger_instance.error(
                    "W%d Iter %d: Incorrect agent states (%d).",
                    worker_id,
                    iteration,
                    len(initial_agent_states),
                )
            worker_stats.error_count += 1
            return DeepCFRWorkerResult(
                stats=worker_stats,
                simulation_nodes=simulation_nodes_this_sim,
            )

        # Alternate updating player each iteration
        updating_player = iteration % NUM_PLAYERS
        min_depth_after_bottom_out_tracker = [float("inf")]
        has_bottomed_out_tracker = [False]

        # Run external sampling traversal
        final_utility_value = _deep_traverse(
            game_state=game_state,
            agent_states=initial_agent_states,
            updating_player=updating_player,
            network_weights=network_weights,
            iteration=iteration,
            config=config,
            network_config=network_config,
            advantage_samples=advantage_samples,
            strategy_samples=strategy_samples,
            depth=0,
            worker_stats=worker_stats,
            progress_queue=progress_queue,
            worker_id=worker_id,
            min_depth_after_bottom_out_tracker=min_depth_after_bottom_out_tracker,
            has_bottomed_out_tracker=has_bottomed_out_tracker,
            simulation_nodes=simulation_nodes_this_sim,
        )

        if final_utility_value is None or len(final_utility_value) != NUM_PLAYERS:
            if logger_instance:
                logger_instance.error(
                    "W%d Iter %d: Traversal returned invalid utility: %s.",
                    worker_id,
                    iteration,
                    final_utility_value,
                )
            worker_stats.error_count += 1
            final_utility_value = np.zeros(NUM_PLAYERS, dtype=np.float64)

        worker_stats.min_depth_after_bottom_out = (
            int(min_depth_after_bottom_out_tracker[0])
            if min_depth_after_bottom_out_tracker[0] != float("inf")
            else 0
        )

        if logger_instance:
            logger_instance.info(
                "W%d Iter %d: Traversal complete. Adv samples: %d, Strat samples: %d, Nodes: %d",
                worker_id,
                iteration,
                len(advantage_samples),
                len(strategy_samples),
                worker_stats.nodes_visited,
            )

        return DeepCFRWorkerResult(
            advantage_samples=advantage_samples,
            strategy_samples=strategy_samples,
            stats=worker_stats,
            simulation_nodes=simulation_nodes_this_sim,
            final_utility=final_utility_value.tolist(),
        )

    except KeyboardInterrupt:
        if logger_instance:
            logger_instance.warning(
                "W%d Iter %d received KeyboardInterrupt.", worker_id, iteration
            )
        worker_stats.error_count += 1
        return DeepCFRWorkerResult(
            advantage_samples=advantage_samples,
            strategy_samples=strategy_samples,
            stats=worker_stats,
            simulation_nodes=simulation_nodes_this_sim,
        )
    except Exception as e_inner:  # JUSTIFIED: worker resilience - top-level worker catch to prevent pool crash
        worker_stats.error_count += 1
        if logger_instance:
            logger_instance.critical(
                "!!! Unhandled Error W%d Iter %d: %s !!!",
                worker_id,
                iteration,
                e_inner,
                exc_info=True,
            )
        print(
            f"!!! FATAL DEEP WORKER ERROR W{worker_id} Iter {iteration}: {e_inner} !!!",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        return DeepCFRWorkerResult(
            advantage_samples=advantage_samples,
            strategy_samples=strategy_samples,
            stats=worker_stats,
            simulation_nodes=simulation_nodes_this_sim,
        )
    finally:
        if logger_instance:
            for handler in logger_instance.handlers[:]:
                if hasattr(handler, "flush"):
                    try:
                        handler.flush()
                    except Exception:
                        pass
                if hasattr(handler, "close"):
                    try:
                        handler.close()
                    except Exception:
                        pass
