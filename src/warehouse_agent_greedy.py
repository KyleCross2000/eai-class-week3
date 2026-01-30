"""
Greedy agent for the warehouse environment.

Implements a simple greedy policy that:
1. Computes Manhattan distance to the current goal (pickup if no item, dropoff if carrying)
2. Chooses the action (N/S/E/W) that reduces distance
3. Falls back to a random valid move if stuck (all moves increase distance)
4. Detects loops by tracking the last N positions; triggers random escape if stuck in a loop
"""

import random
from collections import deque
from typing import Dict, List, Tuple
from warehouse_env import WarehouseEnv


class GreedyAgent:
    """
    A greedy agent that navigates to pickup/dropoff locations in the warehouse.
    """

    def __init__(self, env: WarehouseEnv, loop_history_size: int = 10, escape_steps: int = 5):
        """
        Initialize the greedy agent.

        Args:
            env: WarehouseEnv instance
            loop_history_size: Number of recent positions to track for loop detection
            escape_steps: Number of random steps to take when a loop is detected
        """
        self.env = env
        self.loop_history_size = loop_history_size
        self.escape_steps = escape_steps
        self.position_history = deque(maxlen=loop_history_size)
        self.escape_countdown = 0

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int] | None) -> int:
        """Compute Manhattan distance between two positions."""
        if pos2 is None:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_current_goal(self, obs: Dict[str, object]) -> Tuple[int, int] | None:
        """
        Determine the current goal position.

        Returns:
            Position of pickup if no item, position of dropoff if carrying, None if unknown
        """
        if not obs["has_item"]:
            return obs["pickup_pos"]
        else:
            return obs["dropoff_pos"]

    def _is_valid_move(self, obs: Dict[str, object], action: str) -> bool:
        """Check if a move action would hit a wall by checking the local grid."""
        local_grid = obs["local_grid"]
        
        # Local grid is a list of strings, with robot at center
        center = len(local_grid) // 2

        deltas = {
            "N": (-1, 0),
            "E": (0, 1),
            "S": (1, 0),
            "W": (0, -1),
        }

        if action not in deltas:
            return False

        dr, dc = deltas[action]
        new_local_r = center + dr
        new_local_c = center + dc

        # Check bounds in local grid
        if (new_local_r < 0 or new_local_r >= len(local_grid) or
            new_local_c < 0 or new_local_c >= len(local_grid[new_local_r])):
            return False

        cell = local_grid[new_local_r][new_local_c]
        return cell != "#"

    def _get_valid_moves(self, obs: Dict[str, object]) -> List[str]:
        """Get list of valid move actions."""
        valid = []
        for action in ["N", "E", "S", "W"]:
            if self._is_valid_move(obs, action):
                valid.append(action)
        return valid

    def _get_next_position(self, current_pos: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Compute the next position given an action."""
        deltas = {
            "N": (-1, 0),
            "E": (0, 1),
            "S": (1, 0),
            "W": (0, -1),
        }
        if action not in deltas:
            return current_pos
        dr, dc = deltas[action]
        return (current_pos[0] + dr, current_pos[1] + dc)

    def _detect_loop(self, current_pos: Tuple[int, int]) -> bool:
        """Check if current position was recently visited (within position_history)."""
        return current_pos in self.position_history

    def _choose_action(self, obs: Dict[str, object]) -> str:
        """
        Choose an action using the greedy policy.

        Returns:
            Action string: "N", "E", "S", "W", "WAIT", "PICK", or "DROP"
        """
        current_pos = obs["robot_pos"]
        goal_pos = self._get_current_goal(obs)

        # Handle special actions
        if not obs["has_item"] and current_pos == obs["pickup_pos"]:
            return "PICK"
        elif obs["has_item"] and current_pos == obs["dropoff_pos"]:
            return "DROP"

        # Update position history for loop detection
        self.position_history.append(current_pos)

        # Check for loop and trigger escape if needed
        if self.escape_countdown > 0:
            self.escape_countdown -= 1
            valid_moves = self._get_valid_moves(obs)
            if valid_moves:
                return random.choice(valid_moves)
            return "WAIT"

        if self._detect_loop(current_pos) and len(self.position_history) == self.loop_history_size:
            # Trigger escape sequence
            self.escape_countdown = self.escape_steps
            valid_moves = self._get_valid_moves(obs)
            if valid_moves:
                return random.choice(valid_moves)
            return "WAIT"

        # Greedy policy: choose move that reduces distance to goal
        if goal_pos is None:
            return "WAIT"

        valid_moves = self._get_valid_moves(obs)
        if not valid_moves:
            return "WAIT"

        current_distance = self._manhattan_distance(current_pos, goal_pos)

        # Find move that reduces distance
        best_action = None
        best_distance = current_distance

        for action in valid_moves:
            next_pos = self._get_next_position(current_pos, action)
            next_distance = self._manhattan_distance(next_pos, goal_pos)

            if next_distance < best_distance:
                best_distance = next_distance
                best_action = action

        # If found a move that reduces distance, use it
        if best_action is not None:
            return best_action

        # If stuck (all moves increase distance), use random move
        return random.choice(valid_moves)

    def act(self, obs: Dict[str, object]) -> str:
        """
        Select an action given the current observation.

        Args:
            obs: Observation dictionary from env.step() or env.reset()

        Returns:
            Action string
        """
        return self._choose_action(obs)

    def reset(self) -> None:
        """Reset the agent's internal state."""
        self.position_history.clear()
        self.escape_countdown = 0


def run_episode(env: WarehouseEnv, agent: GreedyAgent, max_steps: int = 500, verbose: bool = False) -> Tuple[float, int]:
    """
    Run a single episode with the agent.

    Args:
        env: WarehouseEnv instance
        agent: GreedyAgent instance
        max_steps: Maximum steps per episode
        verbose: Print debug info

    Returns:
        Tuple of (total_reward, steps_taken)
    """
    obs = env.reset()
    agent.reset()
    total_reward = 0.0
    steps = 0

    for step in range(max_steps):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if verbose:
            print(f"Step {steps}: action={action}, reward={reward:.2f}, pos={obs['robot_pos']}, "
                  f"has_item={obs['has_item']}, battery={obs['battery']}")

        if terminated or truncated:
            break

    if verbose:
        print(f"Episode finished: total_reward={total_reward:.2f}, steps={steps}")

    return total_reward, steps


if __name__ == "__main__":
    # Simple test
    env = WarehouseEnv()
    agent = GreedyAgent(env)

    print("Running single episode with verbose output:\n")
    reward, steps = run_episode(env, agent, verbose=True)
    print(f"\nFinal result: reward={reward:.2f}, steps={steps}")

    # Run a few episodes to check performance
    print("\n" + "="*50)
    print("Running 5 episodes for evaluation:\n")
    rewards = []
    step_counts = []
    for episode in range(5):
        r, s = run_episode(env, agent)
        rewards.append(r)
        step_counts.append(s)
        print(f"Episode {episode+1}: reward={r:.2f}, steps={s}")

    print(f"\nAverage reward: {sum(rewards)/len(rewards):.2f}")
    print(f"Average steps: {sum(step_counts)/len(step_counts):.2f}")
