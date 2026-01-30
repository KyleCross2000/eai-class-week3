"""
Run a single episode with the greedy Manhattan agent and log results.

This script:
1. Resets the environment and initializes the agent to a random position
2. Steps through one full episode using the agent
3. Renders the episode with warehouse animation/visualization using warehouse_viz
4. Logs total reward, final battery, and episode length
"""

import sys
from typing import List, Tuple, Dict
from warehouse_env import WarehouseEnv
from warehouse_agent_greedy import GreedyAgent
from warehouse_viz import replay_animation


class EpisodeRecorder:
    """Records trajectory data for visualization and analysis."""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.grids = []
        self.battery_levels = []
        self.dist_to_pickup = []
        self.dist_to_dropoff = []

    def record_step(self, obs: Dict[str, object], action: str, reward: float, grid: List[List[str]]):
        """Record a step in the episode."""
        self.observations.append(obs.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.grids.append([row[:] for row in grid])  # Deep copy of grid
        self.battery_levels.append(obs["battery"])
        
        # Calculate distances
        pickup_dist = self._manhattan_distance(obs["robot_pos"], obs["pickup_pos"])
        dropoff_dist = self._manhattan_distance(obs["robot_pos"], obs["dropoff_pos"])
        self.dist_to_pickup.append(pickup_dist)
        self.dist_to_dropoff.append(dropoff_dist)

    def record_initial(self, obs: Dict[str, object], grid: List[List[str]]):
        """Record the initial state."""
        self.observations.append(obs.copy())
        self.actions.append(None)
        self.rewards.append(0.0)
        self.grids.append([row[:] for row in grid])
        self.battery_levels.append(obs["battery"])
        
        # Calculate distances
        pickup_dist = self._manhattan_distance(obs["robot_pos"], obs["pickup_pos"])
        dropoff_dist = self._manhattan_distance(obs["robot_pos"], obs["dropoff_pos"])
        self.dist_to_pickup.append(pickup_dist)
        self.dist_to_dropoff.append(dropoff_dist)

    @staticmethod
    def _manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int] | None) -> int:
        """Compute Manhattan distance between two positions."""
        if pos2 is None:
            return 0
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_metrics(self) -> Dict[str, List]:
        """Return metrics dictionary for visualization."""
        return {
            "rewards": self.rewards,
            "battery": self.battery_levels,
            "dist_pickup": self.dist_to_pickup,
            "dist_dropoff": self.dist_to_dropoff,
        }



def run_episode_with_recording(env: WarehouseEnv, agent: GreedyAgent, randomize: bool = True) -> Tuple[float, int, int, EpisodeRecorder]:
    """
    Run a single episode, recording all states for visualization.

    Args:
        env: WarehouseEnv instance
        agent: GreedyAgent instance
        randomize: If True, start at a random position

    Returns:
        Tuple of (total_reward, steps_taken, final_battery, episode_recorder)
    """
    obs = env.reset(randomize=randomize)
    agent.reset()

    recorder = EpisodeRecorder()
    grid = env.render_grid()
    recorder.record_initial(obs, grid)

    total_reward = 0.0
    steps = 0

    # Run episode until termination or truncation
    while True:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        grid = env.render_grid()
        recorder.record_step(obs, action, reward, grid)

        if terminated or truncated:
            break

    final_battery = obs["battery"]
    return total_reward, steps, final_battery, recorder


def log_episode_summary(total_reward: float, steps: int, final_battery: int, initial_obs: Dict[str, object], final_obs: Dict[str, object]):
    """Log a summary of the episode."""
    print("\n" + "="*60)
    print("EPISODE SUMMARY")
    print("="*60)
    print(f"Total reward:      {total_reward:+.2f}")
    print(f"Episode length:    {steps} steps")
    print(f"Final battery:     {final_battery} / {initial_obs['battery']}")
    print(f"Battery consumed:  {initial_obs['battery'] - final_battery} units")
    print(f"Starting position: {initial_obs['robot_pos']}")
    print(f"Final position:    {final_obs['robot_pos']}")
    print(f"Item delivered:    {'Yes' if not final_obs['has_item'] else 'No (still carrying)'}")
    print("="*60 + "\n")


def main():
    """Main entry point."""
    print("Initializing warehouse environment and greedy agent...")
    env = WarehouseEnv()
    agent = GreedyAgent(env, loop_history_size=10, escape_steps=5)

    print("Running episode with random starting position...\n")
    total_reward, steps, final_battery, recorder = run_episode_with_recording(env, agent, randomize=True)

    # Get initial and final observations for logging
    initial_obs = recorder.observations[0]
    final_obs = recorder.observations[-1]

    # Log results
    log_episode_summary(total_reward, steps, final_battery, initial_obs, final_obs)

    # Render animation with warehouse_viz
    print("Launching animated visualization...")
    metrics = recorder.get_metrics()
    replay_animation(recorder.grids, metrics=metrics, interval_ms=150, speed=1.0)


if __name__ == "__main__":
    main()
