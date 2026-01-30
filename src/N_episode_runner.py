"""
Run N episodes of an agent and collect statistics.

This module provides functionality to run multiple episodes and aggregate
performance metrics across all episodes.
"""

from typing import Dict, Any
from warehouse_env import WarehouseEnv
from warehouse_agent_reflex import ReflexWarehouseAgent


def run_n_episodes(
    n_episodes: int,
    agent_class=None,
    max_steps: int = 200,
    randomize: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run N episodes of an agent and collect statistics.
    
    Args:
        n_episodes: Number of episodes to run
        agent_class: Agent class to instantiate (default: ReflexWarehouseAgent)
        max_steps: Maximum steps per episode
        randomize: Whether to randomize pickup/dropoff positions
        verbose: Whether to print per-episode statistics
    
    Returns:
        Dictionary containing aggregated statistics:
        {
            "n_episodes": int,
            "successful_episodes": int,
            "success_rate": float,
            "avg_episode_length": float,
            "min_episode_length": int,
            "max_episode_length": int,
            "avg_battery_used": float,
            "min_battery_used": int,
            "max_battery_used": int,
            "avg_reward": float,
            "min_reward": float,
            "max_reward": float,
            "total_reward": float,
            "episode_details": list[dict],
        }
    """
    if agent_class is None:
        agent_class = ReflexWarehouseAgent
    
    # Initialize tracking variables
    episode_details = []
    successful_episodes = 0
    episode_lengths = []
    battery_used_list = []
    episode_rewards = []
    
    print(f"Running {n_episodes} episodes...")
    print("=" * 80)
    
    for episode_num in range(1, n_episodes + 1):
        # Create fresh environment and agent for each episode
        env = WarehouseEnv(max_steps=max_steps)
        agent = agent_class()
        
        # Reset environment
        obs = env.reset(randomize=randomize)
        initial_battery = obs["battery"]
        
        # Run episode
        total_reward = 0.0
        step_count = 0
        success = False
        
        while step_count < max_steps:
            # Agent decides on action
            action = agent.act(obs)
            
            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if terminated:
                success = True
                break
            
            if truncated:
                break
        
        # Calculate statistics for this episode
        battery_used = initial_battery - obs["battery"]
        
        if success:
            successful_episodes += 1
        
        episode_lengths.append(step_count)
        battery_used_list.append(battery_used)
        episode_rewards.append(total_reward)
        
        # Store episode details
        episode_details.append({
            "episode": episode_num,
            "success": success,
            "steps": step_count,
            "battery_used": battery_used,
            "reward": total_reward,
        })
        
        # Print per-episode stats if verbose
        if verbose:
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"Episode {episode_num:3d}: {status} | Steps: {step_count:3d} | "
                  f"Battery Used: {battery_used:3d} | Reward: {total_reward:8.2f}")
    
    # Calculate aggregate statistics
    success_rate = successful_episodes / n_episodes if n_episodes > 0 else 0.0
    avg_episode_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0
    avg_battery_used = sum(battery_used_list) / len(battery_used_list) if battery_used_list else 0.0
    avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
    
    print("=" * 80)
    print(f"\nAggregate Results ({n_episodes} episodes):")
    print(f"  Successful episodes: {successful_episodes}/{n_episodes} ({success_rate*100:.1f}%)")
    print(f"  Episode length: avg={avg_episode_length:.1f}, min={min(episode_lengths)}, max={max(episode_lengths)}")
    print(f"  Battery used: avg={avg_battery_used:.1f}, min={min(battery_used_list)}, max={max(battery_used_list)}")
    print(f"  Reward: avg={avg_reward:.2f}, min={min(episode_rewards):.2f}, max={max(episode_rewards):.2f}, total={sum(episode_rewards):.2f}")
    print()
    
    # Return comprehensive results dictionary
    return {
        "n_episodes": n_episodes,
        "successful_episodes": successful_episodes,
        "success_rate": success_rate,
        "avg_episode_length": avg_episode_length,
        "min_episode_length": min(episode_lengths) if episode_lengths else 0,
        "max_episode_length": max(episode_lengths) if episode_lengths else 0,
        "avg_battery_used": avg_battery_used,
        "min_battery_used": min(battery_used_list) if battery_used_list else 0,
        "max_battery_used": max(battery_used_list) if battery_used_list else 0,
        "avg_reward": avg_reward,
        "min_reward": min(episode_rewards) if episode_rewards else 0.0,
        "max_reward": max(episode_rewards) if episode_rewards else 0.0,
        "total_reward": sum(episode_rewards),
        "episode_details": episode_details,
    }


def main():
    """Main entry point for running N episodes."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run N episodes of warehouse agent and collect statistics"
    )
    parser.add_argument(
        "-n",
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode (default: 200)",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize pickup/dropoff positions for each episode",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-episode statistics",
    )
    
    args = parser.parse_args()
    
    results = run_n_episodes(
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        randomize=args.randomize,
        verbose=args.verbose,
    )
    
    return results


if __name__ == "__main__":
    main()
