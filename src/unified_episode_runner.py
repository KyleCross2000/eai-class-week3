"""
Unified episode runner for both reflex and greedy warehouse agents.

Provides a consistent interface for running episodes with different agent types
and collecting standardized metrics.
"""

from typing import Dict, Tuple, Any
from warehouse_env import WarehouseEnv
from warehouse_agent_reflex import WarehouseAgentReflex
from warehouse_agent_greedy import GreedyAgent


def run_single_episode(
    agent_class,
    env: WarehouseEnv = None,
    max_steps: int = 200,
    randomize: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a single episode with either reflex or greedy agent.
    
    Provides a unified interface that handles agent instantiation and 
    returns standardized results.
    
    Args:
        agent_class: Class of agent to use (WarehouseAgentReflex or GreedyAgent)
        env: WarehouseEnv instance (created if None)
        max_steps: Maximum steps per episode
        randomize: Whether to randomize pickup/dropoff positions
        verbose: Print step-by-step information
    
    Returns:
        Dictionary with episode results:
        {
            "total_reward": float,
            "steps": int,
            "battery_remaining": int,
            "success": bool,  # True if item was delivered
            "agent_type": str,
            "verbose_log": list,  # If verbose=True
        }
    """
    # Create environment if not provided
    if env is None:
        env = WarehouseEnv(max_steps=max_steps)
    
    # Instantiate agent (GreedyAgent requires env parameter)
    if agent_class == GreedyAgent:
        agent = agent_class(env)
    else:
        agent = agent_class()
    
    # Reset agent if it has reset method
    if hasattr(agent, 'reset'):
        agent.reset()
    
    # Reset environment
    obs = env.reset(randomize=randomize)
    initial_battery = obs["battery"]
    
    # Run episode
    total_reward = 0.0
    steps = 0
    verbose_log = []
    
    while steps < max_steps:
        # Agent decides on action
        action = agent.act(obs)
        
        # Take action in environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if verbose:
            log_entry = {
                "step": steps,
                "action": action,
                "reward": reward,
                "total_reward": total_reward,
                "battery": obs["battery"],
                "has_item": obs["has_item"],
                "pos": obs["robot_pos"],
            }
            verbose_log.append(log_entry)
            print(f"Step {steps:3d}: {action:6s} | R={reward:7.2f} | "
                  f"Total={total_reward:8.2f} | Battery={obs['battery']:3d} | "
                  f"Item={str(obs['has_item']):5s}")
        
        if terminated or truncated:
            break
    
    # Determine success (item was delivered and not still carrying)
    success = not obs["has_item"] and steps < max_steps
    
    results = {
        "total_reward": total_reward,
        "steps": steps,
        "battery_remaining": obs["battery"],
        "battery_used": initial_battery - obs["battery"],
        "success": success,
        "agent_type": agent_class.__name__,
        "terminated": success,  # For compatibility with previous code
    }
    
    if verbose:
        results["verbose_log"] = verbose_log
    
    return results


def run_n_episodes(
    agent_class,
    n_episodes: int,
    env: WarehouseEnv = None,
    max_steps: int = 200,
    randomize: bool = False,
    verbose: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    """
    Run N episodes with an agent and collect aggregate statistics.
    
    Args:
        agent_class: Class of agent to use
        n_episodes: Number of episodes to run
        env: WarehouseEnv instance (created fresh for each episode if None)
        max_steps: Maximum steps per episode
        randomize: Whether to randomize pickup/dropoff positions
        verbose: Print step-by-step information for each episode
        progress: Print progress every 10 episodes
    
    Returns:
        Dictionary with aggregate statistics:
        {
            "n_episodes": int,
            "successful": int,
            "success_rate": float,
            "avg_reward": float,
            "min_reward": float,
            "max_reward": float,
            "avg_steps": float,
            "min_steps": int,
            "max_steps": int,
            "avg_battery_used": float,
            "min_battery_used": int,
            "max_battery_used": int,
            "episodes": list[dict],  # Individual episode results
        }
    """
    episodes = []
    successful = 0
    rewards = []
    steps_list = []
    battery_used_list = []
    
    agent_type = agent_class.__name__
    print(f"Running {n_episodes} episodes with {agent_type}...")
    
    for episode_num in range(1, n_episodes + 1):
        # Create fresh environment for each episode if not reusing
        episode_env = WarehouseEnv(max_steps=max_steps) if env is None else env
        
        result = run_single_episode(
            agent_class,
            env=episode_env,
            max_steps=max_steps,
            randomize=randomize,
            verbose=verbose,
        )
        
        episodes.append({**result, "episode": episode_num})
        
        if result["success"]:
            successful += 1
        
        rewards.append(result["total_reward"])
        steps_list.append(result["steps"])
        battery_used_list.append(result["battery_used"])
        
        if progress and episode_num % 10 == 0:
            print(f"  Completed {episode_num}/{n_episodes} episodes")
    
    # Calculate statistics
    import numpy as np
    
    return {
        "n_episodes": n_episodes,
        "agent_type": agent_type,
        "successful": successful,
        "success_rate": successful / n_episodes if n_episodes > 0 else 0.0,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "min_reward": float(np.min(rewards)) if rewards else 0.0,
        "max_reward": float(np.max(rewards)) if rewards else 0.0,
        "total_reward": float(np.sum(rewards)),
        "avg_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "min_steps": int(np.min(steps_list)) if steps_list else 0,
        "max_steps": int(np.max(steps_list)) if steps_list else 0,
        "avg_battery_used": float(np.mean(battery_used_list)) if battery_used_list else 0.0,
        "min_battery_used": int(np.min(battery_used_list)) if battery_used_list else 0,
        "max_battery_used": int(np.max(battery_used_list)) if battery_used_list else 0,
        "episodes": episodes,
    }


def print_episode_summary(result: Dict[str, Any]) -> None:
    """Print a formatted summary of a single episode."""
    print("\n" + "=" * 60)
    print("EPISODE SUMMARY")
    print("=" * 60)
    print(f"Agent: {result['agent_type']}")
    print(f"Success: {'YES' if result['success'] else 'NO'}")
    print(f"Steps: {result['steps']}")
    print(f"Reward: {result['total_reward']:.2f}")
    print(f"Battery Used: {result['battery_used']} / {result['battery_used'] + result['battery_remaining']}")
    print(f"Battery Remaining: {result['battery_remaining']}")
    print("=" * 60 + "\n")


def print_comparison_summary(results_dict: Dict[str, Dict[str, Any]]) -> None:
    """Print a formatted comparison of multiple agents' results."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    for agent_name, stats in results_dict.items():
        print(f"\n{agent_name} Agent ({stats['n_episodes']} episodes):")
        print(f"  Success Rate: {stats['success_rate']*100:.1f}% ({stats['successful']}/{stats['n_episodes']})")
        print(f"  Avg Steps: {stats['avg_steps']:.1f} (min={stats['min_steps']}, max={stats['max_steps']})")
        print(f"  Avg Battery Used: {stats['avg_battery_used']:.1f} (min={stats['min_battery_used']}, max={stats['max_battery_used']})")
        print(f"  Avg Reward: {stats['avg_reward']:.2f} (min={stats['min_reward']:.2f}, max={stats['max_reward']:.2f})")
        print(f"  Total Reward: {stats['total_reward']:.2f}")
    
    print("\n" + "=" * 80 + "\n")


def main():
    """Demo: Run both agents and compare."""
    # Run reflex agent
    reflex_results = run_n_episodes(
        WarehouseAgentReflex,
        n_episodes=20,
        max_steps=200,
        verbose=False,
        progress=True,
    )
    
    # Run greedy agent
    greedy_results = run_n_episodes(
        GreedyAgent,
        n_episodes=20,
        max_steps=200,
        verbose=False,
        progress=True,
    )
    
    # Compare results
    results_dict = {
        "Reflex": reflex_results,
        "Greedy": greedy_results,
    }
    
    print_comparison_summary(results_dict)


if __name__ == "__main__":
    main()
