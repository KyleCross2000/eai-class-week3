"""
Compare performance of different warehouse agents.

Runs N episodes for each agent and visualizes results with:
- Bar chart of success rates
- Box plots of episode lengths
- Histograms of final battery levels
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
from warehouse_env import WarehouseEnv
from warehouse_agent_reflex import ReflexWarehouseAgent
from warehouse_agent_greedy import GreedyAgent
from N_episode_runner import run_n_episodes


def run_agent_comparison(
    n_episodes: int = 50,
    max_steps: int = 200,
    randomize: bool = False,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Run episodes for both reflex and greedy agents and collect results.
    
    Args:
        n_episodes: Number of episodes to run for each agent
        max_steps: Maximum steps per episode
        randomize: Whether to randomize pickup/dropoff positions
        verbose: Whether to print per-episode statistics
    
    Returns:
        Dictionary with results for each agent:
        {
            "reflex": {...results...},
            "greedy": {...results...},
        }
    """
    results = {}
    
    print("=" * 80)
    print("REFLEX AGENT EVALUATION")
    print("=" * 80)
    results["reflex"] = run_n_episodes(
        n_episodes=n_episodes,
        agent_class=ReflexWarehouseAgent,
        max_steps=max_steps,
        randomize=randomize,
        verbose=verbose,
    )
    
    print("\n" + "=" * 80)
    print("GREEDY AGENT EVALUATION")
    print("=" * 80)
    
    # Create a wrapper class for GreedyAgent to match the interface expected by run_n_episodes
    class GreedyAgentWrapper:
        def __init__(self):
            # We'll set env in act() method
            self.agent = None
            self.env = None
        
        def act(self, observation: Dict[str, object]) -> str:
            if self.agent is None:
                # Create environment and agent on first call
                self.env = WarehouseEnv()
                self.agent = GreedyAgent(self.env)
            return self.agent.act(observation)
    
    # Monkey-patch run_n_episodes to work with GreedyAgent
    def run_n_episodes_greedy(
        n_episodes: int,
        agent_class=None,
        max_steps: int = 200,
        randomize: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Modified run_n_episodes for GreedyAgent."""
        episode_details = []
        successful_episodes = 0
        episode_lengths = []
        battery_used_list = []
        episode_rewards = []
        
        if verbose:
            print(f"Running {n_episodes} episodes...")
            print("=" * 80)
        
        for episode_num in range(1, n_episodes + 1):
            # Create fresh environment and agent for each episode
            env = WarehouseEnv(max_steps=max_steps)
            agent = GreedyAgent(env)
            
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
        
        if verbose:
            print("=" * 80)
            print(f"\nAggregate Results ({n_episodes} episodes):")
            print(f"  Successful episodes: {successful_episodes}/{n_episodes} ({success_rate*100:.1f}%)")
            print(f"  Episode length: avg={avg_episode_length:.1f}, min={min(episode_lengths)}, max={max(episode_lengths)}")
            print(f"  Battery used: avg={avg_battery_used:.1f}, min={min(battery_used_list)}, max={max(battery_used_list)}")
            print(f"  Reward: avg={avg_reward:.2f}, min={min(episode_rewards):.2f}, max={max(episode_rewards):.2f}, total={sum(episode_rewards):.2f}")
            print()
        
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
    
    results["greedy"] = run_n_episodes_greedy(
        n_episodes=n_episodes,
        max_steps=max_steps,
        randomize=randomize,
        verbose=verbose,
    )
    
    return results


def visualize_comparison(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Visualize agent comparison with 3 subplots:
    1. Bar chart of success rates
    2. Box plots of episode lengths
    3. Histograms of final battery levels
    
    Args:
        results: Dictionary with results for each agent from run_agent_comparison()
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Warehouse Agent Comparison", fontsize=16, fontweight="bold")
    
    agent_names = list(results.keys())
    agents_display = ["Reflex", "Greedy"]  # Display names
    
    # Extract episode details for each agent
    episode_data = {}
    for agent in agent_names:
        episode_data[agent] = results[agent]["episode_details"]
    
    # ========== SUBPLOT 1: Success Rate Bar Chart ==========
    ax1 = axes[0]
    success_rates = [results[agent]["success_rate"] * 100 for agent in agent_names]
    colors = ["#2ecc71", "#3498db"]
    bars = ax1.bar(agents_display, success_rates, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.set_ylabel("Success Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Success Rate Comparison", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 110)
    ax1.grid(axis="y", alpha=0.3)
    
    # ========== SUBPLOT 2: Episode Length Box Plots ==========
    ax2 = axes[1]
    episode_lengths = [
        [ep["steps"] for ep in episode_data[agent]]
        for agent in agent_names
    ]
    bp = ax2.boxplot(episode_lengths, labels=agents_display, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel("Episode Length (steps)", fontsize=11, fontweight="bold")
    ax2.set_title("Episode Length Distribution", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    
    # ========== SUBPLOT 3: Final Battery Level Histograms ==========
    ax3 = axes[2]
    
    # Calculate final battery for each episode
    final_batteries = []
    for agent in agent_names:
        batteries = []
        for ep in episode_data[agent]:
            battery_remaining = 200 - ep["battery_used"]  # Assuming max_battery=200
            batteries.append(battery_remaining)
        final_batteries.append(batteries)
    
    # Create overlapping histograms
    bins = np.linspace(0, 200, 21)
    ax3.hist(final_batteries[0], bins=bins, label=agents_display[0], 
            color=colors[0], alpha=0.6, edgecolor="black")
    ax3.hist(final_batteries[1], bins=bins, label=agents_display[1], 
            color=colors[1], alpha=0.6, edgecolor="black")
    
    ax3.set_xlabel("Final Battery Level", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax3.set_title("Final Battery Distribution", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_comparison_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a summary comparison of agent performance."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()
    
    for agent_name in results.keys():
        agent_display = "Reflex" if agent_name == "reflex" else "Greedy"
        stats = results[agent_name]
        
        print(f"{agent_display} Agent:")
        print(f"  Success Rate: {stats['success_rate']*100:.1f}% ({stats['successful_episodes']}/{stats['n_episodes']})")
        print(f"  Episode Length: {stats['avg_episode_length']:.1f} steps (min={stats['min_episode_length']}, max={stats['max_episode_length']})")
        print(f"  Battery Used: {stats['avg_battery_used']:.1f} (min={stats['min_battery_used']}, max={stats['max_battery_used']})")
        print(f"  Reward: avg={stats['avg_reward']:.2f}, min={stats['min_reward']:.2f}, max={stats['max_reward']:.2f}")
        print()


def main():
    """Main entry point for agent comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare warehouse agents (reflex vs greedy)"
    )
    parser.add_argument(
        "-n",
        "--episodes",
        type=int,
        default=50,
        help="Number of episodes per agent (default: 50)",
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
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization (just print stats)",
    )
    
    args = parser.parse_args()
    
    # Run comparison
    results = run_agent_comparison(
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        randomize=args.randomize,
        verbose=args.verbose,
    )
    
    # Print summary
    print_comparison_summary(results)
    
    # Visualize unless skipped
    if not args.no_viz:
        print("Generating visualization...")
        visualize_comparison(results)


if __name__ == "__main__":
    main()
