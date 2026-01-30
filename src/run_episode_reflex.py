"""
Run an episode with the reflex agent and visualize using warehouse_viz.

This script:
1. Creates the warehouse environment
2. Runs the reflex agent for an episode
3. Collects frames and metrics
4. Displays an interactive animation
"""

from warehouse_env import WarehouseEnv
from warehouse_agent_reflex import ReflexWarehouseAgent
from warehouse_viz import replay_animation, save_frames_to_svg


def manhattan_distance(pos1: tuple, pos2: tuple) -> int:
    """Calculate Manhattan distance between two positions."""
    if pos1 is None or pos2 is None:
        return 0
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def run_episode_with_visualization(
    max_steps: int = 200,
    randomize: bool = False,
    save_svg: bool = False,
    svg_dir: str = "warehouse_frames",
) -> None:
    """
    Run an episode with the reflex agent and visualize the results.
    
    Args:
        max_steps: Maximum number of steps to run
        randomize: Whether to randomize pickup/dropoff positions
        save_svg: Whether to save frames as SVG files
        svg_dir: Directory to save SVG frames to
    """
    # Initialize environment and agent
    env = WarehouseEnv(max_steps=max_steps)
    agent = ReflexWarehouseAgent()
    
    # Reset environment
    obs = env.reset(randomize=randomize)
    
    # Collections for visualization
    frames = []
    metrics = {
        "rewards": [],
        "battery": [],
        "dist_pickup": [],
        "dist_dropoff": [],
    }
    
    # Capture initial frame
    frames.append(env.render_grid())
    metrics["rewards"].append(0.0)
    metrics["battery"].append(obs["battery"])
    metrics["dist_pickup"].append(manhattan_distance(obs["robot_pos"], obs["pickup_pos"]))
    metrics["dist_dropoff"].append(manhattan_distance(obs["robot_pos"], obs["dropoff_pos"]))
    
    # Run episode
    total_reward = 0.0
    step_count = 0
    
    print("Warehouse Reflex Agent Episode")
    print("=" * 70)
    print(f"Initial state:")
    print(f"  Robot position: {obs['robot_pos']}")
    print(f"  Pickup location: {obs['pickup_pos']}")
    print(f"  Dropoff location: {obs['dropoff_pos']}")
    print(f"  Battery: {obs['battery']}")
    print("=" * 70)
    print()
    
    while step_count < max_steps:
        # Agent decides on action
        action = agent.act(obs)
        
        # Take action in environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Collect frame and metrics
        frames.append(env.render_grid())
        metrics["rewards"].append(reward)
        metrics["battery"].append(obs["battery"])
        metrics["dist_pickup"].append(manhattan_distance(obs["robot_pos"], obs["pickup_pos"]))
        metrics["dist_dropoff"].append(manhattan_distance(obs["robot_pos"], obs["dropoff_pos"]))
        
        # Print step info
        print(f"Step {step_count:3d}: Action={action:6s} | Reward={reward:7.2f} | "
              f"Total={total_reward:8.2f} | Battery={obs['battery']:3d} | "
              f"Has Item={str(obs['has_item']):5s}")
        
        if terminated:
            print("\n✓ SUCCESS: Item picked up and dropped off!")
            break
        
        if truncated:
            print("\n✗ EPISODE TRUNCATED: Max steps or battery depleted")
            break
    
    print("\n" + "=" * 70)
    print(f"Episode Summary:")
    print(f"  Total steps: {step_count}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Battery remaining: {obs['battery']}")
    print(f"  Item delivered: {not obs['has_item'] and step_count < max_steps}")
    print("=" * 70)
    print()
    
    # Save frames to SVG if requested
    if save_svg:
        print(f"Saving frames to {svg_dir}/...")
        save_frames_to_svg(frames, svg_dir)
        print(f"Saved {len(frames)} frames to {svg_dir}/")
        print()
    
    # Display animation
    print("Launching animation viewer...")
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  LEFT/RIGHT arrows: Step backward/forward (when paused)")
    print()
    replay_animation(frames, metrics=metrics, interval_ms=200, speed=1.0)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run warehouse reflex agent episode with visualization"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps to run (default: 200)",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize pickup/dropoff positions",
    )
    parser.add_argument(
        "--save-svg",
        action="store_true",
        help="Save frames as SVG files",
    )
    parser.add_argument(
        "--svg-dir",
        default="warehouse_frames",
        help="Directory to save SVG frames (default: warehouse_frames)",
    )
    
    args = parser.parse_args()
    
    run_episode_with_visualization(
        max_steps=args.max_steps,
        randomize=args.randomize,
        save_svg=args.save_svg,
        svg_dir=args.svg_dir,
    )


if __name__ == "__main__":
    main()
