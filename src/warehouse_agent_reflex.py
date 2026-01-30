"""
Simple reflex agent for the warehouse environment.

Uses condition-action rules based on:
- Current robot position
- Goal positions (pickup and dropoff)
- Whether the robot carries an item
- Includes all 8 directional movement rules (N, NE, SE, S, SW, W, NW, NE)
"""

from typing import Dict
from warehouse_env import WarehouseEnv


class ReflexWarehouseAgent:
    """
    A simple reflex agent that makes decisions based on current observations
    without maintaining internal state or planning ahead.
    """
    
    def __init__(self):
        """Initialize the reflex agent."""
        pass
    
    def act(self, observation: Dict[str, object]) -> str:
        """
        Decide on an action based on current observation.
        
        Condition-action rules:
        1. If at pickup and no item -> PICK
        2. If at dropoff and has item -> DROP
        3. If has item and not at dropoff -> Move toward dropoff
        4. If no item and not at pickup -> Move toward pickup
        5. If neither item nor goal reachable -> WAIT (fallback)
        
        Movement uses 8 directional rules:
        N, NE, E, SE, S, SW, W, NW (with cardinal directions as fallback)
        """
        robot_pos = observation["robot_pos"]
        has_item = observation["has_item"]
        pickup_pos = observation["pickup_pos"]
        dropoff_pos = observation["dropoff_pos"]
        
        # Rule 1: At pickup location and empty-handed -> PICK
        if robot_pos == pickup_pos and not has_item:
            return "PICK"
        
        # Rule 2: At dropoff location and carrying item -> DROP
        if robot_pos == dropoff_pos and has_item:
            return "DROP"
        
        # Rule 3: Carrying item -> Move toward dropoff
        if has_item and dropoff_pos:
            return self._move_toward(robot_pos, dropoff_pos)
        
        # Rule 4: Not carrying item -> Move toward pickup
        if not has_item and pickup_pos:
            return self._move_toward(robot_pos, pickup_pos)
        
        # Rule 5: Fallback action
        return "WAIT"
    
    def _move_toward(self, current_pos: tuple, goal_pos: tuple) -> str:
        """
        Calculate the direction to move toward the goal using 8-directional rules.
        
        Returns one of: N, NE, E, SE, S, SW, W, NW
        Falls back to cardinal directions if needed.
        """
        curr_r, curr_c = current_pos
        goal_r, goal_c = goal_pos
        
        # Calculate row and column deltas
        dr = 0 if goal_r == curr_r else (1 if goal_r > curr_r else -1)
        dc = 0 if goal_c == curr_c else (1 if goal_c > curr_c else -1)
        
        # 8-directional movement rules
        if dr == -1 and dc == -1:
            return "NW"  # North-West
        elif dr == -1 and dc == 0:
            return "N"   # North
        elif dr == -1 and dc == 1:
            return "NE"  # North-East
        elif dr == 0 and dc == 1:
            return "E"   # East
        elif dr == 1 and dc == 1:
            return "SE"  # South-East
        elif dr == 1 and dc == 0:
            return "S"   # South
        elif dr == 1 and dc == -1:
            return "SW"  # South-West
        elif dr == 0 and dc == -1:
            return "W"   # West
        else:
            return "WAIT"  # Already at goal


def main():
    """Run a simple demo of the reflex agent."""
    # Create environment and agent
    env = WarehouseEnv()
    agent = ReflexWarehouseAgent()
    
    # Reset environment
    obs = env.reset()
    
    print("Warehouse Reflex Agent Demo")
    print("=" * 50)
    print(env.render_with_legend())
    print()
    
    # Run episode
    total_reward = 0
    step_count = 0
    max_steps = 100
    
    while step_count < max_steps:
        # Agent decides on action
        action = agent.act(obs)
        
        # Take action in environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        print(f"Step {step_count}: Action = {action:6s} | Reward = {reward:6.2f} | "
              f"Total = {total_reward:7.2f} | Has Item = {obs['has_item']}")
        
        if terminated:
            print("\n✓ Goal achieved! Item picked up and dropped off successfully!")
            break
        
        if truncated:
            print("\n✗ Episode truncated (max steps or battery depleted)")
            break
    
    print("\n" + "=" * 50)
    print(f"Episode finished in {step_count} steps with total reward: {total_reward:.2f}")
    print(env.render_with_legend())


if __name__ == "__main__":
    main()
