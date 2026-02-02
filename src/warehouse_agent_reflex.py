import random

class WarehouseAgentReflex:
    def __init__(self):
        pass
    
    def act(self, state):
        # Unpack state using correct keys from WarehouseEnv
        pos = state['robot_pos']
        carrying = state['has_item']
        pickup = state['pickup_pos']
        dropoff = state['dropoff_pos']
        
        # Valid actions are always ['N', 'E', 'S', 'W', 'WAIT', 'PICK', 'DROP']
        # We'll filter for valid moves by checking the environment, but here we assume all are possible
        valid_actions = []
        # Only allow moves that don't hit walls, and only allow PICK/DROP when appropriate
        # For simplicity, assume all actions are valid (as in the original agent), or adapt if env provides valid_actions
        if 'valid_actions' in state:
            valid_actions = state['valid_actions']
        else:
            valid_actions = ['N', 'E', 'S', 'W', 'WAIT', 'PICK', 'DROP']
        
        # Rule 1: At pickup and not carrying → PICK
        if pos == pickup and not carrying and 'PICK' in valid_actions:
            return 'PICK'
        
        # Rule 2: At dropoff and carrying → DROP
        if pos == dropoff and carrying and 'DROP' in valid_actions:
            return 'DROP'
        
        # Rule 3: Carrying, need to move toward dropoff
        if carrying and dropoff:
            if dropoff[0] < pos[0] and 'N' in valid_actions:
                return 'N'
            if dropoff[0] > pos[0] and 'S' in valid_actions:
                return 'S'
            if dropoff[1] < pos[1] and 'W' in valid_actions:
                return 'W'
            if dropoff[1] > pos[1] and 'E' in valid_actions:
                return 'E'
        
        # Rule 4: Not carrying, need to move toward pickup
        elif not carrying and pickup:
            if pickup[0] < pos[0] and 'N' in valid_actions:
                return 'N'
            if pickup[0] > pos[0] and 'S' in valid_actions:
                return 'S'
            if pickup[1] < pos[1] and 'W' in valid_actions:
                return 'W'
            if pickup[1] > pos[1] and 'E' in valid_actions:
                return 'E'
        
        # Fallback: random valid action
        return random.choice(valid_actions)

def main():
    """Run a simple demo of the reflex agent."""
    from warehouse_env import WarehouseEnv
    
    # Create environment and agent
    env = WarehouseEnv()
    agent = WarehouseAgentReflex()
    
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
