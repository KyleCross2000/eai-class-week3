import numpy as np
import matplotlib.pyplot as plt
from warehouse_agent_reflex import WarehouseAgentReflex
from warehouse_agent_greedy import GreedyAgent
from unified_episode_runner import run_n_episodes, print_comparison_summary

NUM_EPISODES = 50

def convert_to_legacy_format(unified_results):
    """Convert unified_episode_runner results to legacy stats format for plotting."""
    episodes = unified_results['episodes']
    return {
        'success': [ep['success'] for ep in episodes],
        'steps': [ep['steps'] for ep in episodes],
        'battery': [ep['battery_remaining'] for ep in episodes],
        'reward': [ep['total_reward'] for ep in episodes],
    }

def analyze_stats(stats):
    arr = lambda k: np.array(stats[k])
    success_rate = arr('success').mean()
    successful_steps = arr('steps')[arr('success') == 1]
    mean_steps = successful_steps.mean() if len(successful_steps) > 0 else float('nan')
    median_steps = np.median(successful_steps) if len(successful_steps) > 0 else float('nan')
    return {
        'success_rate': success_rate,
        'mean_steps': mean_steps,
        'median_steps': median_steps,
    }

def plot_comparison(stats_dict):
    labels = list(stats_dict.keys())
    n = len(labels)
    # Success rate
    success_rates = [analyze_stats(stats_dict[l])['success_rate'] for l in labels]
    # Box plots for episode lengths (successful only)
    steps_data = [np.array(stats_dict[l]['steps'])[np.array(stats_dict[l]['success']) == 1] for l in labels]
    # Histograms for battery
    battery_data = [stats_dict[l]['battery'] for l in labels]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    # Bar chart: Success rate
    axs[0].bar(labels, [100 * s for s in success_rates], color=['#4e79a7', '#f28e2b'])
    axs[0].set_ylabel('Success Rate (%)')
    axs[0].set_title('Success Rate')
    for i, v in enumerate(success_rates):
        axs[0].text(i, 100 * v + 2, f"{100*v:.1f}%", ha='center')
    # Box plot: Episode length
    axs[1].boxplot(steps_data, tick_labels=labels)
    axs[1].set_ylabel('Episode Length (steps)')
    axs[1].set_title('Episode Lengths (Successful)')
    # Histogram: Final battery
    axs[2].hist(battery_data, bins=10, label=labels, alpha=0.7)
    axs[2].set_xlabel('Final Battery')
    axs[2].set_ylabel('Count')
    axs[2].set_title('Final Battery Distribution')
    axs[2].legend()
    plt.tight_layout()
    plt.show()

print("Running Reflex Agent...")
reflex_results = run_n_episodes(WarehouseAgentReflex, NUM_EPISODES, progress=True)
print("\nRunning Greedy Agent...")
greedy_results = run_n_episodes(GreedyAgent, NUM_EPISODES, progress=True)

# Convert to legacy format for plotting
reflex_stats = convert_to_legacy_format(reflex_results)
greedy_stats = convert_to_legacy_format(greedy_results)
stats_dict = {'Reflex': reflex_stats, 'Greedy': greedy_stats}

# Print unified comparison summary
print_comparison_summary({'Reflex': reflex_results, 'Greedy': greedy_results})

# Print legacy-style analysis
for label, stats in stats_dict.items():
    analysis = analyze_stats(stats)
    print(f"{label} Agent:")
    print(f"  Success Rate: {analysis['success_rate']*100:.1f}%")
    print(f"  Mean Steps (successful): {analysis['mean_steps']:.1f}")
    print(f"  Median Steps (successful): {analysis['median_steps']:.1f}")

# Display visualization
plot_comparison(stats_dict)