import matplotlib.pyplot as plt

def generate_sliding_windows():
    return [
        {'type': 'window_1', 'train': (0.0, 0.12), 'validate': (0.12, 0.16), 'test': (0.16, 0.2)},
        {'type': 'window_2', 'train': (0.16, 0.28), 'validate': (0.28, 0.32), 'test': (0.32, 0.36)},
        {'type': 'window_3', 'train': (0.32, 0.44), 'validate': (0.44, 0.48), 'test': (0.48, 0.52)},
        {'type': 'window_4', 'train': (0.48, 0.60), 'validate': (0.60, 0.64), 'test': (0.64, 0.68)},
        {'type': 'window_5', 'train': (0.64, 0.76), 'validate': (0.76, 0.80), 'test': (0.8, 0.84)},
        {'type': 'window_6', 'train': (0.8, 0.92), 'validate': (0.92, 0.96), 'test': (0.96, 1.0)}
    ]

def plot_sliding_windows(windows):
    # Define colors for each segment
    colors = {"train": "tab:blue", "validate": "tab:orange", "test": "tab:green"}

    fig, ax = plt.subplots(figsize=(10, 6))

    # Loop over each window and draw its segments as broken horizontal bars
    for i, window in enumerate(windows):
        # Position each window on a separate row
        row_position = i
        for segment in ['train', 'validate', 'test']:
            start, end = window[segment]
            duration = end - start
            # Label only for the first row to avoid duplicate legend entries
            label = f"{segment} ({duration * 100:.0f}%)" if i == 0 else ""
            ax.broken_barh([(start, duration)], (row_position - 0.4, 0.8),
                           facecolors=colors[segment], edgecolor='black', label=label)

    # Set labels and ticks
    ax.set_xlabel("Timeline (%)")
    ax.set_ylabel("Sliding Windows")
    ax.set_yticks(range(len(windows)))
    ax.set_yticklabels([w["type"] for w in windows])
    ax.set_xlim(0, 1)
    ax.set_title("Sliding Windows Visual Representation")

    # Create a legend (avoid duplicate labels)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.savefig("sliding_windows.png")
    plt.show()

# Generate sliding windows and plot them
windows = generate_sliding_windows()
plot_sliding_windows(windows)
