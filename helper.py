import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
    
# --- Minimize the plot window ---
    try:
        # This works for TkAgg backend (default on many systems)
        manager = plt.get_current_fig_manager()
        if hasattr(manager.window, 'iconify'):
            manager.window.iconify()
    except Exception as e:
        pass  # Ignore if backend does not support minimizing