import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation

import ast, sys
import analysis

# Plot payoffs over time for a set of runs. Optinally plot the mean payoffs
# over all runs as well (via the `mean` argument)
def plot_sim_payoffs(ax, runs, time_step_lim, mean=True, mean_text=True, v=None) :
    for run in runs :
        ax.plot(run, color='blue', alpha=0.5, ls='-')
    
    if mean :
        mean_payoffs = np.mean(runs, axis=0)
        ax.plot(mean_payoffs, color='orange')

        if mean_text :
            display_text = f'Mean payoff @ {time_step_lim} = {round(mean_payoffs[time_step_lim - 1], 2)}'
            ax.text(0.95, 0.05, display_text, ha='right', transform=ax.transAxes)

    ax.set_xlabel('Time steps (t)')
    ax.set_ylabel('Mean payoff')
    ax.set_xlim(left=0, right=time_step_lim)
    if v : ax.set_ylim(bottom=-1, top=v+1)

    return ax

# Plot payoffs over time for a set of runs for multiple simulation instances. 
# All runs and (optionally) their mean is plotted.
def plot_sim_payoffs_all(sims_df, time_step_lim=None, normalize=False, n_cols=None, n_rows=None, figsize=(10, 6), **kwargs) :
    # Reformat results if provided as exploded df
    sims_df = analysis.implode_results(sims_df)

    # Automatically calculate number of columns and rows for subplots if not specified
    if not n_cols :
        n_cols = int(np.ceil(np.sqrt(len(sims_df))))
    if not n_rows :
        n_rows = len(sims_df)//n_cols + int(np.ceil((len(sims_df)%n_cols)/n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, sim in sims_df.iterrows() :
        i = int(i)  # Ensure index is in int format (not str)

        # Determine which axis to plot on next
        try :
            ax_i, ax_j = i//n_cols, i%n_cols
            ax = axs[ax_i][ax_j]
        except TypeError :
            try :
                ax = axs[i]
            except TypeError :
                ax = axs

        # Plot mean payoffs for simulation runs
        if normalize :
            try :
                v = ast.literal_eval(sim.vocab_size)[0]
            except AttributeError :     # If sim has no 'vocab_size'
                if type(normalize) is not bool :
                    v = normalize
        else :
            v = None
        time_steps = sim.payoff_reports_n if not time_step_lim else time_step_lim
        plot_sim_payoffs(ax, sim.avg_payoffs, time_step_lim=time_steps, v=v, **kwargs)
        ax.set_title(f'Sim id: {i}')
    
    return fig, axs

# Plot payoffs for all nodes for a single run. The `type` of the plot can be one of:
# - 'line' -- plots the payoffs for all nodes over all time steps
# - 'hist' -- plots the payoff distribution for nodes at a single time step, based on
# the `time_step` argument (defaults to the last time step)
def plot_node_payoffs(ax, runs, i_run=0, type='line', time_step=0, **kwargs) :
    if type == 'line' :
        ax.plot(runs[i_run], **kwargs)
        ax.set_title(f"Node payoffs for run {i_run + 1}")
    elif type == 'hist' :
        if not time_step : time_step = len(runs[i_run])
        ax.hist(runs[i_run][time_step-1], bins=50, **kwargs)
        ax.set_title(f"Payoff distribution at time step {time_step} for run {i_run + 1}")
    else :
        print(f"Unrecognized type {type}, must be either 'line' or 'hist'.")
        sys.exit()

    return ax

# Update function for animation
def update_animation(num, G, node_colors, pos, ax, step_size=1, title=True, draw_params={}):
    ax.clear()

    color_idx = num * step_size

    # Draw network colored by node payoffs 
    nx.draw(G, pos=pos, node_color=node_colors[color_idx], ax=ax, **draw_params)

    # Set the title
    if title :
        ax.set_title(f"Step {color_idx}", pad=30)

# Run animation
def run_animation(G, node_colors, step_lim, step_size=1, pos=None, cbar=False, title=True, interval=200, gif_filename='ani.gif', **draw_params):
    n_frames = step_lim // step_size
    
    if step_lim > len(node_colors) :
        raise ValueError("Cannot animate more steps than provided in node_colors.")

    # Build plot
    fig, ax = plt.subplots(figsize=(10,6))
    if not pos : pos = nx.spring_layout(G, iterations=1000)

    ani = animation.FuncAnimation(fig, update_animation, frames=n_frames, fargs=(G, node_colors, pos, ax, step_size, title, draw_params), interval=interval, repeat_delay=500, save_count=100)

    if cbar :
        vmin = draw_params['vmin']
        vmax = draw_params['vmax']
        cmap = draw_params['cmap']

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cb = plt.colorbar(sm, ticks=np.linspace(vmin, vmax, 5))
        cb.ax.tick_params(labelsize=16, length=10)

    ani.save(gif_filename, writer='pillow')

    plt.show()

if __name__ == '__main__' :
    # Get results filename
    try :
        results_filepath = sys.argv[1]
    except IndexError :
        print("No results file specified. Exiting...")
        sys.exit()

    # Import results from file
    try :
        results_df = analysis.import_results(results_filepath)
    except FileNotFoundError :
        sys.exit()

    # Plot all runs in results dataframe
    fig, axs = plot_sim_payoffs_all(results_df)

    # Display simulation plot
    plt.show()

    # If a second argument is provided, save figure to given file
    try :
        output_plot_path = sys.argv[2]
        plt.savefig(output_plot_path)
    except IndexError :
        pass
    except :
        print(f"Could not save figure to '{output_plot_path}'.")