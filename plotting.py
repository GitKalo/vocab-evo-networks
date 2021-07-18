from datetime import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ast, os, sys

# Plot payoffs over time for a set of runs. Optinally plot the mean payoffs
# over all runs as well (via the `mean` argument)
def plot_run_payoffs(ax, runs, time_step_lim, mean=True, v=None) :
    for run in runs :
        ax.plot(run, color='blue', alpha=0.5)
    mean_payoffs = np.mean(runs, axis=0)
    if mean : ax.plot(mean_payoffs, color='orange')

    display_text = f'Mean payoff @ {time_step_lim} = {round(mean_payoffs[time_step_lim - 1], 2)}'
    ax.text(0.95, 0.05, display_text, ha='right', transform=ax.transAxes)

    ax.set_xlabel('Time steps (t)')
    ax.set_ylabel('Mean payoff')
    ax.set_xlim(left=0, right=time_step_lim)
    if v : ax.set_ylim(bottom=-1, top=v+1)

    return ax

# Plot payoffs over time for a set of runs for multiple simulation instances. 
# All runs and (optionally) their mean is plotted.
def plot_run_payoffs_all(sims_df, mean=True, normalize=False, n_cols=None, n_rows=None, figsize=(10, 6)) :
    # Automatically calculate number of columns and rows for subplots if not specified
    if not n_cols :
        n_cols = int(np.ceil(np.sqrt(len(sims_df))))
    if not n_rows :
        n_rows = len(sims_df)//n_cols + int(np.ceil((len(sims_df)%n_cols)/n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, sim in sims_df.iterrows() :
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
        v = None if not normalize else ast.literal_eval(sim.vocab_size)[0]
        plot_run_payoffs(ax, sim.avg_payoffs, time_step_lim=sim.time_steps, mean=mean, v=v)
        ax.set_title(f'Sim id: {i}')
    
    return fig, axs

# Plot payoffs for all nodes for a single run. The `type` of the plot can be one of:
# - 'line' -- plots the payoffs for all nodes over all time steps
# - 'hist' -- plots the payoff distribution for nodes at a single time step, based on
# the `time_step` argument (defaults to the last time step)
def plot_node_payoffs(ax, runs, i_run=0, type='line', time_step=0) :
    if type == 'line' :
        ax.plot(runs[i_run])
        ax.set_title(f"Node payoffs for run {i_run + 1}")
    elif type == 'hist' :
        if not time_step : time_step = len(runs[i_run])
        ax.hist(runs[i_run][time_step-1], bins=50)
        ax.set_title(f"Payoff distribution for nodes at time step {time_step} for run {i_run + 1}")
    else :
        print(f"Unrecognized type {type}, must be either 'line' or 'hist'.")
        sys.exit()

    return ax

# Import CSV of simulation results
def import_results(results_filepath) :
    try :
        imported_df = pd.read_csv(results_filepath, index_col=0, converters={
            'avg_payoffs': ast.literal_eval,
            'node_payoffs': ast.literal_eval
            })
        return imported_df
    except Exception :
        print(f"Could not find the results file '{results_filepath}'. Exiting...")
        raise

if __name__ == '__main__' :
    # Get results filename
    try :
        results_filepath = sys.argv[1]
    except IndexError :
        print("No results file specified. Exiting...")
        sys.exit()

    # Import results from file
    try :
        results_df = import_results(results_filepath)
    except FileNotFoundError :
        sys.exit()

    # Plot all runs in results dataframe
    fig, axs = plot_run_payoffs_all(results_df)

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