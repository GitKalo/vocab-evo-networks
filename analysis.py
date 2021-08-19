import networkx as nx
import pandas as pd
import numpy as np
import matplotlib

import sys, ast, os, errno

EXPORT_FILETYPES = ['csv', 'parq', 'parquet']
EXPORT_PATH_DEFAULT = './sim_results/results.parq'
PARQUET_OBJECT_ENCODINGS = {
    'vocab_size': 'json',
    'learning_strategy': 'json',
    'network_type': 'json',
    'network_update': 'json',
    'er_prob': 'float',
    'ba_links': 'int',
    'hk_prob': 'int',
    'ring_neighbors': 'int',
    'avg_payoffs': 'json',
    'node_payoffs': 'json',
}

# Calculate standard errors of mean payoffs
def std_err_estimator(avg_payoffs) :
    stds = np.std(avg_payoffs, axis=0)
    sqrt_sample_size = np.sqrt(len(avg_payoffs))
    std_errs = stds / sqrt_sample_size
    return std_errs

# Get payoff normalized by distance between initial and final
def norm_payoff(avg_payoffs, time_step) :
    return (avg_payoffs[time_step] - avg_payoffs[-1]) / (avg_payoffs[0] - avg_payoffs[-1])

# Get absolute normalized payoffs
def norm_payoff_abs(avg_payoffs, time_step) :
    return np.abs((avg_payoffs[time_step] - avg_payoffs[-1]) / (avg_payoffs[0] - avg_payoffs[-1]))

# Get relaxation time -- only applicable for monotonic payoff functions!
def t_relax(avg_payoffs) :
    return np.sum([norm_payoff(avg_payoffs, i) for i in range(len(avg_payoffs))])

# Get list of distances to final payoff based on normalzed payoffs
def get_distances(avg_payoffs) :
    return [norm_payoff_abs(avg_payoffs, i) for i in range(len(avg_payoffs))]

# Get convergence time based on pre-determined distance to final payoff (treshold parameter)
def get_t_conv_treshold(distances, treshold) :
    for i, d in enumerate(distances[::-1]) :
        if d > treshold :
            t_conv = len(distances) - (i - 1)
            break

    return t_conv

# Convergence time based on payoffs -- micro average of simulation runs
def t_conv(run_payoffs, treshold) :
    all_distances = [get_distances(p) for p in run_payoffs]
    all_t_conv = [get_t_conv_treshold(dist, treshold) for dist in all_distances]
    return np.mean(all_t_conv)

# Color nodes by total payoff
def get_node_payoffs(sims_df, i_sim=0, i_run=0, time_step=None) :
    ts = -1 if not time_step else time_step
    node_payoffs = sims_df.iloc[i_sim].node_payoffs[i_run][ts-1]
    
    return node_payoffs

# Get list of node colors based on distinct languages
def get_node_colors(agents, index_lang) :
    colors = list(matplotlib.colors.XKCD_COLORS.values())
    node_colors = []

    for a in agents :
        for i, l in enumerate(index_lang) :
            if np.all(a.active_matrix == l) :
                node_colors.append(colors[i])

    return node_colors

# Get index of distinct languages in population
def get_lang_index(agents) :
    index_lang = []
    dict_counts = {}

    for a in agents :
        if not np.any([np.all(a.active_matrix == l) for l in index_lang]) :
            index_lang.append(a.active_matrix)
            dict_counts[len(index_lang) - 1] = 1
        else :
            for i, l in enumerate(index_lang) :
                if np.all(a.active_matrix == l) :
                    dict_counts[i] += 1
    
    return index_lang, dict_counts

# Import simulation results
def import_results(results_filepath) :
    try :
        filetype = results_filepath.split('.')[-1]
        if filetype == 'csv' :
            imported_df = pd.read_csv(results_filepath, index_col=0, converters={
                'avg_payoffs': ast.literal_eval,
                'node_payoffs': ast.literal_eval
                })
        elif filetype in ['parquet', 'parq'] :
            imported_df = pd.read_parquet(results_filepath)
        else :
            raise BaseException(f"Unrecognized file type '{filetype}'.")
        
        return imported_df
    except FileNotFoundError :
        print(f"Could not find the results file '{results_filepath}'. Exiting...")
        raise
    except BaseException as e :
        print(e)
        print("Exiting...")
        sys.exit()

# Import all networks in folder
def import_networks(networks_folder, results_df) :
    networks = []

    for sim in range(len(results_df)) :
        sim_nwks = []
        for run in range(results_df.iloc[sim].runs) :
            G = nx.read_gpickle(os.path.join(os.path.abspath(__file__), f'{networks_folder}/sim_{sim+1}_nwks/network_run_{run}.pickle')).item()
            sim_nwks.append(G)
        
        networks.append(sim_nwks)

    return networks

# Export simulation results
def export_results(results_df, results_filepath=None) :
    if not results_filepath :
        results_filepath = EXPORT_PATH_DEFAULT

    # Extract file and directory name
    if len(results_filepath.lstrip('.').split('.')) > 1 :
        results_dirname = os.path.dirname(results_filepath)
        results_filename = os.path.basename(results_filepath)
        file_extension = results_filepath.split('.')[-1]
    else :
        print("Please provide a valid filepath, including a file extension.")
        print("Exiting...")
        sys.exit()

    # Create results output directory if non-existent
    if not os.path.exists(results_dirname) :
        try :
            os.makedirs(results_dirname)
        except OSError as exc :
            if exc.errno != errno.EEXIST :
                raise

    if file_extension not in EXPORT_FILETYPES :
        print(f"Unrecognized or unsupported file extension '{file_extension}'.")
        print("Exiting...")
        sys.exit()
    elif file_extension == 'csv' :
        results_df.to_csv(results_filepath)
    elif file_extension in ['parq', 'parquet'] :
        results_df.to_parquet(results_filepath, engine='fastparquet', object_encoding=PARQUET_OBJECT_ENCODINGS, row_group_offsets=50)

    print(f"Saved to {os.path.abspath(results_filepath)}.")

# Combine results from multiple files into single dataframe
def combine_results(res_files) :
    dfs = []
    for f in res_files :
        dfs.append(import_results(f))
    results_df = pd.concat(dfs)
    results_df.reset_index(drop=True, inplace=True)
    return results_df

# Index a results df by sim ID and run ID by exploding on payoff columns
def reindex_df(results_df) :
    new_df = results_df.explode(['avg_payoffs', 'node_payoffs'])
    new_df['sim'] = new_df.index
    new_df = new_df.reset_index(drop=True)
    new_df['run'] = new_df.index
    new_df = new_df.set_index(['sim', 'run'])
    n_runs = new_df.runs.iloc[0]
    new_df = new_df.rename(lambda x : x%n_runs, axis=0, level=1)

    return new_df