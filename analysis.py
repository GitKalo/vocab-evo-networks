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
    'hk_prob': 'float',
    'ring_neighbors': 'int',
    'avg_payoffs': 'json',
    'node_payoffs': 'json',
}

# Calculate standard errors of mean payoffs
def std_err(samples) :
    std = np.std(samples)
    sqrt_sample_size = np.sqrt(len(samples))
    std_errs = std / sqrt_sample_size
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

# Get convergence time based on pre-determined distance to final payoff (threshold parameter)
def get_t_conv_threshold(distances, threshold) :
    for i, d in enumerate(distances[::-1]) :
        if d > threshold :
            t_conv = len(distances) - (i - 1)
            break

    return t_conv

def get_t_conv_runs(run_payoffs, threshold) :
    all_distances = [get_distances(p) for p in run_payoffs]
    all_t_conv = [get_t_conv_threshold(dist, threshold) for dist in all_distances]
    return all_t_conv

# Convergence time based on payoffs -- micro average of simulation runs
def t_conv(run_payoffs, threshold) :
    return np.mean(get_t_conv_runs(run_payoffs, threshold))

# Color nodes by total payoff
def get_node_payoffs(sims_df, i_sim=0, i_run=0, time_step=None) :
    ts = 0 if not time_step else time_step
    node_payoffs = sims_df.iloc[i_sim].node_payoffs[i_run][ts-1]
    
    return node_payoffs

# Get list of node colors based on distinct languages
def get_node_colors_list(langs, index_lang, color_list=None) :
    if color_list is None :
        color_list = list(matplotlib.colors.XKCD_COLORS.values())
    elif len(index_lang) > len(color_list) :
        raise ValueError("Number of provided colors must be at least equal to the number of different languages.")
        
    node_colors = []

    for lang in langs :
        for i, l in enumerate(index_lang) :
            if np.all(lang == l) :
                node_colors.append(color_list[i])

    return node_colors

# Get list of node colors based on distinct languages
def get_node_colors_pop(agents, index_lang, color_list=None) :
    return get_node_colors_list([a.active_matrix for a in agents], index_lang, color_list)

def get_node_colors_seq(list_lang_reports, color_list=None) :
    node_color_seq = []

    for list_langs in list_lang_reports :
        index_lang, dict_counts = get_lang_index_list(list_langs)
        sorted_langs = sorted(enumerate(index_lang), key=lambda l : dict_counts[l[0]], reverse=True)
        sorted_langs = list(list(zip(*sorted_langs))[1])
        node_colors = get_node_colors_list(list_langs, sorted_langs, color_list)
        node_color_seq.append(node_colors)

    return node_color_seq

# Get index of distinct languages in population
def get_lang_index_pop(agents) :
    return get_lang_index_list([a.active_matrix for a in agents])

def get_lang_index_list(langs) :
    index_lang = []
    dict_counts = {}

    for lang in langs :
        if not np.any([np.all(lang == l) for l in index_lang]) :
            index_lang.append(lang)
            dict_counts[len(index_lang) - 1] = 1
        else :
            for i, l in enumerate(index_lang) :
                if np.all(lang == l) :
                    dict_counts[i] += 1
    
    return index_lang, dict_counts

# Import simulation results
def import_results(results_filepath) :
    try :
        filetype = results_filepath.split('.')[-1]
        if filetype == 'csv' :
            imported_df = pd.read_csv(results_filepath, index_col=[0, 1], converters={
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

# Combine results from multiple files into single dataframe
def import_results_multiple(list_results_filepaths) :
    dfs = []
    for f in list_results_filepaths :
        dfs.append(implode_results(import_results(f)))
    results_df = pd.concat(dfs)
    results_df.reset_index(drop=True, inplace=True)
    return results_df

# Import all networks in folder
def import_networks(networks_folder, results_df) :
    networks = []

    for sim in range(len(results_df)) :
        sim_nwks = []
        for run in range(results_df.iloc[sim].n_runs) :
            G = nx.read_gpickle(f'{networks_folder}/sim_{sim+1}_nwks/network_run_{run}.pickle').item()
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

    # Ensure results are in exploded (single run per row) format
    if not isinstance(results_df.index, pd.MultiIndex) :
        results_df = explode_results(results_df)

    if file_extension not in EXPORT_FILETYPES :
        print(f"Unrecognized or unsupported file extension '{file_extension}'.")
        print("Exiting...")
        sys.exit()
    elif file_extension == 'csv' :
        results_df.to_csv(results_filepath)
    elif file_extension in ['parq', 'parquet'] :
        try :
                results_df.to_parquet(results_filepath, engine='fastparquet', object_encoding=PARQUET_OBJECT_ENCODINGS, row_group_offsets=10)
        except TypeError :  
                results_df.to_parquet(results_filepath, engine='fastparquet', object_encoding='json', row_group_offsets=10)
        except OverflowError as e :
                print(e)
                results_df.to_parquet(results_filepath, engine='fastparquet', object_encoding='json', row_group_offsets=2)
                # print("Encountered overflow error in writing to parquet, saving to csv instead.")
                # results_df.to_csv(results_filepath)

    print(f"Saved to {os.path.abspath(results_filepath)}.")

# Index a results df by sim ID and run ID by exploding on payoff columns
# Function is *not* idempotent, it will likely throw an error if attempting to apply
# multiple times to the same dataframe
def explode_results(results_df) :
    exploded_df = results_df.explode(['avg_payoffs', 'node_payoffs'])
    exploded_df['sim'] = exploded_df.index
    exploded_df = exploded_df.reset_index(drop=True)
    exploded_df['run'] = exploded_df.index
    exploded_df = exploded_df.set_index(['sim', 'run'])
    n_runs = exploded_df.n_runs.iloc[0]
    exploded_df = exploded_df.rename(lambda x : x%n_runs, axis=0, level=1)

    return exploded_df

# Index an exploded results df by sim ID, aggregating payoffs into list
# Function is idempotent, it can safely be applied multiple times to the same dataset
def implode_results(exploded_df) :
    agg_df = pd.DataFrame()
    exploded_df.index.name = 'sim'
    
    for name, data in exploded_df.iteritems() :
        if name not in ['avg_payoffs', 'node_payoffs'] :
            agg_df[name] = data.groupby('sim').first()
        else :
            agg_df[name] = data.groupby('sim').aggregate(lambda s : s.iloc[0] if len(s) == 1 else s.tolist())

    return agg_df
