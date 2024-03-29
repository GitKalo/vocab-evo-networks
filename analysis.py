import networkx as nx
import pandas as pd
import numpy as np
import matplotlib

import sys, ast, os, errno, pickle

RESULTS_FILETYPES = ['csv', 'parq', 'parquet']
RESULTS_PATH_DEFAULT = './sim_results/results.parq'
NETWORKS_PATH_DEFAULT = './sim_networks/networks.pickle'
PARQUET_OBJECT_ENCODINGS = {
    'sample_strategy': 'json',
    'nwk_topology': 'json',
    'nwk_update': 'json',
    'nwk_random_p': 'float',
    'nwk_sf_links': 'int',
    'nwk_clustered_p': 'float',
    'nwk_ring_neighbors': 'int',
    'avg_payoffs': 'json',
    'node_payoffs': 'json',
    'node_langs': 'json',

    # reports
    'rewires': 'json',
    'max_degree': 'json',
    'avg_shortest_path': 'json',
    'avg_clustering': 'json',
    'transitivity': 'json',

    # unused params, left for compatibility
    'vocab_size': 'json',
    'learning_strategy': 'json',
    'network_type': 'json',
    'network_update': 'json',
    'er_prob': 'float',
    'ba_links': 'int',
    'hk_prob': 'float',
    'ring_neighbors': 'int'
}

EXPLODE_COLUMNS = ['avg_payoffs', 'node_payoffs', 'node_langs', \
    'rewires', 'max_degree', 'avg_shortest_path', 'avg_clustering', 'transitivity']

# Calculate standard errors of mean payoffs
def std_err(samples) :
    std = np.std(samples)
    sqrt_sample_size = np.sqrt(len(samples))
    std_errs = std / sqrt_sample_size
    return std_errs

# Get payoff normalized by distance between initial and final
def norm_payoff(avg_payoffs, start, final, absolute=True) :
    norm = (avg_payoffs - final) / (start - final)

    if absolute : norm = np.abs(norm)
    
    return norm

# Get relaxation time -- only applicable for monotonic payoff functions!
def t_relax(avg_payoffs) :
    return np.sum([norm_payoff(avg_payoffs, i) for i in range(len(avg_payoffs))])

# Get list of distances to final payoff based on normalzed payoffs
def get_distances(avg_payoffs) :
    if not isinstance(avg_payoffs, np.ndarray) : avg_payoffs = np.array(avg_payoffs)
    assert avg_payoffs.ndim == 1, "Function only works for 1D arrays"
    return norm_payoff(avg_payoffs, avg_payoffs[0], avg_payoffs[-1], absolute=True)

# Get convergence time based on pre-determined distance to final payoff (threshold parameter)
def get_t_conv_single(distances, threshold) :
    t_conv = len(distances) - 1
    for i, d in enumerate(distances[::-1]) :
        if d > threshold :
            t_conv = len(distances) - (i - 1)
            break
    else :
        t_conv = 1

    return t_conv

def get_t_conv_runs(sim_payoffs, threshold) :
    all_distances = [get_distances(p) for p in sim_payoffs]
    all_t_conv = [get_t_conv_single(dist, threshold) for dist in all_distances]
    return all_t_conv

# Convergence time based on payoffs -- micro average of simulation runs
def t_conv(sim_payoffs, threshold) :
    return np.mean(get_t_conv_runs(sim_payoffs, threshold))

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

def get_lang_dist(run_langs) :
    langs_list_init, langs_counts_init = get_lang_index_list(run_langs[0])
    langs_counts_sorted_iter = iter(sorted_dict(langs_counts_init).items())

    a_index, a_counts = next(langs_counts_sorted_iter)
    b_index, b_counts = next(langs_counts_sorted_iter)
    langs_ab = (langs_list_init[a_index], langs_list_init[b_index])

    dists = [{}] * len(run_langs)

    for i, t in enumerate(run_langs) :
        langs_list, langs_counts = get_lang_index_list(t)
        langs_counts_sorted = sorted_dict(langs_counts)
        
        langs_dist = {idx: [cnt, langs_list[idx]] for idx, cnt in langs_counts_sorted.items()}

        dists[i] = langs_dist

    return langs_ab, dists

def get_lang_dist_dual(run_langs) :
    
    langs_list_init, langs_counts_init = get_lang_index_list(run_langs[0])
    langs_counts_sorted_iter = iter(sorted_dict(langs_counts_init).items())

    a_index, a_counts = next(langs_counts_sorted_iter)
    b_index, b_counts = next(langs_counts_sorted_iter)
    a_lang = langs_list_init[a_index]
    b_lang = langs_list_init[b_index]
    # print(np.array(a_lang))
    # print(np.array(b_lang))

    dists = [()] * len(run_langs)

    for i, t in enumerate(run_langs) :
        langs_list, langs_counts = get_lang_index_list(t)

        try :
            a_counts = langs_counts[langs_list.index(a_lang)]
            del langs_counts[langs_list.index(a_lang)]
        except ValueError : # Lang has gone extinct
            a_counts = 0

        try :
            b_counts = langs_counts[langs_list.index(b_lang)]
            del langs_counts[langs_list.index(b_lang)]
        except ValueError :
            b_counts = 0

        other_counts = 0
        for j in langs_counts :
            other_counts += langs_counts[j]

        dist = (a_counts, b_counts, other_counts)
        dists[i] = dist

    return dists

def sorted_dict(d, sort_value_index=None, descending=True) :
    if sort_value_index is None :
        return {k:v for k,v in sorted(d.items(), key=lambda item: item[1], reverse=descending)}
    else :
        return {k:v for k,v in sorted(d.items(), key=lambda item: item[1][sort_value_index], reverse=descending)}

def hamming_dist(a, b) :
    if not isinstance(a, np.ndarray) :
        a = np.array(a)
    if not isinstance(b, np.ndarray) :
        b = np.array(b)

    return np.count_nonzero(a != b)

# Import simulation results
def import_results(results_filepath) :
    try :
        filetype = results_filepath.split('.')[-1]
        if filetype == 'csv' :
            # TODO: add handling of other string columns
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
def import_networks_dir(networks_folder, results_df) :
    networks = []

    for sim in range(results_df.index.get_level_values(0).nunique()) :
        sim_nwks = []
        for run in range(results_df.iloc[sim].n_runs) :
            G = nx.read_gpickle(f'{networks_folder}/sim_{sim}_nwks/network_run_{run}.pickle').item()
            sim_nwks.append(G)
        
        networks.append(sim_nwks)

    return networks

def import_networks_file(networks_filepath) :
    with open(networks_filepath, mode='rb') as networks_file :
        return pickle.load(networks_file)

# Export simulation results
def export_results(results_df, results_filepath=None) :
    if not results_filepath :
        results_filepath =  RESULTS_PATH_DEFAULT

    # Extract file and directory name
    if len(results_filepath.lstrip('.').split('.')) > 1 :
        results_dirname = os.path.dirname(results_filepath)
        results_filename = os.path.basename(results_filepath)
        file_extension = results_filepath.split('.')[-1]
    else :
        raise ValueError(f"Invalid filepath '{results_filepath}', must include a file extension.")

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

    if file_extension not in RESULTS_FILETYPES :
        raise ValueError(f"Unrecognized or unsupported file extension '{file_extension}'.")
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

    print(f"Saved results to '{os.path.abspath(results_filepath)}'.")

def export_networks_file(networks_dict, networks_filepath=None) :
    if not networks_filepath :
        networks_filepath = NETWORKS_PATH_DEFAULT
    
    # Extract file and directory name
    if len(networks_filepath.lstrip('.').split('.')) > 1 :
        networks_dirname = os.path.dirname(networks_filepath)
        networks_filename = os.path.basename(networks_filepath)
        file_extension = networks_filepath.split('.')[-1]
    else :
        raise ValueError(f"Invalid filepath '{networks_filepath}', must include a file extension.")

    # Create output directory if non-existent
    if not os.path.exists(networks_dirname) :
        try :
            os.makedirs(networks_dirname)
        except OSError as exc :
            if exc.errno != errno.EEXIST :
                raise
    
    if file_extension != 'pickle' :
        raise ValueError(f"Unrecognized or unsupported file extension '{file_extension}'. Can only save to '.pickle' files.")

    with open(os.path.join(networks_filepath), mode='wb') as networks_file :
        pickle.dump(networks_dict, networks_file)

    print(f"Saved networks to '{os.path.abspath(networks_filepath)}'.")

# Index a results df by sim ID and run ID by exploding on payoff columns
# Function is *not* idempotent, it will likely throw an error if attempting to apply
# multiple times to the same dataframe
def explode_results(results_df) :
    exploded_df = results_df.explode([col for col in EXPLODE_COLUMNS if col in results_df.columns])
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
        if name not in EXPLODE_COLUMNS :
            agg_df[name] = data.groupby('sim').first()
        else :
            agg_df[name] = data.groupby('sim').aggregate(lambda s : s.iloc[0] if len(s) == 1 else s.tolist())

    return agg_df
