import time, os, sys, json, errno

import pandas as pd
import numpy as np
import networkx as nx

import analysis
from src import sim

__DEFAULT_RESULTS_DIR = './sim_results/'    # Default directory for writing CSV of sim results

sim_networks = {}
sim_node_langs = {}

# Run simulation based on simulation runs dict.
# The runs dict has simulation ids as keys, and parameters dicts as values.
def run_sim(sim_params) :
    n_sim = len(sim_params)

    res_series = []     # For storing results of simulation runs
    for id, params in sim_params.items() :
        # Create simulation object
        simulation = sim.Simulation(**params)

        # Run simulation and record runtime
        check_time = time.time()
        simulation.run()
        runtime = time.time() - check_time
        print("Ran simulation %s (of %i) in %.2f minutes." % (id, n_sim, runtime/60))

        # Save simulation run results and parameters
        sim_results = simulation.as_series(include_payoffs=True).append(pd.Series({'runtime': runtime}))
        sim_results.name = id
        res_series.append(sim_results)

        sim_networks[id] = simulation.get_networks()
        sim_node_langs[id] = simulation.get_node_langs()

    # Create and return dataframe of simulation run results
    res_df = pd.DataFrame(res_series)
    return res_df, sim_networks, sim_node_langs

# If run as standalone module, take simulation runs parameter file as input, run 
# simulations, and output CSV file of results (which can also be given as input).
# -- 1st argument: path to simulation paramter file
# -- 2nd argument (optional): path to file or directory for writing results (by default
#  outputted to './sim_results/')
if __name__ == '__main__' :
    # Get parameter filename
    try :
        param_filepath = sys.argv[1]
        param_filename = os.path.basename(param_filepath)
    except IndexError :
        print("No parameter file specified. Exiting...")
        sys.exit()

    # Get parameter file
    try :
        sim_params = json.load(open(param_filepath, mode='r'))
    except FileNotFoundError :
        print(f"Could not find the parmeter file '{param_filename}'. Exiting...")
        sys.exit()

    # Run simulations and record runtime
    print("---  Starting simulation...  ---")
    start_time = time.time()
    check_time = start_time

    results_df, _, _ = run_sim(sim_params)

    print("---  Finished in %.2f minutes (real time)  ---" % ((time.time() - start_time) / 60))

    # Get results output filename and directory
    base_filename = '_'.join(param_filename.split('_')[0:-1])
    try :
        # If given a results path argument, extract file and directory name
        input_results_path = sys.argv[2]
        if len(input_results_path.split('.')) > 1 :
            results_filename = os.path.basename(input_results_path)
            results_dirname = os.path.dirname(input_results_path)
        else :
            results_filename = base_filename + '_results.csv'   # Default results filename
            results_dirname = input_results_path
    except IndexError :
        # If not given a results path argument, use default results output
        results_filename = base_filename + '_results.csv'
        results_dirname = __DEFAULT_RESULTS_DIR
    
    # Write CSV results to results file
    results_filepath = os.path.join(results_dirname, results_filename)

    # Export simulation results
    analysis.export_results(analysis.explode_results(results_df), results_filepath)

    # Pickle networks if provided argument
    try :
        # Check for correct argument switch ('-n')
        switch = sys.argv[3]
        if not switch == '-n' :
            print(f"Unrecognized switch '{switch}'. Exiting...")
            sys.exit()

        input_networks_dir = sys.argv[4]
        # if len(input_networks_dir.split('.')) > 1 :
        #     print("Invalid directory for network output (not a directory). Exiting...")
        #     sys.exit()

        # Create networks output directory if non-existent
        if not os.path.exists(input_networks_dir) :
            try :
                os.makedirs(input_networks_dir)
            except OSError as exc :
                if exc.errno != errno.EEXIST :
                    raise

        for sim_id, run_networks in sim_networks.items() :
            sim_networks_dir = os.path.join(input_networks_dir, f'sim_{sim_id}_nwks')
            try :
                os.makedirs(sim_networks_dir)
            except OSError as exc :
                if exc.errno != errno.EEXIST :
                    raise
            
            it = np.nditer(run_networks, flags=['refs_ok', 'c_index'])
            for G in it :
                nx.write_gpickle(G, os.path.join(sim_networks_dir, f'network_run_{it.index}.pickle'))
    except IndexError :
        pass
