import time, os, sys, json, errno, argparse, pickle

import pandas as pd
import numpy as np
import networkx as nx

import analysis
from src import sim

sim_networks = {}

# Run simulation based on simulation runs dict.
# The runs dict has simulation ids as keys, and parameters dicts as values.
def run_sim(sim_params, include_payoffs=True, include_langs=False) :
    n_sim = len(sim_params)

    res_series = []     # For storing results of simulation runs
    for sim_id, params in sim_params.items() :
        # Create simulation object
        simulation = sim.Simulation(**params)

        # Run simulation and record runtime
        check_time = time.time()
        simulation.run()
        runtime = time.time() - check_time
        print(f"Ran simulation {int(sim_id)+1} (of {n_sim}) in {runtime/60:.2f} minutes.")

        # Save simulation run results and parameters
        sim_results = simulation.as_series(include_payoffs=include_payoffs, include_langs=include_langs).append(pd.Series({'runtime': runtime}))
        sim_results.name = sim_id
        res_series.append(sim_results)

        sim_networks[sim_id] = simulation.get_networks()

    # Create and return dataframe of simulation run results
    res_df = pd.DataFrame(res_series)
    return res_df, sim_networks

# If run as standalone module, read arguments, run sim, and record results
if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description="Run simulations and record their results.")
    parser.add_argument('param_filepath')
    parser.add_argument('results_output_path')
    parser.add_argument('-n', '--networks-filepath')
    parser.add_argument('--include-langs', action='store_true')

    args = parser.parse_args()

    # Get parameter filename
    param_filename = os.path.basename(args.param_filepath)
    
    # Get parameter file
    try :
        sim_params = json.load(open(args.param_filepath, mode='r'))
    except FileNotFoundError :
        print(f"Could not find the parmeter file '{args.param_filepath}'. Exiting...")
        sys.exit()

    # Run simulations and record runtime
    print("---  Starting simulation...  ---")
    start_time = time.time()
    check_time = start_time

    results_df, _ = run_sim(sim_params, include_langs=args.include_langs)

    print("---  Finished in %.2f minutes (real time)  ---" % ((time.time() - start_time) / 60))

    # Determine file and directory name
    base_filename = '_'.join(param_filename.split('_')[0:-1])

    # If given a filepath, extract filename and dirname
    if len(args.results_output_path.split('.')) > 1 :
        results_filename = os.path.basename(args.results_output_path)
        results_dirname = os.path.dirname(args.results_output_path)
    else :
        # If only given a dirname, generate default filename
        results_filename = base_filename + '_results.csv'   # Default results filename
        results_dirname = args.results_output_path
    
    # Export simulation results to results filepath
    results_filepath = os.path.join(results_dirname, results_filename)
    analysis.export_results(analysis.explode_results(results_df), results_filepath)

    # Pickle networks if provided argument
    if args.networks_filepath is not None :
        networks_dir = os.path.dirname(args.networks_filepath)

        # Create networks output directory if non-existent
        if not os.path.exists(networks_dir) :
            try :
                os.makedirs(networks_dir)
            except OSError as exc :
                if exc.errno != errno.EEXIST :
                    raise

        with open(os.path.join(args.networks_filepath), mode='wb') as networks_file :
            pickle.dump(sim_networks, networks_file)

        # for sim_id, run_networks in sim_networks.items() :
        #     sim_networks_dir = os.path.join(args.networks_dir, f'sim_{sim_id}_nwks')
        #     try :
        #         os.makedirs(sim_networks_dir)
        #     except OSError as exc :
        #         if exc.errno != errno.EEXIST :
        #             raise
            
        #     it = np.nditer(run_networks, flags=['refs_ok', 'c_index'])
        #     for G in it :
        #         nx.write_gpickle(G, os.path.join(sim_networks_dir, f'network_run_{it.index}.pickle'))