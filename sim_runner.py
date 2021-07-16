import time, os, sys, json, errno

import pandas as pd

from src import sim

__DEFAULT_RESULTS_DIR = './sim_results/'    # Default directory for writing CSV of sim results

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
        sim_results = simulation.as_series().append(pd.Series({'runtime': runtime}))
        sim_results.name = id
        res_series.append(sim_results)

    # Create and return dataframe of simulation run results
    res_df = pd.DataFrame(res_series)
    return res_df

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

    results_df = run_sim(sim_params)

    print("---  Finished in %.2f minutes (real time)  ---" % ((time.time() - start_time) / 60))

    # Get results output filename and directory
    base_filename = '_'.join(param_filename.split('_')[0:-1])
    try :
        # If given a results path argument, extract file and directory name
        input_results_path = sys.argv[2]
        if os.path.isfile(input_results_path) :
            results_filename = os.path.basename(input_results_path)
            results_dirname = os.path.dirname(input_results_path)
        else :
            results_filename = base_filename + '_results.csv'   # Default results filename
            results_dirname = input_results_path
    except IndexError :
        # If not given a results path argument, use default results output
        results_filename = base_filename + '_results.csv'
        results_dirname = __DEFAULT_RESULTS_DIR

    # Create results output directory if non-existent
    if not os.path.exists(results_dirname) :
        try :
            os.makedirs(results_dirname)
        except OSError as exc :
            if exc.errno != errno.EEXIST :
                raise
    
    # Write CSV results to results file
    results_filepath = os.path.join(results_dirname, results_filename)
    results_df.to_csv(results_filepath)