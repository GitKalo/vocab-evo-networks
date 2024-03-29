import time, os, sys, json, argparse

import pandas as pd

import analysis
from src import sim

sim_networks = {}

def run_sim(sim_params, include_payoffs=True, include_langs=False) :
    """
    Runs simulations based on a dictionary of simulation parameters. The dictionary has simulation ids as keys, and parameters dicts as values.

    Parameters
    ----------
    include_payoffs : bool (default: True)
        Whether to include payoffs when returning simulation as series. Should be "True" in virtually all cases.
    include_langs : bool (default: False)
        Whether to record agent languages during the simulation. Greatly increases the size of simulation results.

    Returns
    -------
    res_df
        A DataFrame of simulation results.
    sim_networks
        A dictionary containing initial and final networks for all simulation runs.
    """

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
    parser.add_argument('param_filepath', help="Path to parameter file in JSON format.")
    parser.add_argument('results_output_filepath', help="Path for results output. Valid file extensions are '.csv.' or '.parq'. If an ivalid filepath is given, a default one will be used instead.")
    parser.add_argument('-n', '--networks-output-filepath', help="Path for serialized network output. Valid file extensions are '.pickle'. If an invalid filepath is given, a default one will be used instead.")
    parser.add_argument('--include-langs', action='store_true', help="Whether to include agent languages in results report. Will be passed as an argument to `run_sim()` method and Simulation instance.")
    parser.add_argument('-d', '--output-dir', help="Directory to export results and networks to. Overrides directory information in results and networks output arguments.")

    args = parser.parse_args()

    # Get parameter filename
    param_filename = os.path.basename(args.param_filepath)
    
    # Get parameter file
    try :
        sim_params = json.load(open(args.param_filepath, mode='r'))
    except FileNotFoundError :
        print(f"Could not find the parmeter file '{args.param_filepath}'. Exiting...")
        sys.exit()

    common_params = {}
    if 'common' in sim_params :
        common_params = sim_params['common']
        del sim_params['common']
        
        for sim_id, params in sim_params.items() :
            sim_params[sim_id] = {**common_params, **params}

    # Run simulations and record runtime
    print("---  Starting simulation...  ---")
    start_time = time.time()
    check_time = start_time

    results_df, _ = run_sim(sim_params, include_langs=args.include_langs)

    print("---  Finished in %.2f minutes (real time)  ---" % ((time.time() - start_time) / 60))

    # Determine output file and directory names
    base_filename = '_'.join(param_filename.split('_')[0:-1])
    
    results_filepath = args.results_output_filepath
    results_filepath_default = f"./{base_filename}_results.parq"

    if args.output_dir is not None :
        results_filepath = os.path.join(args.output_dir, os.path.basename(results_filepath))
        results_filepath_default = os.path.join(args.output_dir, os.path.basename(results_filepath_default))

    try :
        analysis.export_results(analysis.explode_results(results_df), results_filepath)
    except ValueError as e :
        print(e)
        print(f"Saving to default instead...")
        analysis.export_results(analysis.explode_results(results_df), results_filepath_default)

    if args.networks_output_filepath is not None :
        networks_filepath = args.networks_output_filepath
        networks_filepath_default = f"./{base_filename}_networks.pickle"

        if args.output_dir is not None :
            networks_filepath = os.path.join(args.output_dir, os.path.basename(networks_filepath))
            networks_filepath_default = os.path.join(args.output_dir, os.path.basename(networks_filepath_default))

        try :
            analysis.export_networks_file(sim_networks, networks_filepath)
        except ValueError as e :
            print(e)
            print(f"Saving to default instead...")
            analysis.export_networks_file(sim_networks, networks_filepath_default)