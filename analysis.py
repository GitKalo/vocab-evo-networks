import pandas as pd

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
        results_df.to_parquet(results_filepath, engine='fastparquet', object_encoding=PARQUET_OBJECT_ENCODINGS)

    print(f"Saved to {os.path.abspath(results_filepath)}.")