import pandas as pd

import sys, ast

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