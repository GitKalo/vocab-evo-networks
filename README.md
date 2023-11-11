# Vocabulary evolution on complex networks

This project defines an **agent-based simulation** for exploring the effects of complex social structure on the emergence of vocabulary (in the form of lexical convention) in a structured population of linguistic agents. It is based on an evolutionary model based on Nowak's Evolutionary Language Game and defined by the authors in Ref. [1].

Research based on this model has been published and presented at the *Complex Networks & Their Applications X* conference in 2021 [1] and PLOS ONE [2]. The model was originally prosposed as part of a final-year Individual Project (COMP3200) at the University of Southampton.

## Running simulations

To run simulations, you need Python >=3.9 and the necessary third-party modules (most importantly NumPy, Pandas, and NetworkX). To install the latter, run something like the following (exact command may vary depending on your setup):

```bash
python -m pip install -r requirements.txt
```

Simulations can be run directly through a Python script by either importing the `Simulation` class from `src/sim.py` and initializing it with the necessary parameters, or by importing the `sim_runner` module and using its `run_sim` function with a dictionary of parameters. 

However, a quicker and more user-friendly way to run simulations is through the command line, by using a `.json` **parameter file** specifying the relevant simulation parameters. An example parameter file `example_params.json` is provided. The example is very minimal. For details on available parameters, see the comments or help section of the `src/sim.py` module (e.g. by running `python -c "import src.sim as sim; help(sim)"`).

If you have a parameter file, you can **run simulations** using the `sim_runner.py` script, like so (this particular example should take about two minutes to run):

```bash
python sim_runner.py example_params.json example_results.csv -n example_networks.pickle -d ./example_simulation
```

Briefly, this will run simulations based on the parameters provided in the `example_params.json` script and save the results to a new directory named `example_simulation` under the current working directory, which will contain the simulation results (`example_results.csv`) and serialized networks (`example_networks.pickle`). See `python sim_runner.py -h` for further details on the command line arguments. 

There can sometimes be issues with saving results as CSV files, so it is recommended to save use parqet files instead. You can do so by simply changing the file extension of the respective command line argument to (e.g. to `example_results.parq`).

To quickly **visualize the results**, you can use the `plotting.py` script through the command line like so:

```bash
python plotting.py example_simulation/example_results.csv
```

The `analysis.py` script contains various functions that can help in **analyzing simulation results**. Most of them are fairly atomic and easy to understand, but feel free to address the authors (see below) for help.

## Authors

All software was written by **Kaloyan Danovski**. The model is developed and explored by **Kaloyan Danovski** and **Markus Brede**.

To get in touch, please email Kaloyan at kaloyan[dot]danovski[at]gmail[dot]com.

## References

[1] Danovski, K., Brede, M., 2022a. Effects of Population Structure on the Evolution of Linguistic Convention, in: Benito, R.M., Cherifi, C., Cherifi, H., Moro, E., Rocha, L.M., Sales-Pardo, M. (Eds.), Complex Networks & Their Applications X, Studies in Computational Intelligence. Springer International Publishing, Cham, pp. 693–704. https://doi.org/10.1007/978-3-030-93409-5_57

[2] Danovski, K., Brede, M., 2022b. On the evolutionary language game in structured and adaptive populations. PLoS ONE 17, e0273608. https://doi.org/10.1371/journal.pone.0273608
