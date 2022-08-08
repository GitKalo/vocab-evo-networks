import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multiprocessing.pool import Pool

from . import agent

class Simulation :
    """
    Simulations are independent and immutable instances of the vocabulary evolution 
    model. They are described by the parameters provided.
    
    The parameters of the simulation are passed when the class is instantiated, and 
    are immutable. An instance of the simulation using those parameteres can then 
    be run using the `run()` method, which executes the model with the specified 
    parameters and records the results. The results can later be retrieved through
    the `as_series()` and `as_dict()` methods.

    Simulations can also be run using the `sim_runner.py` script, which will read
    parameters from a `json` file, instantiate and run simulations, and record the
    results. This is the most user-friendly and the recommended way to run simulations.
    Run `python sim_runner.py -h` for further details.

    Attributes
    ----------
        TODO: add attribute descriptions

    Parameters (passed to `__init__` method)
    ----------------------------------------
        pop_size : int
            The number of agents in the population; system size.
        t_max : int
            The number of time steps after which the simulations stops.
        nwk_topology : str
            The topology of the network underlying the population. See `supported_topologies` attribute.
        n_runs : int
            The number of independent realizations of the simulation to be run.
        nwk_update : str
            The network update strategy. See `supported_update_strategies` attribute. (default: 'relabel')
        nwk_lattice_periodic : bool
            For lattice topologies, whether to set periodic boundary conditions. (default: True)
        nwk_ring_neighbors : int
            For ring topologies, the number of neighbors for each node.
        nwk_ring_rewire_p : float
            For ring topologies, the probability of rewiring on initialization, used to generate Watts-Strogatz small-world networks. (default: 0)
        nwk_random_p : float
            For random topologies, the connection probability.
        nwk_sf_links : int
            For scale-free topologies (Albert-Barabasi model), the number of new links preferentially attached when a new node is added to the network.
        nwk_clustered_p : float
            For clustered topologies (Holme-Kim model), the probability of creating a triangle for each new connection.
        nwk_rand-reg_degree : int
            For random-regular graphs, the degree of each node.
        nwk_lambda : float
            The rewire probability; or the probability of a rewire event ocurring instead of a repoduction event at each time step. (default: 0)
        nwk_rewire_disconnect : str
            The rewire disconnect strategy. See `rewire_disconnect_strategies` attribute. (default: 'uniform')
        nwk_rewire_disconnect_total : bool
            Whether to use an agent's total payoffs when calculating probability for a neighbor to disconnect. If "False", will use local payoffs instead.
        nwk_rewire_reconnect : str
            The rewire reconnect strategy. See `rewire_reconnect_strategies` attribute. (default: 'uniform')
        nwk_rewire_reconnect_margin : float
            Probability of picking agent to rewire to uniformly at random, rather than with fitness-proportional probability.
        nwk_rewire_threshold : float
            Payoff threshold for neighbors to be eligibile to disconnect. Only used with certain `nwk_rewire_disconnect` strategies.
        n_objects : int
            Number of objects in agents' languages. (default: agent.Agent.default_objects)
        n_signals : int
            Number of signals in agents' languages. (default: agent.Agent.default_signals)
        sample_strategy : str
            The strategy employed for sampling during the learning process. See `supported_sampling_strategies` attribute. (default: 'fitness-proportional')
        sample_localize : bool
            Whether to localize sampling (sample only from parent's neighbors). If "False", new agents will sample from the entire population. (default: True)
        sample_size : int
            Number of agents to sample from during learning. (default: 4)
        sample_num : int
            Number of samples to draw per object from each agent during learning. (default: 1)
        sample_influence : float
            The neighbor influence; or how samples from neighbors are weighed vs samples from the parent. Values closer to one place more weight on neighbor sample at the expense of the parent sample, and vice versa. (default: 1)
        sample_mistake_p : float
            Probability of making a mistake when sampling for each object. (default: 0)
        comp : bool
            Whether to simulate language competition or not. Drastically changes simulations by initializing the population with only two languages. (default: False)
        comp_initp : float
            The initial language proportion for competition simulations. For example, a value of 0.2 will result in a 20:80 proportion of the two langauges at the initial generation.
        payoff_reports_n : int
            The number of times to record payoffs (as well as languages and network stats), evenly distributed along the number of time steps. (default: 1000)
        n_processes : int
            The number of system processes to use for running independent realizations of the simulation. Used for parallelization and drastically speeds up execution. If no value is set, defaults to the number of runs or the maximum number of available processors.

    Methods
    -------
        TODO: Add method descriptions
    """

    supported_topologies = [
        'lattice',
        'ring',
        'complete',
        'random',
        'rand-reg',
        'scale-free',
        'clustered'
    ]

    supported_update_strategies = [
        'regenerate',
        'relabel'
    ]

    supported_sampling_strategies = [
        'fitness-proportional',
        'random'
    ]

    rewire_disconnect_strategies = [
        'uniform',
        'inverse',
        'inverse-threshold',
        'proportional',
        'proportional-threshold'
    ]

    rewire_reconnect_strategies = [
        'uniform',
        'proportional'
    ]

    # Params that default to None require setting at initialization 
    _default_params = {
        'nwk_update': 'relabel',
        'nwk_lattice_periodic': True,
        'nwk_ring_rewire_p': 0,
        'nwk_ring_neighbors': None,
        'nwk_random_p': None,
        'nwk_sf_links': None,
        'nwk_clustered_p': None,
        'nwk_rand-reg_degree': None,
        'nwk_lambda': 0,
        'nwk_rewire_disconnect': 'uniform',
        'nwk_rewire_disconnect_total': None,
        'nwk_rewire_reconnect': 'uniform',
        'nwk_rewire_reconnect_margin': None,
        'nwk_rewire_threshold': None,
        'n_objects': agent.Agent.default_objects,
        'n_signals': agent.Agent.default_signals,
        'sample_strategy': 'fitness-proportional',
        'sample_localize': True,
        'sample_size': 4,
        'sample_num': 1,
        'sample_influence': 1,
        'sample_mistake_p': 0,
        'comp': False,
        'comp_initp': None,
        'payoff_reports_n': 1000,
        'n_processes': None
    }

    def __init__(self, pop_size, t_max, n_runs, nwk_topology, **kwargs) :
        self._params = {}

        self._params['pop_size'] = pop_size
        self._params['t_max'] = t_max
        self._params['nwk_topology'] = nwk_topology
        self._params['n_runs'] = n_runs

        self._params.update(self._default_params)
        for k, v in kwargs.items() :
            if k in self._params :
                self._params[k] = v
            else :
                raise ValueError(f"Unrecognized keyword argument: '{k}'")

        # Input validation for network type and update strategy
        if self._params['nwk_topology'] in self.__class__.supported_topologies :
            if self._params['nwk_topology'] == 'lattice' :
                if np.sqrt(self._params['pop_size']) % 1 > 0 :
                    raise ValueError("For regular lattices, the pop size must be a square number.")
                else :
                    self._params['nwk_lattice_dim_size'] = int(np.sqrt(self._params['pop_size']))
            elif self._params['nwk_topology'] == 'ring' and self._params['nwk_ring_neighbors'] is None :
                raise TypeError("For ring graphs, the 'nwk_ring_neighbors' parameter should be specified.")
            elif self._params['nwk_topology'] == 'random' and self._params['nwk_random_p'] is None :
                raise TypeError("For random networks, the 'nwk_random_p' parameter should be specified.")
            elif self._params['nwk_topology'] == 'rand-reg' and self._params['nwk_rand-reg_degree'] is None :
                raise TypeError("For random regular networks, the 'nwk_rand-reg_degree' parameter should be specified.")
            elif self._params['nwk_topology'] == 'scale-free' and self._params['nwk_sf_links'] is None :
                raise TypeError("For scale-free networks, the 'nwk_sf_links' parameter should be specified.")
            elif self._params['nwk_topology'] == 'clustered' and (self._params['nwk_sf_links'] is None or self._params['nwk_clustered_p'] is None) :
                raise TypeError("For clustered networks, both the 'nwk_sf_links' and 'nwk_clustered_p' parameters should be specified.")
        else :
            raise ValueError(f"Unrecognized network topology: '{self._params['nwk_topology']}'")

        if self._params['nwk_update'] not in self.__class__.supported_update_strategies :
            raise ValueError(f"Unrecognized network update strategy: '{self._params['nwk_update']}'")

        if self._params['nwk_rewire_disconnect'] not in self.__class__.rewire_disconnect_strategies :
            raise ValueError(f"Unrecognized rewire disconnect strategy: '{self._params['nwk_rewire_disconnect']}'")

        if self._params['nwk_rewire_disconnect'] != 'uniform' and self._params['nwk_rewire_disconnect_total'] is None :
            raise TypeError("For non-'uniform' disconnect strategies, the 'nwk_rewire_disconnect_total' parameter should be specified.")

        if self._params['nwk_rewire_disconnect'] in ['inverse-threshold', 'proportional-threshold'] :
            if self._params['nwk_rewire_disconnect_total'] is not False :
                raise TypeError("For '-threshold' disconnect strategies, payoffs should be individual, not total.")
            if self._params['nwk_rewire_threshold'] is None :
                raise TypeError("For '-threshold' disconnect strategies, the 'nwk_rewire_threshold' parameter should be specified.")

        if self._params['nwk_rewire_reconnect'] not in self.__class__.rewire_reconnect_strategies :
            raise ValueError(f"Unrecognized rewire reconnect strategy: '{self._params['nwk_rewire_reconnect']}'")

        if self._params['nwk_rewire_reconnect'] == 'proportional' and self._params['nwk_rewire_reconnect_margin'] is None :
            raise TypeError("For the 'proportional' reconnect strategy, the 'nwk_rewire_reconnect_margin' parameter should be specified.")

        if self._params['sample_strategy'] not in self.__class__.supported_sampling_strategies :
            raise ValueError(f"Unrecognzied sampling strategy: '{self._params['sample_strategy']}'")

        if self._params['comp'] :
            if self._params['comp_initp'] is None :
                raise ValueError(f"For competition simulations, initial proportion 'comp_initp' should be specified.")
            else :
                assert 0 <= self._params['comp_initp'] <= 1, "Initial proportion 'comp_initp' param must be in range [0, 1]."

        # Payoff reports cannot be more than time steps simulated
        self._params['payoff_reports_n'] = min(self._params['t_max'], self._params['payoff_reports_n'])

        # Determine which time steps to record reports on
        self._params['payoff_reports_i'] = np.linspace(0, self._params['t_max'] - 1, self._params['payoff_reports_n'], dtype=int)
        
        # Default number of processes to the number of simulation runs
        if self._params['n_processes'] is None :
            self._params['n_processes'] = self._params['n_runs']

        # Initialize results containers
        self.__sim_avg_payoffs = np.zeros((self._params['n_runs'], self._params['payoff_reports_n']))     # Average payoffs for each run
        self.__sim_node_payoffs = np.zeros((self._params['n_runs'], self._params['payoff_reports_n'], self._params['pop_size']))     # Node payoffs for each run, populated if network update is 'relabel'
        self.__sim_node_langs = [[]] * self._params['n_runs']
        self.__sim_networks = np.array([nx.Graph] * self._params['n_runs'])

        self.__reports = {
            'rewires': np.zeros((self._params['n_runs'], self._params['payoff_reports_n']), dtype=int),
            'max_degree': np.zeros((self._params['n_runs'], self._params['payoff_reports_n']), dtype=int),
            'avg_shortest_path': np.zeros((self._params['n_runs'], self._params['payoff_reports_n'])),
            'avg_clustering': np.zeros((self._params['n_runs'], self._params['payoff_reports_n'])),
            'transitivity': np.zeros((self._params['n_runs'], self._params['payoff_reports_n']))
        }

    def run(self) :
        """
        Executes the simulation, records the results, and displays them through `pyplot`.
        """

        if self._params['n_processes'] == 1 :
            for i_run in range(self._params['n_runs']) :
                _, avg, node, langs, nwks, reports_dict = self.exec_run(i_run)

                self.__sim_avg_payoffs[i_run] = avg
                self.__sim_node_payoffs[i_run] = node
                self.__sim_node_langs[i_run] = langs
                self.__sim_networks[i_run] = nwks

                for k, v in reports_dict.items() :
                    self.__reports[k][i_run] = v
        else :
            with Pool(self._params['n_processes']) as pool :
                results = pool.map(self.exec_run, range(self._params['n_runs']))
                for i_run, avg, node, langs, nwks, reports_dict in results :
                    self.__sim_avg_payoffs[i_run] = avg
                    self.__sim_node_payoffs[i_run] = node
                    self.__sim_node_langs[i_run] = langs
                    self.__sim_networks[i_run] = nwks

                    for k, v in reports_dict.items() :
                        self.__reports[k][i_run] = v

    def exec_run(self, i_run) :
        # Re-seed rng to get different results in parallel processes
        np.random.seed()

        # Generate agents in first generation (with random matrices)
        first_gen = {agent_id : agent.Agent(agent_id, self._params['n_objects'], self._params['n_signals']) for agent_id in range(self._params['pop_size'])}

        # If simulating competition dynamics, generate initial languages and generation
        if self._params['comp'] :
            # Generate languages A and B (binary, equal payoffs)
            binary_dist = [1] + [0] * (self._params['n_objects'] - 1)
            lang_a = np.array([np.random.permutation(binary_dist) for _ in range(self._params['n_signals'])])

            max_payoff = np.count_nonzero(np.sum(lang_a, axis=0))   # Number of non-zero signal distributions is a proxy for max payoff in binary matrices
            payoff_b = 0
            while (max_payoff != payoff_b) :
                lang_b = np.array([np.random.permutation(binary_dist) for _ in range(self._params['n_signals'])])
                payoff_b = np.count_nonzero(np.sum(lang_b, axis=0))

            # Determine number of A and B agents in population
            num_a = int(np.round(self._params['pop_size'] * self._params['comp_initp']))
            num_b = self._params['pop_size'] - num_a

            # Update languages
            agent_ids = np.array(list(first_gen.keys()))
            np.random.shuffle(agent_ids)

            for agent_a in agent_ids[:num_a] :
                first_gen[agent_a].update_language(lang_a)
            for agent_b in agent_ids[num_a:] :
                first_gen[agent_b].update_language(lang_b)
        else:
            for _, v in first_gen.items() :
                v.update_language(agent.random_assoc_matrix(self._params['n_objects'], self._params['n_signals']))
        
        # Generate network and embed first generation
        G = nx.relabel_nodes(self.generate_network(), first_gen)
        run_network_initial = G.copy()

        node_payoffs = np.zeros(self._params['pop_size'])

        run_avg_payoffs = np.zeros(self._params['payoff_reports_n'])   # Contains the average payoffs for each time step
        run_node_payoffs = np.zeros((self._params['payoff_reports_n'], self._params['pop_size']))   # Payoffs for each node, populated if network update is 'relabel'
        run_langs = [[]] * self._params['payoff_reports_n']

        run_reports_dict = {k: np.zeros(v.shape[1]) for k, v in self.__reports.items()}

        reports_counter = 0
        reports_next_step = 0
        rewire_counter = 0

        for step_num in range(self._params['t_max']) :
            # Decide whether to do a reproduction or a rewire step
            if np.random.binomial(1, self._params['nwk_lambda']) :
                # Rewire step
                G, node_payoffs, rewire_success = self.step_rewire(G)
                if rewire_success : rewire_counter += 1     # Record number of rewires
            else :
                # Reproduction step
                G, node_payoffs = self.step_reproduction(G)
            
            # Record payoffs
            if step_num == reports_next_step :
                # If nodes are relabeled, record payoff for each node
                if self._params['nwk_update'] == 'relabel' :
                    run_node_payoffs[reports_counter] = node_payoffs
        
                # Record average payoffs
                macro_average_payoff = np.mean(node_payoffs) if node_payoffs.size else None
                run_avg_payoffs[reports_counter] = macro_average_payoff

                # Take snapshot of node languages
                run_langs[reports_counter] = [a.active_matrix.tolist() for a in list(G.nodes)]

                # Generate reports
                run_reports_dict['rewires'][reports_counter] = rewire_counter
                run_reports_dict['max_degree'][reports_counter] = max(dict(G.degree).values())
                run_reports_dict['avg_shortest_path'][reports_counter] = self.avg_shortest_path_cc(G)
                run_reports_dict['avg_clustering'][reports_counter] = nx.average_clustering(G)
                run_reports_dict['transitivity'][reports_counter] = nx.transitivity(G)

                reports_counter += 1
                try :
                    reports_next_step = self._params['payoff_reports_i'][reports_counter]
                except IndexError :     # Ignore error at last payoff report
                    pass

        run_network = G.copy()

        return (i_run, run_avg_payoffs, run_node_payoffs, run_langs, (run_network_initial, run_network), run_reports_dict)

    def step_rewire(self, G) :
        agents = list(G.nodes)

        # Calculate symmetric payoffs and add as graph attribute
        payoffs = {(a1, a2): {'payoff': agent.payoff(a1, a2)} for a1, a2 in list(G.edges())}
        nx.set_edge_attributes(G, payoffs)

        # Calculate total and normalized payoffs for each agent
        total_payoffs = np.fromiter(self.get_total_payoffs(G), float)
        normalized_payoffs = self.get_normalized_payoffs(total_payoffs)

        # Pick agent for rewire and prepare rewire pools

        a_source = np.random.choice(agents)
        disconnect_pool = list(nx.neighbors(G, a_source))

        # If we picked an isolated agent, do nothing
        if not disconnect_pool :
            return G, total_payoffs, False

        a_id = agents.index(a_source)
        reconnect_pool = np.delete(agents, a_id)     # Prevent self-links
        reconnect_payoffs = np.delete(normalized_payoffs, a_id)

        # Pick agent to disconnect from based on strategy
        if self._params['nwk_rewire_disconnect'] == 'uniform' :
            a_old = np.random.choice(disconnect_pool)
        else :
            disconnect_ids = [agents.index(a) for a in disconnect_pool]

            if self._params['nwk_rewire_disconnect_total'] :
                disconnect_payoffs = self.get_normalized_payoffs(normalized_payoffs[disconnect_ids])
            else :
                disconnect_payoffs = self.get_normalized_payoffs(np.array([G.get_edge_data(a_source, a, None)['payoff'] for a in disconnect_pool]))

            if self._params['nwk_rewire_disconnect'] in ['inverse-threshold', 'proportional-threshold'] :
                disconnect_pool = np.array(disconnect_pool)     # Convert to ndarray for easier indexing
                eligible_mask = disconnect_payoffs < self._params['nwk_rewire_threshold']

                if self._params['nwk_rewire_disconnect'] == 'proportional-threshold':
                    eligible_mask = ~eligible_mask
                
                eligible = disconnect_pool[eligible_mask]

                if len(eligible) > 0 :
                    a_old = np.random.choice(eligible)
                else :
                    return G, total_payoffs, False
            elif len(disconnect_pool) > 1 :
                if self._params['nwk_rewire_disconnect'] == 'inverse' :
                    try :
                        a_old = np.random.choice(disconnect_pool, p=((1 - disconnect_payoffs)) / (len(disconnect_payoffs) - 1))
                    except ValueError :
                        # Not the best way to handle this error, but it occurs so rarely that it should not affect the dynamics
                        a_old = np.random.choice(disconnect_pool)
                elif self._params['nwk_rewire_disconnect'] == 'proportional' :
                    a_old = np.random.choice(disconnect_pool, p=disconnect_payoffs)
            else :
                a_old = disconnect_pool[0]

        # Prevent same and duplicate connections
        disconnect_mask = np.isin(reconnect_pool, disconnect_pool, invert=True)
        reconnect_pool = reconnect_pool[disconnect_mask]

        # Pick agent to reconnect to based on strategy
        if self._params['nwk_rewire_reconnect'] == 'uniform' :
            a_new = np.random.choice(reconnect_pool)
        elif self._params['nwk_rewire_reconnect'] == 'proportional' :
            reconnect_payoffs = self.get_normalized_payoffs(reconnect_payoffs[disconnect_mask])

            if np.random.binomial(1, self._params['nwk_rewire_reconnect_margin']) :
                # Chance to rewire to a random agent
                a_new = np.random.choice(reconnect_pool)
            else :
                try :
                    a_new = np.random.choice(reconnect_pool, p=reconnect_payoffs)
                except ValueError :
                    # If no other connections exist in the population, rewire to a random agent
                    a_new = np.random.choice(reconnect_pool)

        G.remove_edge(a_source, a_old)
        G.add_edge(a_source, a_new)

        return G, total_payoffs, True

    def step_reproduction(self, G) :
        """
        Simulates communication, reproduction, and langauge learning of agents
        on a network.

        Returns an updated version of the population embedded in the network
        and the average payoff of single communication.
        """
        agents = list(G.nodes)

        # Calculate symmetric payoffs and add as graph attribute
        payoffs = {(a1, a2): {'payoff': agent.payoff(a1, a2)} for a1, a2 in list(G.edges())}
        nx.set_edge_attributes(G, payoffs)

        # Calculate total payoffs for each agent (over communication with neighbors)
        total_payoffs = np.fromiter(self.get_total_payoffs(G), float)

        normalized_payoffs = self.get_normalized_payoffs(total_payoffs)

        # TODO: Fix function calls for 'regenerate'
        if self._params['nwk_update'] == 'regenerate' :
            # Create new generation (of the same size)
            new_generation = []
            for n in range(len(agents)) :
                # Pick parent proportional to fitness
                try :
                    parent = np.random.choice(agents, p=normalized_payoffs)
                except ValueError as err :
                    parent = np.random.choice(agents)

                # Create child that samples A from parent
                child = agent.Agent(n, self._params['n_objects'], self._params['n_signals'])
                if self._params['sample_localize'] == True :
                    parent_neighbors = list(nx.neighbors(G, parent))
                    neighbor_mask = [a in parent_neighbors for a in agents]
                    neighbor_payoffs = normalized_payoffs[neighbor_mask]
                    child.update_language(self.get_sampled_matrix_local(parent, parent_neighbors, neighbor_payoffs))
                else :
                    child.update_language(self.get_sampled_matrix_global(parent, agents, normalized_payoffs))

                new_generation.append(child)
            # Generate new network and embed new generation
            new_G = nx.relabel_nodes(self.generate_network(), {idx:agent for idx, agent in enumerate(new_generation)})
        elif self._params['nwk_update'] == 'relabel' :
            # Pick parent proportional to fitness
            try :
                parent = np.random.choice(agents, p=normalized_payoffs)
            except ValueError as err :
                parent = np.random.choice(agents)

            # Pick random neighbour of parent to replace
            parent_neighbors = list(nx.neighbors(G, parent))
            try :
                neighbor = np.random.choice(parent_neighbors)
            except ValueError :
                # If parent has no neighbors, replace parent
                neighbor = parent

            # Create child
            child = agent.Agent(parent.get_id(), self._params['n_objects'], self._params['n_signals'])
            if self._params['sample_localize'] == True :
                sample_pool = parent_neighbors
                sample_mask = [a in sample_pool for a in agents]
                sample_payoffs = normalized_payoffs[sample_mask]
            else :
                sample_pool = agents
                sample_payoffs = normalized_payoffs
            
            child.update_language(self.get_sampled_matrix(parent, sample_pool, sample_payoffs))

            # Generate new network by replacing neighbor with child
            new_G = nx.relabel_nodes(G, {neighbor:child})

        # Return new network and the individual payoffs for all agents
        return new_G, total_payoffs

    def generate_network(self) :
        """
        Generate a network based on the `network_type` property.
        """
        nwk = self._params['nwk_topology']
        if nwk == 'lattice' :
            G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(self._params['nwk_lattice_dim_size'], self._params['nwk_lattice_dim_size'], periodic=self._params['nwk_lattice_periodic']))
        elif nwk == 'ring' :
            G = nx.watts_strogatz_graph(self._params['pop_size'], self._params['nwk_ring_neighbors'], self._params['nwk_ring_rewire_p'])
        elif nwk == 'complete' :
            G = nx.complete_graph(self._params['pop_size'])
        elif nwk == 'random' :
            G = nx.erdos_renyi_graph(self._params['pop_size'], self._params['nwk_random_p'])
        elif nwk == 'rand-reg' :
            G = nx.random_regular_graph(self._params['nwk_rand-reg_degree'], self._params['pop_size'])
        elif nwk == 'scale-free' :
            G = nx.barabasi_albert_graph(self._params['pop_size'], self._params['nwk_sf_links'])
        elif nwk == 'clustered' :
            G = nx.powerlaw_cluster_graph(self._params['pop_size'], self._params['nwk_sf_links'], self._params['nwk_clustered_p'])

        return G

    def get_sampled_matrix(self, parent, sample_pool, sample_payoffs) :
        # Sample from parent — number of samples used to maintain the same fidelity as neighbor sample
        parent_sample = agent.sample(parent, self._params['sample_size'], self._params['sample_mistake_p'])

        # If there are no agents to sample from, use parent sample
        if not sample_pool :
            return parent_sample

        # Sample from pool (neighbors for localized learning, population otherwise)
        if self._params['sample_strategy'] == 'fitness-proportional' :
            try :
                neighbor_sample_models = np.random.choice(sample_pool, size=self._params['sample_size'], p=sample_payoffs)
            except ValueError :
                neighbor_sample_models = np.random.choice(sample_pool, size=self._params['sample_size'])
        elif self._params['sample_strategy'] == 'random' :
            neighbor_sample_models = np.random.choice(sample_pool, size=self._params['sample_size'])

        neighbor_sample = np.sum(list(map(lambda m : agent.sample(m, self._params['sample_num'], self._params['sample_mistake_p']), neighbor_sample_models)), axis=0)

        # Scale sample based on influence
        parent_sample_influence = parent_sample * (1 - self._params['sample_influence'])
        neighbor_sample_influence = neighbor_sample * self._params['sample_influence']

        # Return sampled matrix
        return parent_sample_influence + neighbor_sample_influence

    def get_total_payoffs(self, G) :
        for a in G :
            links = G.edges(nbunch=a, data=True)
            agent_total_payoff = np.sum([d['payoff'] for _, _, d in links])
            
            try :
                agent_total_payoff = float(agent_total_payoff) / len(links)
            except ZeroDivisionError :
                pass

            yield agent_total_payoff

    def get_normalized_payoffs(self, total_payoffs) :
        sum_payoffs = np.sum(total_payoffs)
        normalized_payoffs = total_payoffs
        if sum_payoffs : normalized_payoffs = np.array(total_payoffs) / sum_payoffs

        return normalized_payoffs

    def avg_shortest_path_cc(self, G) :
        try :
            return nx.average_shortest_path_length(G)
        except nx.NetworkXError :
            return nx.average_shortest_path_length(G.subgraph(max(nx.connected_components(G), key=len)))

    def as_series(self, include_payoffs=True, include_langs=False, include_reports=True) :
        series = pd.Series(self.as_dict(include_payoffs, include_langs, include_reports))
        return series

    def as_dict(self, include_payoffs=False, include_langs=False, include_reports=False) :
        sim_dict = self.get_params()
        if include_payoffs :
            sim_dict.update(dict(
                avg_payoffs=self.get_avg_payoffs().tolist(),
                node_payoffs=self.get_node_payoffs().tolist(),
                ))
        if include_langs :
            sim_dict.update(dict(node_langs=self.get_node_langs()))
        if include_reports :
            sim_dict.update({k: v.tolist() for k, v in self.__reports.items()})

        return sim_dict

    def get_params(self) :
        output_params = self._params
        del output_params['payoff_reports_i']
        
        return output_params

    def get_networks(self) :
        """
        Return the list of last-time-step networks for all runs.
        """
        # TODO: validate that network attribute exists
        return self.__sim_networks

    def get_node_langs(self) :
        return self.__sim_node_langs

    def get_avg_payoffs(self) :
        return self.__sim_avg_payoffs

    def get_node_payoffs(self) :
        return self.__sim_node_payoffs

def main() :
    pop_size = 20
    n_time_steps = 5000
    n_runs = 4
    topology = 'scale-free'

    simulation = Simulation(pop_size, n_time_steps, n_runs, topology, nwk_sf_links=4)

    simulation.run()

    # Plot and display the average payoff
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for payoffs in simulation.get_avg_payoffs() :
        ax1.plot(payoffs, color='blue')
    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('Average payoffs')
    ax1.set_title('Results')

    Gs = simulation.get_networks()
    nx.draw(Gs[0][1], ax=ax2)
    ax2.set_title('Network')

    plt.show()

if __name__ == '__main__' :

    print("This module is used to run agent-based simulations. See `help(sim)` for further details. It will now run a quick sample simulation and plot the results.")

    main()