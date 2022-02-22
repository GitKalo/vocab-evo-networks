import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy.random.mtrand import sample
import pandas as pd

from multiprocessing.pool import Pool

from . import agent, util

class Simulation :
    """
    Simulations are independent and immutable instances of the vocabulary evolution 
    model with certain parameters (see below).
    
    The parameters of the simulation are passed when the class is instantiated, and 
    cannot be changed. The instance can then be run using the `run()` method, which 
    executes the model with the specified parameters and records the results.

    The parameters that can be specified to the Simulation are the following:
        pop_size : int          # The number of agents in the population
        time_steps : int        # The number of time steps at each run of the simulation.
        ...                     Intuitively, every time step represents a new generation, although the 
        ...                     details of what that entails differ base don the network update strategy.
        runs : int              # The number of independent runs that the simulation executes. Each
        ...                     has a separate network structure and population. The final results reported
        ...                     by the simulation are averaged over all runs.
        objects : int           # The number of objects in every agent's language.
        signals : int           # The number of signals in every agent's language.
        network_type : str      # The type of network used, which determines the structure of the population.
        ...                     Must be one of the available network types listed in the `network_types` attribute.
        network_update : str    # The network update strategy used by the simulation. Must be one of the
        ...                     available network update strategies listed in the `network_updates` attribute.
    """
    supported_topologies = [
        'lattice',
        'lattice_extra',
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

    learning_strategies = [
        'parental',
        'role-model',
        'random'
    ]

    rewire_disconnect_strategies = [
        'uniform',
        'inverse'
    ]

    rewire_reconnect_strategies = [
        'uniform',
        'proportional'
    ]

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
        'nwk_rewire_reconnect': 'proportional',
        'n_objects': agent.Agent.default_objects,
        'n_signals': agent.Agent.default_signals,
        'sample_strategy': 'role-model',
        'sample_localize': True,
        'sample_size': 4,
        'sample_num': 1,
        'sample_influence': 0.5,
        'sample_mistake_p': 0,
        'sample_include_parent': False,    #TODO: phase out
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
            if self._params['nwk_topology'] in ['lattice', 'lattice_extra'] :
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

        if self._params['nwk_rewire_reconnect'] not in self.__class__.rewire_reconnect_strategies :
            raise ValueError(f"Unrecognized rewire reconnect strategy: '{self._params['nwk_rewire_reconnect']}'")

        if self._params['sample_strategy'] not in self.__class__.learning_strategies :
            raise ValueError(f"Unrecognzied sampling strategy: '{self._params['sample_strategy']}'")

        self._params['payoff_reports_i'] = np.linspace(0, self._params['t_max'] - 1, self._params['payoff_reports_n'], dtype=int)
        
        # Default number of processes to the number of simulation runs
        if self._params['n_processes'] is None :
            self._params['n_processes'] = self._params['n_runs']

        # Initialize results containers
        self.__sim_avg_payoffs = np.zeros((self._params['n_runs'], self._params['payoff_reports_n']))     # Average payoffs for each run
        self.__sim_node_payoffs = np.zeros((self._params['n_runs'], self._params['payoff_reports_n'], self._params['pop_size']))     # Node payoffs for each run, populated if network update is 'relabel'
        self.__sim_node_langs = [[]] * self._params['n_runs']
        self.__sim_networks = np.array([nx.Graph] * self._params['n_runs'])

    def run(self) :
        """
        Executes the simulation, records the results, and displays them through `pyplot`.
        """

        if self._params['n_processes'] == 1 :
            for i_run in range(self._params['n_runs']) :
                _, avg, node, langs, nwks = self.exec_run(i_run)

                self.__sim_avg_payoffs[i_run] = avg
                self.__sim_node_payoffs[i_run] = node
                self.__sim_node_langs[i_run] = langs
                self.__sim_networks[i_run] = nwks
        else :
            with Pool(self._params['n_processes']) as pool :
                results = pool.map(self.exec_run, range(self._params['n_runs']))
                for i_run, avg, node, langs, nwks in results :
                    self.__sim_avg_payoffs[i_run] = avg
                    self.__sim_node_payoffs[i_run] = node
                    self.__sim_node_langs[i_run] = langs
                    self.__sim_networks[i_run] = nwks

    def exec_run(self, i_run) :
        # Re-seed rng to get different results in parallel processes
        np.random.seed()

        # Generate agents in first generation (with random matrices)
        first_gen = {agent_id : agent.Agent(agent_id, self._params['n_objects'], self._params['n_signals']) for agent_id in range(self._params['pop_size'])}
        for _, v in first_gen.items() :
            v.update_language(agent.random_assoc_matrix(self._params['n_objects'], self._params['n_signals']))
        
        # Generate network and embed first generation
        G = nx.relabel_nodes(self.generate_network(), first_gen)
        run_network_initial = G.copy()

        node_payoffs = np.zeros(self._params['pop_size'])

        run_avg_payoffs = np.zeros(self._params['payoff_reports_n'])   # Contains the average payoffs for each time step
        run_node_payoffs = np.zeros((self._params['payoff_reports_n'], self._params['pop_size']))   # Payoffs for each node, populated if network update is 'relabel'
        run_langs = [[]] * self._params['payoff_reports_n']

        reports_counter = 0
        reports_next_step = 0

        for step_num in range(self._params['t_max']) :
            # Decide whether to do a reproduction or a rewire step
            if np.random.binomial(1, self._params['nwk_lambda']) :
                # Rewire step
                G, node_payoffs = self.step_rewire(G)
            else :
                # Reproduction step
                G, node_payoffs = self.step_reproduction(G)
            
            # Record payoffs
            if step_num == reports_next_step :   #TODO: optimize (set or next_report_id)
                # If nodes are relabeled, record payoff for each node
                if self._params['nwk_update'] == 'relabel' :
                    run_node_payoffs[reports_counter] = node_payoffs
        
                # Record average payoffs
                macro_average_payoff = np.mean(node_payoffs) if node_payoffs.size else None
                run_avg_payoffs[reports_counter] = macro_average_payoff

                # Take snapshot of node languages
                run_langs[reports_counter] = [a.active_matrix.tolist() for a in list(G.nodes)]
        
                reports_counter += 1
                try :
                    reports_next_step = self._params['payoff_reports_i'][reports_counter]
                except IndexError :     # Ignore error at last payoff report
                    pass

        run_network = G.copy()

        return (i_run, run_avg_payoffs, run_node_payoffs, run_langs, (run_network_initial, run_network))

    def step_rewire(self, G) :
        agents = list(G.nodes)

        # Calculate symmetric payoffs and add as graph attribute
        payoffs = {(a1, a2): {'payoff': agent.payoff(a1, a2)} for a1, a2 in list(G.edges())}
        nx.set_edge_attributes(G, payoffs)

        # Calculate total and normalized payoffs for each agent
        total_payoffs = np.fromiter(self.get_total_payoffs(G), float)

        sum_payoffs = np.sum(total_payoffs)
        normalized_payoffs = total_payoffs
        if sum_payoffs : normalized_payoffs = np.array(total_payoffs) / sum_payoffs

        # Pick agent for rewire and prepare rewire pools

        a_source = np.random.choice(agents)
        disconnect_pool = list(nx.neighbors(G, a_source))

        # If we picked an isolated agent, do nothing
        if not disconnect_pool :
            return G, total_payoffs

        a_id = agents.index(a_source)
        reconnect_pool = np.delete(agents, a_id)     # Prevent self-links
        reconnect_payoffs = np.delete(normalized_payoffs, a_id)

        # Pick agent to disconnect from based on strategy
        if self._params['nwk_rewire_disconnect'] == 'uniform' :
            a_old = np.random.choice(disconnect_pool)
        elif self._params['nwk_rewire_disconnect'] == 'inverse' :
            disconnect_ids = [agents.index(a) for a in disconnect_pool]
            disconnect_payoffs = self.get_normalized_payoffs(normalized_payoffs[disconnect_ids])

            if len(disconnect_pool) > 1 :
                # TODO: Dangerous! Not sure why this probability inversion works, but it seems to. Need to test! 
                a_old = np.random.choice(disconnect_pool, p=((1 - disconnect_payoffs)) / (len(disconnect_payoffs) - 1))
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

            try :
                a_new = np.random.choice(reconnect_pool, p=reconnect_payoffs)
            except ValueError :
                # If no other connections exist in the population, 
                # rewire to a random agent in the pool
                a_new = np.random.choice(reconnect_pool)

        G.remove_edge(a_source, a_old)
        G.add_edge(a_source, a_new)

        return G, total_payoffs

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
                    child.update_language(self.get_sampled_matrix_local(parent, parent_neighbors, neighbor_payoffs))    # Change to include only neighbours
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
                if self._params['sample_include_parent'] == True :
                    sample_pool = [parent, *parent_neighbors]
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
        if nwk in ['lattice', 'lattice_extra'] :
            G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(self._params['nwk_lattice_dim_size'], self._params['nwk_lattice_dim_size'], periodic=self._params['nwk_lattice_periodic']))

            if nwk == 'lattice_extra' :
                # Add a random edge
                # nodes = np.random.choice(G.nodes, size=2, replace=False)
                # G.add_edge(nodes[0], nodes[1])

                # Remove a random edge
                edge = list(G.edges)[np.random.choice(len(list(G.edges)))]
                G.remove_edge(edge[0], edge[1])
        elif nwk == 'ring' :
            G = nx.watts_strogatz_graph(self._params['pop_size'], self._params['nwk_ring_neighbors'], self._params['nwk_ring_rewire_p'])
        elif nwk == 'complete' :
            G = nx.complete_graph(self._params['pop_size'])
        elif nwk == 'random' :
            G = nx.erdos_renyi_graph(self._params['pop_size'], self._params['nwk_random_p'])
        elif nwk == 'random_regular' :
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
        if self._params['sample_strategy'] == 'role-model' :
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

    def as_series(self, include_payoffs=True, include_langs=False) :
        series = pd.Series(self.as_dict(include_payoffs, include_langs))
        return series

    def as_dict(self, include_payoffs=False, include_langs=False) :
        sim_dict = self.get_params()
        if include_payoffs :
            sim_dict.update(dict(
                avg_payoffs=self.get_avg_payoffs().tolist(),
                node_payoffs=self.get_node_payoffs().tolist(),
                ))
        if include_langs :
            sim_dict.update(dict(node_langs=self.get_node_langs()))

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

if __name__ == '__main__' :
    pop_size = 50
    n_time_steps = 10000
    n_runs = 20

    simulation = Simulation(pop_size, n_time_steps, n_runs, 
    nwk_topology='scale-free', nwk_update='relabel', sample_strategy='role-model')

    simulation.run()

    # Plot and display the average payoff
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for payoffs in simulation.get_avg_payoffs() :
        ax1.plot(payoffs, color='blue')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Payoff')
    ax1.set_title('Parental learning')

    G_view = simulation.get_network_view()
    nx.draw(G_view, ax=ax2)
    ax2.set_title('Network')

    plt.show()
