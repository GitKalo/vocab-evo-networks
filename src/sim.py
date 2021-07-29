import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    network_types = [
        'lattice',
        'ring',
        'complete',
        'random',
        'scale-free',
        'clustered'
    ]

    network_updates = [
        'regenerate',
        'relabel'
    ]

    learning_strategies = [
        'parental',
        'role-model',
        'random'
    ]

    def __init__(self, pop_size, time_steps, runs, network_type, network_update,
                    learning='parental',
                    ring_rewire_prob=0,
                    ring_neighbors=None,
                    er_prob=None,
                    ba_links=None, 
                    hk_prob=None, 
                    objects=agent.Agent.default_objects,
                    signals=agent.Agent.default_signals, 
                    sample_num=1, 
                    agents_sampled=2, 
                    p_mistake=0, 
                    localize_learning=False,
                    periodic_lattice=False,
                    n_payoff_reports=1000) :
        self.__pop_size = pop_size
        self.__n_time_steps = time_steps
        self.__n_runs = runs
        self.__n_objects = objects
        self.__n_signals = signals
        self.__n_learning_samples = sample_num
        self.__n_agents_sampled = agents_sampled
        self.__p_mistake = p_mistake

        # Input validation for network type and update strategy
        if network_type in self.__class__.network_types :
            self.__network_type = network_type
            if network_type == 'lattice' :
                if np.sqrt(self.__pop_size) % 1 > 0 :
                    raise ValueError("For regular lattices, the pop size must be a square number.")
                else :
                    self.__lattice_dim_size = int(np.sqrt(self.__pop_size))
            elif network_type == 'ring' and ring_neighbors is None :
                raise TypeError("For ring graphs, the ring_neighbors argument should be passed.")
            elif network_type == 'random' and er_prob is None :
                raise TypeError("For random networks, the er_prob argument should be passed.")
            elif network_type == 'scale-free' and ba_links is None :
                raise TypeError("For scale-free networks, the ba_links argument should be passed.")
            elif network_type == 'clustered' and (ba_links is None or hk_prob is None) :
                raise TypeError("For clustered networks, both the ba_links and hk_prob arguments should be passed.")
        else :
            raise ValueError(f"Network type '{network_type}' not recognized.")

        self.__ring_rewire_prob = ring_rewire_prob
        self.__ring_neighbors = ring_neighbors
        self.__er_prob = er_prob
        self.__ba_links = ba_links
        self.__hk_prob = hk_prob
        self.__periodic_lattice = periodic_lattice

        if network_update in self.__class__.network_updates :
            self.__network_update = network_update
        else :
            raise ValueError(f"Network update strategy '{network_update}'' is not recognized.")

        if learning in self.__class__.learning_strategies :
            self.__learning_strategy = learning
        else :
            raise ValueError(f"Learning strategy '{learning}'' is not recognized.")

        self.__localize_learning=localize_learning

        self.__n_payoff_reports = n_payoff_reports
        self.__i_payoff_reports = np.linspace(0, self.__n_time_steps - 1, n_payoff_reports, dtype=int)

        # Initialize list of networks
        self.__run_networks = np.array([nx.Graph] * self.__n_runs)

    def run(self) :
        """
        Executes the simulation, records the results, and displays them through `pyplot`.
        """
        run_node_payoffs = np.zeros((self.__n_runs, self.__n_payoff_reports, self.__pop_size))   # Node payoffs for each run, populated if network update is 'relabel'
        run_avg_payoffs = np.zeros((self.__n_runs, self.__n_time_steps))    # Contains the average payoffs for each run
        for i_run in range(self.__n_runs) :
            # Generate agents in first generation (with random matrices)
            first_gen = {agent_id : agent.Agent(agent_id, self.__n_objects, self.__n_signals) for agent_id in range(self.__pop_size)}
            for k, v in first_gen.items() :
                v.update_language(agent.random_assoc_matrix(self.__n_objects, self.__n_signals))
            
            # Generate network and embed first generation
            G = nx.relabel_nodes(self.generate_network(), first_gen)

            step_node_payoffs = np.zeros((self.__n_payoff_reports, self.__pop_size))   # Payoffs for each node,, populated if network update is 'relabel'
            step_avg_payoffs = np.zeros(self.__n_time_steps)   # Contains the average payoffs for each time step
            reports_counter = 0

            for step_num in range(self.__n_time_steps) :
                # Simulate communication and reproduction
                G, node_payoffs = self.next_generation(G)
                
                # If nodes are relabeled, record payoff for each node
                if self.__network_update == 'relabel' :
                    if step_num in self.__i_payoff_reports :
                        print(f'lol {step_num}')    # REMOVE
                        step_node_payoffs[reports_counter] = node_payoffs
                        reports_counter += 1

                # Record average payoff
                macro_average_payoff = np.mean(node_payoffs) if node_payoffs.size else None
                step_avg_payoffs[step_num] = macro_average_payoff
            run_node_payoffs[i_run] = step_node_payoffs
            run_avg_payoffs[i_run] = step_avg_payoffs

            self.__run_networks[i_run] = G.copy()

        self.__run_node_payoffs = run_node_payoffs
        self.__run_avg_payoffs = run_avg_payoffs

    def next_generation(self, G) :
        """
        Simulates communication, reproduction, and langauge learning of agents
        on a network.

        Returns an updated version of the population embedded in the network
        and the average payoff of single communication.
        """
        agents = list(G.nodes)
        # Total payoff for each agent (over communication with all others)
        total_payoffs = np.array([])

        # TODO: change variable names (speaker/listener)
        for speaker in agents :
            list_connections = list(nx.neighbors(G, speaker))
            agent_total_payoff = np.sum(agent.payoff(speaker, l) for l in list_connections)

            try :
                agent_total_payoff = agent_total_payoff / len(list_connections)
            except ZeroDivisionError :
                pass

            total_payoffs = np.append(total_payoffs, agent_total_payoff)

        # Generate list of normalized fitness scores
        sum_payoffs = np.sum(total_payoffs)
        normalized_payoffs = total_payoffs
        if sum_payoffs : normalized_payoffs = np.array(total_payoffs) / sum_payoffs

        if self.__network_update == 'regenerate' :
            # Create new generation (of the same size)
            new_generation = []
            for n in range(len(agents)) :
                # Pick parent proportional to fitness
                try :
                    parent = np.random.choice(agents, p=normalized_payoffs)
                except ValueError as err :
                    parent = np.random.choice(agents)

                # Create child that samples A from parent
                child = agent.Agent(n, self.__n_objects, self.__n_signals)
                if self.__localize_learning == True :
                    parent_neighbors = list(nx.neighbors(G, parent))
                    neighbor_mask = [a in parent_neighbors for a in agents]
                    neighbor_payoffs = normalized_payoffs[neighbor_mask]
                    child.update_language(self.get_sampled_matrix_local(parent, parent_neighbors, neighbor_payoffs))    # Change to include only neighbours
                else :
                    child.update_language(self.get_sampled_matrix_global(parent, agents, normalized_payoffs))

                new_generation.append(child)
            # Generate new network and embed new generation
            new_G = nx.relabel_nodes(self.generate_network(), {idx:agent for idx, agent in enumerate(new_generation)})
        elif self.__network_update == 'relabel' :
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
            
            # Create child that samples A from parent
            child = agent.Agent(parent.get_id(), self.__n_objects, self.__n_signals)
            if self.__localize_learning == True :
                neighbor_mask = [a in parent_neighbors for a in agents]
                neighbor_payoffs = normalized_payoffs[neighbor_mask]
                child.update_language(self.get_sampled_matrix_local(parent, parent_neighbors, neighbor_payoffs)) # Change to include only neighbours
            else :
                child.update_language(self.get_sampled_matrix_global(parent, agents, normalized_payoffs))

            # Generate new network by replacing neighbor with child
            new_G = nx.relabel_nodes(G, {neighbor:child})

        # Return new network and the individual payoffs for all agents
        return new_G, total_payoffs

    def generate_network(self) :
        """
        Generate a network based on the `network_type` property.
        """
        if self.__network_type == 'lattice' :
            G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(self.__lattice_dim_size, self.__lattice_dim_size, periodic=self.__periodic_lattice))
        elif self.__network_type == 'ring' :
            G = nx.watts_strogatz_graph(self.__pop_size, self.__ring_neighbors, self.__ring_rewire_prob)
        elif self.__network_type == 'complete' :
            G = nx.complete_graph(self.__pop_size)
        elif self.__network_type == 'random' :
            G = nx.erdos_renyi_graph(self.__pop_size, self.__er_prob)
        elif self.__network_type == 'scale-free' :
            G = nx.barabasi_albert_graph(self.__pop_size, self.__ba_links)
        elif self.__network_type == 'clustered' :
            G = nx.powerlaw_cluster_graph(self.__pop_size, self.__ba_links, self.__hk_prob)

        return G

    def get_sampled_matrix_global(self, parent, pop, pop_payoffs) :
        if self.__learning_strategy == 'parental' :
            A = agent.sample(parent, self.__n_learning_samples, self.__p_mistake)
        elif self.__learning_strategy == 'role-model' :
            try :
                models = np.random.choice(pop, size=self.__n_agents_sampled, p=pop_payoffs)
            except ValueError :
                models = np.random.choice(pop, size=self.__n_agents_sampled)
            A = np.sum(list(map(lambda m : agent.sample(m, self.__n_learning_samples, self.__p_mistake), models)), axis=0)
        elif self.__learning_strategy == 'random' :
            models = np.random.choice(pop, size=self.__n_agents_sampled)
            A = np.sum(list(map(lambda m : agent.sample(m, self.__n_learning_samples, self.__p_mistake), models)), axis=0)

        return A

    def get_sampled_matrix_local(self, parent, neighbors, neighbour_payoffs) :
        if self.__learning_strategy == 'parental' :
            A = agent.sample(parent, self.__n_learning_samples, self.__p_mistake)
        elif self.__learning_strategy == 'role-model' :
            try :
                models = np.random.choice(neighbors, size=self.__n_agents_sampled, p=neighbour_payoffs)
            except ValueError :
                models = np.random.choice(neighbors, size=self.__n_agents_sampled)
            A = np.sum(list(map(lambda m : agent.sample(m, self.__n_learning_samples, self.__p_mistake), models)), axis=0)
        elif self.__learning_strategy == 'random' :
            models = np.random.choice(neighbors, size=self.__n_agents_sampled)
            A = np.sum(list(map(lambda m : agent.sample(m, self.__n_learning_samples, self.__p_mistake), models)), axis=0)

        return A

    def as_series(self, include_payoffs=True) :
        series = pd.Series(self.as_dict(include_payoffs))
        return series

    def as_dict(self, include_payoffs=True) :
        sim_dict = self.get_params()
        if include_payoffs :
            sim_dict.update(dict(
                avg_payoffs=self.get_avg_payoffs().tolist(),
                node_payoffs=self.get_node_payoffs().tolist()
                ))

        return sim_dict

    def get_params(self) :
        params = {
            'pop_size': self.__pop_size,
            'time_steps': self.__n_time_steps,
            'runs': self.__n_runs,
            'vocab_size': (self.__n_objects, self.__n_signals),
            'sample_num': self.__n_learning_samples,
            'agents_sampled': self.__n_agents_sampled,
            'learning_strategy': self.__learning_strategy,
            'p_mistake': self.__p_mistake,
            'network_type': self.__network_type,
            'network_update': self.__network_update,
            'er_prob': self.__er_prob,
            'ba_links': self.__ba_links,
            'hk_prob': self.__hk_prob,
            'localize_learning': self.__localize_learning
        }

        return params

    def get_networks(self) :
        """
        Return the list of last-time-step networks for all runs.
        """
        # TODO: validate that network attribute exists
        return self.__run_networks

    def get_avg_payoffs(self) :
        return self.__run_avg_payoffs

    def get_node_payoffs(self) :
        return self.__run_node_payoffs

    # TODO: implement copy method

if __name__ == '__main__' :
    pop_size = 10
    n_time_steps = 100
    n_runs = 20

    simulation = Simulation(pop_size, n_time_steps, n_runs, 
    network_type='scale-free', network_update='regenerate', ba_links=2, sample_size=5)

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