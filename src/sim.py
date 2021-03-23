import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import elg, util

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
        symbols : int           # The number of symbols in every agent's language.
        network_type : str      # The type of network used, which determines the structure of the population.
        ...                     Must be one of the available network types listed in the `network_types` attribute.
        network_update : str    # The network update strategy used by the simulation. Must be one of the
        ...                     available network update strategies listed in the `network_updates` attribute.
    """
    network_types = [
        'regular',
        'random',
        'scale-free',
        'clustered'
    ]

    network_updates = [
        'regenerate',
        'relabel'
    ]

    def __init__(self, pop_size, time_steps, runs, network_type, network_update,
        er_prob=None, ba_links=None, hk_prob=None, objects=elg.Agent.default_objects,
        symbols=elg.Agent.default_symbols, sample_num=1) :
        self.__pop_size = pop_size
        self.__n_time_steps = time_steps
        self.__n_runs = runs
        self.__n_objects = objects
        self.__n_symbols = symbols
        self.__n_learning_samples = sample_num

        # Input validation for network type and update strategy
        if network_type in self.__class__.network_types :
            self.__network_type = network_type
            if network_type == 'random' and er_prob is None :
                raise TypeError("For random networks, the er_prob argument should be passed.")
            elif network_type == 'scale-free' and ba_links is None :
                raise TypeError("For scale-free networks, the ba_links argument should be passed.")
            elif network_type == 'clustered' and (ba_links is None or hk_prob is None) :
                raise TypeError("For clustered networks, both the ba_links and hk_prob arguments should be passed.")
        else :
            raise ValueError("Network type not recognized.")

        self.__er_prob = er_prob
        self.__ba_links = ba_links
        self.__hk_prob = hk_prob

        if network_update in self.__class__.network_updates :
            self.__network_update = network_update
        else :
            raise ValueError("Network update strategy is not recognized.")

    def run(self) :
        """
        Executes the simulation, records the results, and displays them through `pyplot`.
        """
        run_avg_payoffs = []    # Contains the average payoffs for each run
        for i_run in range(self.__n_runs) :
            # Generate agents in first generation (with random matrices)
            first_gen = {agent_id : elg.Agent(agent_id, self.__n_objects, self.__n_symbols) for agent_id in range(self.__pop_size)}
            for k, v in first_gen.items() :
                v.update_language(elg.random_assoc_matrix(self.__n_objects, self.__n_symbols))
            
            # Generate network and embed first generation
            G = nx.relabel_nodes(self.generate_network(), first_gen)

            step_avg_payoffs = []   # Contains the average payoffs for each time step
            for step_num in range(self.__n_time_steps) :
                # Simulate communication and reproduction
                G, average_payoff = self.next_generation(G)
                step_avg_payoffs.append(average_payoff)
            run_avg_payoffs.append(step_avg_payoffs)

        self.__network = G
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
        total_payoffs = []
        # Individual payoffs of single communication between every two agents 
        individual_payoffs = []

        # TODO: change variable names (speaker/listener)
        for speaker in agents :
            agent_total_payoff = 0
            list_connections = list(nx.neighbors(G, speaker))
            for listener in list_connections :
                payoff = elg.payoff(speaker, listener)
                agent_total_payoff += payoff
                individual_payoffs.append(payoff)
            total_payoffs.append(agent_total_payoff)

        # Generate list of normalized fitness scores
        sum_payoffs = sum(total_payoffs)
        normalized_payoffs = [x / sum_payoffs if sum_payoffs else 0 for x in total_payoffs]

        if self.__network_update == 'regenerate' :
            # Create new generation (of the same size)
            new_generation = []
            for n in range(len(agents)) :
                # Pick parent proportional to fitness
                try :
                    parent = agents[util.pick_item(normalized_payoffs)]
                except AssertionError as err :
                    print(err)
                    raise
                except ValueError as err :
                    # TODO: Find a better way to report error
                    print(err, "Proceeding with old generation.")
                    new_generation = agents
                    break

                # Create child that samples A from parent
                child = elg.Agent(n, self.__n_objects, self.__n_symbols)
                child.update_language(elg.sample(parent, self.__n_learning_samples))

                new_generation.append(child)
            # Generate new network and embed new generation
            new_G = nx.relabel_nodes(self.generate_network(), {idx:agent for idx, agent in enumerate(new_generation)})
        elif self.__network_update == 'relabel' :
            # Pick parent proportional to fitness
            try :
                parent = agents[util.pick_item(normalized_payoffs)]
            except AssertionError as err :
                print(err)
                return

            # Create child that samples A from parent
            child = elg.Agent(parent.get_id(), self.__n_objects, self.__n_symbols)
            child.update_language(elg.sample(parent, self.__n_learning_samples))

            # Pick random neighbour of parent to replace
            parent_neighbors = list(nx.neighbors(G, parent))
            neighbor = np.random.choice(parent_neighbors)

            # Generate new network by replacing neighbor with child
            new_G = nx.relabel_nodes(G, {neighbor:child})

        average_payoff = np.mean(individual_payoffs) if individual_payoffs else None
        # Return new network and the average payoff of single communication
        return new_G, average_payoff

    def generate_network(self) :
        """
        Generate a network based on the `network_type` property.
        """
        if self.__network_type == 'regular' :
            G = nx.complete_graph(self.__pop_size)
        elif self.__network_type == 'random' :
            G = nx.erdos_renyi_graph(self.__pop_size, self.__er_prob)
        elif self.__network_type == 'scale-free' :
            G = nx.barabasi_albert_graph(self.__pop_size, self.__ba_links)
        elif self.__network_type == 'clustered' :
            G = nx.powerlaw_cluster_graph(self.__pop_size, self.__ba_links, self.__hk_prob)

        return G

    def as_series(self, payoffs=True, network=True) :
        return pd.Series(self.as_dict(payoffs, network))

    def as_dict(self, payoffs=True, network=True) :
        sim_dict = self.get_params()
        if payoffs :
            sim_dict.update({'payoffs':self.get_avg_payoffs()})
        if network :
            sim_dict.update({'network_dict': nx.convert.to_dict_of_dicts(self.get_network_view())})

        return sim_dict

    def get_params(self) :
        params = {
            'pop_size': self.__pop_size,
            'time_steps': self.__n_time_steps,
            'runs': self.__n_runs,
            'vocab_size': (self.__n_objects, self.__n_symbols),
            'sample_num': self.__n_learning_samples,
            'network_type': self.__network_type,
            'network_update': self.__network_update,
            'er_prob': self.__er_prob,
            'ba_links': self.__ba_links,
            'hk_prob': self.__hk_prob
        }

        return params

    def get_network_view(self) :
        """
        Get a read-only copy of the network. Can be used for plotting the
        network or analyzing its properties.
        """
        # TODO: validate that network attribute exists
        return self.__network.copy(as_view=True)

    def get_avg_payoffs(self) :
        return self.__run_avg_payoffs

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