import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from . import elg, util

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

    def __init__(self, pop_size=100, time_steps=100, runs=20, 
        objects=elg.Agent.default_objects, symbols=elg.Agent.default_symbols, network_type='regular', network_update='regenerate'):
        self.__pop_size = pop_size
        self.__n_time_steps = time_steps
        self.__n_runs = runs
        self.__n_objects = objects
        self.__n_symbols = symbols
        self.__n_learning_samples = 1

        # Input validation for network type and update strategy
        if network_type in self.__class__.network_types :
            self.__network_type = network_type
        else :
            raise ValueError("Network type not recognized.")

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

        # Plot and display the average payoff
        fig, ax = plt.subplots()
        for payoff in run_avg_payoffs :
            ax.plot(payoff, color='blue')
        ax.set_xlabel('Time')
        ax.set_ylabel('Payoff')
        ax.set_title('Parental learning, k = ' + str(self.__n_learning_samples))

        plt.show()

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
        normalized_payoffs = [x / sum_payoffs for x in total_payoffs]

        if self.__network_update == 'regenerate' :
            # Create new generation (of the same size)
            new_generation = []
            for n in range(len(agents)) :
                # Pick parent proportional to fitness
                try :
                    parent = agents[util.pick_item(normalized_payoffs)]
                except AssertionError as err :
                    print(err)
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

        # Return new network and the average payoff of single communication
        return new_G, np.mean(individual_payoffs)

    def generate_network(self) :
        """
        Generate a network based on the `network_type` property.
        """
        if self.__network_type == 'regular' :
            G = nx.complete_graph(self.__pop_size)
        elif self.__network_type == 'random' :
            G = nx.erdos_renyi_graph(self.__pop_size, 0.5)
        elif self.__network_type == 'scale-free' :
            G = nx.barabasi_albert_graph(self.__pop_size, 2)
        elif self.__network_type == 'clustered' :
            G = nx.powerlaw_cluster_graph(self.__pop_size, 2, 0.5)

        return G

    def get_network_view(self) :
        """
        Get a read-only copy of the network. Can be used for plotting the
        network or analyzing its properties.
        """
        # TODO: validate that network attribute exists
        return self.__network.copy(as_view=True)

    # TODO: implement copy method

if __name__ == '__main__' :
    pop_size = 10
    n_time_steps = 100
    n_runs = 20

    simulation = Simulation(pop_size, n_time_steps, n_runs)

    simulation.run()