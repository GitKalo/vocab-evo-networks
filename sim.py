import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import elg, util

class Simulation :
    def __init__(self, pop_size=100, time_steps=100, runs=20, 
        objects=elg.Agent.default_objects, symbols=elg.Agent.default_symbols):
        self.__pop_size = pop_size
        self.__n_time_steps = time_steps
        self.__n_runs = runs
        self.__n_objects = objects
        self.__n_symbols = symbols
        self.__n_learning_samples = 1

    def run(self) :
        run_avg_payoffs = []
        for i_run in range(self.__n_runs) :
            first_gen = {agent_id : elg.Agent(agent_id, self.__n_objects, self.__n_symbols) for agent_id in range(self.__pop_size)}
            for k, v in first_gen.items() : v.update_language(elg.random_assoc_matrix(self.__n_objects, self.__n_symbols))
            G = self.generate_graph()
            nx.relabel_nodes(G, first_gen, copy=False)

            step_avg_payoffs = []
            for step_num in range(self.__n_time_steps) :
                G, average_payoff = self.next_generation(G)
                step_avg_payoffs.append(average_payoff)
            run_avg_payoffs.append(step_avg_payoffs)

        fig, ax = plt.subplots()
        for payoff in run_avg_payoffs :
            ax.plot(payoff, color='blue')
        ax.set_xlabel('Time')
        ax.set_ylabel('Payoff')
        ax.set_title('Parental learning, k = 5')

        plt.show()

    def next_generation(self, G) :
        agents = list(G.nodes)
        total_payoffs = []
        individual_payoffs = []
        
        for speaker in agents :
            agent_total_payoff = 0
            list_connections = list(nx.neighbors(G, speaker))
            for listener in list_connections :
                payoff = elg.payoff(speaker, listener)
                agent_total_payoff += payoff
                individual_payoffs.append(payoff)
            total_payoffs.append(agent_total_payoff)

        # generate list of normalized fitness scores
        sum_payoffs = sum(total_payoffs)
        normalized_payoffs = [x / sum_payoffs for x in total_payoffs]

        new_agents = []
        for n in range(len(agents)) :
            # pick agent proportional to fitness
            parent = agents[util.pick_item(normalized_payoffs)]

            # create child that samples A from parent
            child = elg.Agent(n, self.__n_objects, self.__n_symbols)
            child.update_language(elg.sample(parent, self.__n_learning_samples))

            new_agents.append(child)

        # pick random agent and replace with new one on graph
        new_G = nx.complete_graph(new_agents)

        # return new graph
        return new_G, np.mean(individual_payoffs)

    def generate_graph(self, type='regular') :
        return nx.complete_graph(self.__pop_size)

if __name__ == '__main__' :
    pop_size = 10
    n_time_steps = 100
    n_runs = 20

    simulation = Simulation(pop_size, n_time_steps, n_runs)

    simulation.run()