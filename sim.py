import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import elg, util

class Simulation :
    def __init__(self, pop_size=100, time_steps=100, runs=20):
        self.__pop_size = pop_size
        self.__n_time_steps = time_steps
        self.__n_runs = runs
        self.__n_learning_samples = 1

    def run(self) :
        run_average_payoffs = []
        for i_run in range(self.__n_runs) :
            first_gen = []
            for n in range(self.__pop_size) :
                a = elg.Agent(n)
                a.update_language(elg.random_assoc_matrix(5, 5))
                first_gen.append(a)

            G = nx.complete_graph(first_gen)

            average_payoffs = []
            for step_num in range(self.__n_time_steps) :
                G, payoffs = self.next_generation(G)
                average_payoffs.append(np.mean(payoffs))
            run_average_payoffs.append(average_payoffs)

        fig, ax = plt.subplots()
        for run_payoffs in run_average_payoffs :
            ax.plot(run_payoffs, color='blue')
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
        normalized_payoffs = list(map(lambda x : x / sum_payoffs, total_payoffs))

        new_agents = []
        for n in range(len(agents)) :
            # pick agent proportional to fitness
            parent = agents[util.pick_item(normalized_payoffs)]

            # create child that samples A from parent
            max_id = max([a.get_id() for a in agents])
            child = elg.Agent(max_id + 1)
            child.update_language(elg.sample(parent, self.__n_learning_samples))

            new_agents.append(child)

        # pick random agent and replace with new one on graph
        new_G = nx.complete_graph(new_agents)

        # return new graph
        return new_G, individual_payoffs

if __name__ == '__main__' :
    pop_size = 10
    n_time_steps = 100
    n_runs = 20

    simulation = Simulation(pop_size, n_time_steps, n_runs)

    simulation.run()