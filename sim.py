import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import elg, util

def generation( G ) :
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
        max_id = max([a.id for a in agents])
        child = elg.Agent(max_id + 1)
        child.update_language(elg.sample(parent, 5))

        new_agents.append(child)

    # pick random agent and replace with new one on graph
    new_G = nx.complete_graph(new_agents)

    # return new graph
    return new_G, individual_payoffs

if __name__ == '__main__' :
    pop_size = 100
    n_time_steps = 200
    n_runs = 20

    run_average_payoffs = []
    for run_num in range(n_runs) :
        agent_list = []
        for n in range(pop_size) :
            a = elg.Agent(n)
            a.assoc_matrix = elg.random_assoc_matrix(3, 3)
            a.update_active_matrix()
            a.update_passive_matrix()

            agent_list.append(a)

        G = nx.complete_graph(agent_list)

        average_payoffs = []
        for step_num in range(n_time_steps) :
            G, payoffs = generation(G)
            average_payoffs.append(np.mean(payoffs))
        run_average_payoffs.append(average_payoffs)

    fig, ax = plt.subplots()
    for run_payoffs in run_average_payoffs :
        ax.plot(run_payoffs, color='blue')
    ax.set_xlabel('Time')
    ax.set_ylabel('Payoff')
    ax.set_title('Parental learning, k = 5')

    plt.show()