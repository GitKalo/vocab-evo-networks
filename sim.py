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

    # pick agent proportional to fitness
    parent = agents[util.pick_item(normalized_payoffs)]

    # create child that samples A from parent
    max_id = max([a.id for a in agents])
    child = elg.Agent(max_id + 1)
    child.set_assoc_matrix(elg.sample(parent, 5))
    child.update_active_matrix()
    child.update_passive_matrix()

    # pick random agent and replace with new one on graph
    old_agent = np.random.choice(G.nodes)
    new_G = nx.relabel_nodes(G, {old_agent: child})

    # return new graph
    return new_G, individual_payoffs

if __name__ == '__main__' :
    pop_size = 10
    n_time_steps = 100

    agent_list = []
    for n in range(pop_size) :
        a = elg.Agent(n)
        a.assoc_matrix = elg.random_assoc_matrix(5, 5)
        a.update_active_matrix()
        a.update_passive_matrix()

        agent_list.append(a)

    G = nx.complete_graph(agent_list)

    for step_num in range(n_time_steps) :
        # print("\nTime step ", step_num)
        G = generation(G)

    nx.draw_circular(G, with_labels=True, font_weight='bold')
    plt.show()