import networkx as nx
import matplotlib.pyplot as plt

import random

import elg

def generation( G, agents ) :
    for speaker in agents :
        agent_payoff = 0
        list_connections = list(nx.neighbors(G, speaker))
        for listener in list_connections :
            agent_payoff += elg.payoff(speaker, listener)

        print("Agent", speaker.id, "has payoff", agent_payoff)

    # reproduction

    # return new graph

if __name__ == '__main__' :
    pop_size = 5

    agent_list = []
    for n in range(pop_size) :
        a = elg.Agent(n)
        a.assoc_matrix = elg.random_assoc_matrix(3, 3)
        a.update_active_matrix()
        a.update_passive_matrix()

        agent_list.append(a)

    G = nx.complete_graph(agent_list)

    nx.draw_circular(G, with_labels=True, font_weight='bold')

    n_time_steps = 10

    for step_num in range(n_time_steps) :
        print("\nTime step ", step_num)
        generation(G, agent_list)

    plt.show()