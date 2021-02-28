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

# taken from https://stackoverflow.com/questions/13214809/pretty-print-2d-python-list
def pp_matrix(matrix) :
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

if __name__ == '__main__' :
    pop_size = 5

    agent_list = [elg.Agent(n) for n in range(pop_size)]

    G = nx.complete_graph(agent_list)

    nx.draw_circular(G, with_labels=True, font_weight='bold')

    n_time_steps = 10

    for step_num in range(n_time_steps) :
        print("\nTime step ", step_num)
        generation(G, agent_list)

    plt.show()