import networkx as nx
import matplotlib.pyplot as plt

import elg

agent_list = [
    elg.Agent(0),
    elg.Agent(1),
    elg.Agent(2),
    elg.Agent(3),
    elg.Agent(4),
    elg.Agent(5)
]

G = nx.complete_graph(agent_list)

nx.draw_circular(G, with_labels=True, font_weight='bold')

plt.show()