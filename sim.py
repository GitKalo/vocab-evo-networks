import networkx as nx
import matplotlib.pyplot as plt

import random

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

pop_size = len(agent_list)
n_time_steps = 10

for step_num in range(n_time_steps) :
    speaker = random.choice(list(G.nodes))
    list_connections = list(nx.neighbors(G, speaker))
    listener = random.choice(list_connections)
    print("\nTime step ", step_num)
    speaker.speak(listener)


plt.show()