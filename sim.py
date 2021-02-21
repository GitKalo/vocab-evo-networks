import networkx as nx
import matplotlib.pyplot as plt

import elg

agent = elg.Agent(1, 3, 2)

agent.talk('Hello')

for r in agent.P_matrix :
    print(r)

for r in agent.Q_matrix :
    print(r)