import numpy as np

import util

class Agent :
    N_objects = 5
    N_symbols = 5

    def __init__(self, id, n_objects=N_objects, n_symbols=N_symbols) :
        self.id = id
        self.active_matrix = [[0] * n_symbols] * n_objects
        self.passive_matrix = [[0] * n_objects] * n_symbols
        self.assoc_matrix = [[0] * n_symbols] * n_objects

    def speak(self, listener) :
        print("Agent", self.id, "speaking to agent", listener.id)
        obj = np.random.randint(len(self.active_matrix))
        sym = util.pick_item(self.active_matrix[obj])
        print("Agent", self.id, "is using symbol", sym, "to talk about object", obj)

    def update_active_matrix(self) :
        self.active_matrix = np.zeros(np.shape(self.assoc_matrix))

        for i in range(len(self.active_matrix)) :
            row_sum = sum(self.assoc_matrix[i])
            for j in range(len(self.active_matrix[i])) :
                self.active_matrix[i][j] = self.assoc_matrix[i][j] / row_sum

    def update_passive_matrix(self) :
        self.passive_matrix = np.zeros(tuple(reversed(np.shape(self.assoc_matrix))))

        for j in range(len(self.passive_matrix)) :
            col_sums = np.sum(self.assoc_matrix, axis=0)
            for i in range(len(self.passive_matrix[j])) :
                self.passive_matrix[j][i] = (self.assoc_matrix[i][j] / col_sums[j]) if col_sums[j] != 0 else 0

    def set_active_matrix(self, new_active_matrix) :
        self.active_matrix = new_active_matrix

    def set_passive_matrix(self, new_passive_matrix) :
        self.passive_matrix = new_passive_matrix

    def set_assoc_matrix(self, new_assoc_matrix) :
        self.assoc_matrix = new_assoc_matrix

    def __str__(self) :
        return 'a[' + str(self.id) + ']'

def payoff(agent_1, agent_2) :
    # TODO: add check for correct shape of agents' matrices
    return 0.5 * sum(
        [sum(
            agent_1.active_matrix[i][j] * agent_2.passive_matrix[j][i] + agent_2.active_matrix[i][j] * agent_1.passive_matrix[j][i] for j in range(len(agent_1.assoc_matrix))
            ) for i in range(len(agent_1.assoc_matrix[0]))]
        )

def sample(agent, k) :
    assoc = np.zeros(np.shape(agent.assoc_matrix))

    for obj in range(len(agent.active_matrix)) :
        for _ in range(k) :
            response = util.pick_item(agent.active_matrix[obj])
            assoc[obj][response] += 1

    return assoc

def random_matrix(n_rows, n_cols) :

    # generate a random matrix of given size whose rows sum to 1

    return None

def random_assoc_matrix(n_rows, m_cols) :
    return np.random.randint(1, 10, size=(n_rows, m_cols))