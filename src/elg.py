import numpy as np

from . import util

class Agent :
    default_objects = 5
    default_symbols = 5

    def __init__(self, agent_id, n_objects=default_objects, n_symbols=default_symbols) :
        self.__id = agent_id
        self.__n_objects = n_objects
        self.__n_symbols = n_symbols
        self.active_matrix = [[0] * n_symbols] * n_objects
        self.passive_matrix = [[0] * n_objects] * n_symbols
        self.assoc_matrix = [[0] * n_symbols] * n_objects

    def update_language(self, assoc_matrix) :
        if assoc_matrix is None :
            self.set_assoc_matrix([])
        else :
            self.set_assoc_matrix(assoc_matrix)
        self.update_active_matrix()
        self.update_passive_matrix()

    def update_active_matrix(self) :
        self.active_matrix = np.zeros(np.shape(self.assoc_matrix))

        for i in range(len(self.active_matrix)) :
            row_sum = sum(self.assoc_matrix[i])
            for j in range(len(self.active_matrix[i])) :
                self.active_matrix[i][j] = (self.assoc_matrix[i][j] / row_sum) if row_sum != 0 else 0

    def update_passive_matrix(self) :
        self.passive_matrix = np.zeros(tuple(reversed(np.shape(self.assoc_matrix))))

        for j in range(len(self.passive_matrix)) :
            col_sums = np.sum(self.assoc_matrix, axis=0)
            for i in range(len(self.passive_matrix[j])) :
                self.passive_matrix[j][i] = (self.assoc_matrix[i][j] / col_sums[j]) if col_sums[j] != 0 else 0

    def set_assoc_matrix(self, new_assoc_matrix) :
        if len(new_assoc_matrix) == 0 :
            self.__n_objects = 0
            self.__n_symbols = 0
        else :
            self.__n_objects, self.__n_symbols = np.shape(new_assoc_matrix)
            if not self.__n_symbols :
                raise ValueError("Agents need at least one symbol.")
        
        self.assoc_matrix = new_assoc_matrix

    def set_active_matrix(self, new_active_matrix) :
        self.active_matrix = new_active_matrix

    def set_passive_matrix(self, new_passive_matrix) :
        self.passive_matrix = new_passive_matrix

    def get_n_obj(self) :
        return self.__n_objects

    def get_n_sym(self) :
        return self.__n_symbols

    def get_id(self) :
        return self.__id

    def __str__(self) :
        return 'a[' + str(self.__id) + ']'

def payoff(a1, a2) :
    if np.shape(a1.assoc_matrix) != np.shape(a2.assoc_matrix) :
        raise ValueError("Payoff of communication can only be calculated for agents with the same number of objects/symbols.")

    return 0.5 * sum(
        [sum(
            [a1.active_matrix[i][j] * a2.passive_matrix[j][i] + 
                a2.active_matrix[i][j] * a1.passive_matrix[j][i] for j in range(a1.get_n_sym())]
            ) for i in range(a1.get_n_obj())]
        )

def sample(agent, k) :
    assoc = np.zeros(np.shape(agent.assoc_matrix))

    for obj in range(len(agent.active_matrix)) :
        for _ in range(k) :
            try :
                response = util.pick_item(agent.active_matrix[obj])
            except AssertionError as err :
                print(err)
                return None
            assoc[obj][response] += 1

    return assoc

def random_assoc_matrix(n_rows, m_cols) :
    return np.random.randint(1, 10, size=(n_rows, m_cols))