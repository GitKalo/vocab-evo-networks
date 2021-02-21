class Agent :
    N_objects = 5
    N_symbols = 5

    def __init__(self, id, n_objects=N_objects, n_symbols=N_symbols) :
        self.id = id
        self.P_matrix = [[round(1 / n_symbols, 2)] * n_symbols] * n_objects
        self.Q_matrix = [[round(1 / n_objects, 2)] * n_objects] * n_symbols

    def talk(self, message) :
        print(message)