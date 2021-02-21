class Agent :
    N_objects = 5
    N_symbols = 5

    def __init__(self, id, n_objects=N_objects, n_symbols=N_symbols) :
        self.id = id
        self.P_matrix = [[round(1 / n_symbols, 2)] * n_symbols] * n_objects
        self.Q_matrix = [[round(1 / n_objects, 2)] * n_objects] * n_symbols

    def speak(self, listener) :
        print("Agent", self.id, "speaking to agent", listener.id)
        obj = Agent.pick_list(self.P_matrix)
        sym = Agent.pick_item(self.P_matrix[obj])
        print("Agent", self.id, "is using symbol", sym, "to talk about object", obj)

    @staticmethod
    def pick_list(lists) :
        return np.random.randint(len(lists))

    @staticmethod
    def pick_item(list_p) :
        try :
            sum_p = np.sum(list_p)
            assert (sum_p == 1) or (sum_p > 0.99 and round(sum_p) == 1) # TODO: Improve validation
            a = np.random.random_sample(1)
            count, acc = 0, 0

            for e in list_p :
                acc += e
                if a <= acc :
                    return count
                count += 1
        except AssertionError :
            print("NO! Bad list!")