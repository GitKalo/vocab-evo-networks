import numpy as np

class Agent :
    N_objects = 5
    N_symbols = 5

    def __init__(self, id, n_objects=N_objects, n_symbols=N_symbols) :
        self.id = id
        # TODO: Random initial matrix generation (for first generation)
        self.P_matrix = [[round(1 / n_symbols, 2)] * n_symbols] * n_objects
        self.Q_matrix = [[round(1 / n_objects, 2)] * n_objects] * n_symbols
        self.A_matrix = [[0] * n_symbols] * n_objects

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
            print("NO! Bad list!")  # TODO: Change error handline message (and add return value?)

    def __str__(self) :
        return 'a[' + str(self.id) + ']'

def payoff(agent_1, agent_2) :
    return 0.5 * sum(
        [sum(
            agent_1.P_matrix[i][j] * agent_2.Q_matrix[j][i] + agent_2.P_matrix[i][j] * agent_1.Q_matrix[j][i] for j in range(Agent.N_symbols)
            ) for i in range(Agent.N_objects)]
        )

def sample(agent_1, agent_2, k) :
    
    # agent 1 constructs association matrix by sampling k responses from agent 2

    return None

def random_matrix(n_rows, n_cols) :

    # generate a random matrix of given size whose rows sum to 1

    return None