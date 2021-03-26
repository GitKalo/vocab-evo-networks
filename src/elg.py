import numpy as np

from . import util

class Agent :
    """
    An individual of the popualtion in the vocabulary evolution model.
    
    Each agent has an independent language composed of three matrices:
        - Active matrix -- Probabilities of object-symbol associations
        - Passive matrix -- Probabilities of symbol-object associations
        - Association matrix -- A record of responses to object-symbol associations,
            sampled depending on the simulation instance sampling method (or
            random if agent is in from the first generation).

    The active and passive matrices are derived from the association matrix by
    normalizing the number of responses for each object-symbol pair.

    The number of objects and symbols in an agent's vocabulary, which determines the size
    of the matrices described above, can be provided through the `n_objects` and `n_symbols`
    at initialization. If not provided, the values of the `default_objects` and `default_symbols`
    attributes are used instead.
    """
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
        """
        Update the agent's association matrix and active/passive matrices 
        (derived from the former).
        """
        if (assoc_matrix is None) or (len(assoc_matrix) == 0) :
            self._set_assoc_matrix(np.array([[]]))
        else :
            self._set_assoc_matrix(np.array(assoc_matrix))
        self.update_active_matrix()
        self.update_passive_matrix()

    def update_active_matrix(self) :
        """
        Derive and update the agent's active matrix from its association matrix.
        """
        self.active_matrix = np.zeros(np.shape(self.assoc_matrix))

        row_sums = np.sum(self.assoc_matrix, axis=1)
        for i in range(len(self.active_matrix)) :
            self.active_matrix[i] = self.assoc_matrix[i] / row_sums[i] if row_sums[i] != 0 else np.zeros(len(self.assoc_matrix[i]))

    def update_passive_matrix(self) :
        """
        Derive and update the agent's passive matrix from its association matrix. 
        """
        self.passive_matrix = np.zeros(tuple(reversed(np.shape(self.assoc_matrix))))

        col_sums = np.sum(self.assoc_matrix, axis=0)
        for i in range(len(self.passive_matrix)) :
            self.passive_matrix[i] = self.assoc_matrix[:, i] / col_sums[i] if col_sums[i] != 0 else np.zeros(len(self.assoc_matrix))

    def _set_assoc_matrix(self, new_assoc_matrix) :
        """
        Set the association matrix of the agent, updating the attributes containing
        the number of objects and symbols in the process.
        
        Agents with a non-empty association matrix (can communicate about at least one object)
        must also be able to communicate about at least one symbol.
        """
        self.__n_objects, self.__n_symbols = np.shape(new_assoc_matrix)
        
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
    """
    Calculate the total payoff of communication between two agents. The agents' matrices
    must have the same shape (they have to be communicating about the same number of objects
    and symbols).

    Intuitively, the payoff of communication represents how well two agents can understand 
    each other when conversing. Practically, it is calculated from the probability of
    one agent emitting a certain symbol to talk about a certain object, multiplied by the probability
    of the other agent inferring the initial object upon hearing the symbol, summed over all
    objects and symbols, for each agent.
    """
    if np.shape(a1.assoc_matrix) != np.shape(a2.assoc_matrix) :
        raise ValueError("Payoff of communication can only be calculated for agents with the same number of objects/symbols.")

    # return 0.5 * np.sum(
    #     [np.sum(
    #         [a1.active_matrix[i][j] * a2.passive_matrix[j][i] + 
    #             a2.active_matrix[i][j] * a1.passive_matrix[j][i] for j in range(a1.get_n_sym())]
    #         ) for i in range(a1.get_n_obj())]
    #     )

    return 0.5 * np.sum(np.diagonal(np.matmul(a1.active_matrix, a2.passive_matrix)) + np.diagonal(np.matmul(a2.active_matrix, a1.passive_matrix)))

def sample(agent, k) :
    """
    Construct an association matrix by sampling responses from `agent`. For
    each of `agent`'s objects, `k` responses are sampled by emitting a
    symbol based on the emission probabilities specified by `agent`'s
    active matrix.

    The values in the association matrix represents the number of time that
    a symbol was emitted in reference to an object. Its size is the identical
    to that of `agent`'s active matrix.
    """
    assoc = np.zeros(np.shape(agent.assoc_matrix))

    for obj in range(len(agent.active_matrix)) :
        for _ in range(k) :
            try :
                # Sample response
                response = util.pick_item(agent.active_matrix[obj])
            except AssertionError as err :
                print(err)
                return None
            except ValueError as err :
                print(err)
                break
            # Record response
            assoc[obj][response] += 1

    return assoc

def random_assoc_matrix(n_rows, m_cols) :
    """
    Generate a random matrix of size `n_rows` by `m_cols`.
    Each entry in the matrix is a random number in the interval [0, 10).
    
    One use is for generating the languages of agents in the first generation. 
    """
    return np.random.randint(1, 10, size=(n_rows, m_cols))  #TODO: why larger than o?