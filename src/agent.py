import numpy as np

from . import util

class Agent :
    """
    An individual of the popualtion in the vocabulary evolution model.
    
    Each agent has an independent language composed of three matrices:
        - Active matrix -- Probabilities of object-signal associations
        - Passive matrix -- Probabilities of signal-object associations
        - Association matrix -- A record of responses to object-signal associations,
            sampled depending on the simulation instance sampling method (or
            random if agent is in from the first generation).

    The active and passive matrices are derived from the association matrix by
    normalizing the number of responses for each object-signal pair.

    The number of objects and signals in an agent's vocabulary, which determines the size
    of the matrices described above, can be provided through the `n_objects` and `n_signals`
    at initialization. If not provided, the values of the `default_objects` and `default_signals`
    attributes are used instead.
    """
    default_objects = 5
    default_signals = 5

    def __init__(self, agent_id, n_objects=default_objects, n_signals=default_signals, p_mistake=0) :
        self.__id = agent_id
        self.__n_objects = n_objects
        self.__n_signals = n_signals
        self.__p_mistake = p_mistake
        self.active_matrix = [[0] * n_signals] * n_objects
        self.passive_matrix = [[0] * n_objects] * n_signals
        self.assoc_matrix = [[0] * n_signals] * n_objects

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
        the number of objects and signals in the process.
        
        Agents with a non-empty association matrix (can communicate about at least one object)
        must also be able to communicate about at least one signal.
        """
        self.__n_objects, self.__n_signals = np.shape(new_assoc_matrix)
        
        self.assoc_matrix = new_assoc_matrix

    def set_active_matrix(self, new_active_matrix) :
        self.active_matrix = new_active_matrix

    def set_passive_matrix(self, new_passive_matrix) :
        self.passive_matrix = new_passive_matrix

    def get_n_obj(self) :
        return self.__n_objects

    def get_n_sym(self) :
        return self.__n_signals

    def get_id(self) :
        return self.__id

    def __str__(self) :
        return 'a[' + str(self.__id) + ']'

def payoff(a1, a2) :
    """
    Calculate the total payoff of communication between two agents. The agents' matrices
    must have the same shape (they have to be communicating about the same number of objects
    and signals).

    Intuitively, the payoff of communication represents how well two agents can understand 
    each other when conversing. Practically, it is calculated from the probability of
    one agent emitting a certain signal to talk about a certain object, multiplied by the probability
    of the other agent inferring the initial object upon hearing the signal, summed over all
    objects and signals, for each agent.
    """
    if np.shape(a1.assoc_matrix) != np.shape(a2.assoc_matrix) :
        raise ValueError("Payoff of communication can only be calculated for agents with the same number of objects/signals.")

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
    signal based on the emission probabilities specified by `agent`'s
    active matrix.

    The values in the association matrix represents the number of time that
    a signal was emitted in reference to an object. Its size is the identical
    to that of `agent`'s active matrix.
    """
    assoc = np.zeros(np.shape(agent.assoc_matrix))

    for obj in range(len(agent.active_matrix)) :
        for _ in range(k) :
            try :
                # Sample response
                response = np.random.choice(self.__n_signals, agent.active_matrix[obj])
                if self.__p_mistake and np.random.binomial(1, self.__p_mistake) :
                    response = np.random.choice(self.__n_signals)
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