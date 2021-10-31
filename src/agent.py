import numpy as np

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

    def __init__(self, agent_id, n_objects=default_objects, n_signals=default_signals) :
        self.__id = agent_id
        self.__n_objects = n_objects
        self.__n_signals = n_signals
        self.update_language(np.zeros((n_objects, n_signals)))

    def update_language(self, assoc_matrix) :
        """
        Update the agent's association matrix and active/passive matrices 
        (derived from the former).
        """
        if assoc_matrix is None :
            assoc_matrix = np.array([])
        elif not isinstance(assoc_matrix, np.ndarray) :
            assoc_matrix = np.array(assoc_matrix)
        elif assoc_matrix.ndim != 2 :
            raise ValueError("Agent language must be represented by a 2D matrix")

        self._set_assoc_matrix(assoc_matrix)
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
        try :
            self.__n_objects = new_assoc_matrix.shape[0]            
            self.__n_signals = new_assoc_matrix.shape[1]
        except IndexError :
            self.__n_signals = 0
                    
        self.assoc_matrix = new_assoc_matrix

    def get_n_objects(self) :
        return self.__n_objects

    def get_n_signals(self) :
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
    try :
        payoff = 0.5 * np.sum(np.diagonal(np.matmul(a1.active_matrix, a2.passive_matrix)) + np.diagonal(np.matmul(a2.active_matrix, a1.passive_matrix)))
    except ValueError :
        print("Payoff of communication can only be calculated for agents with the same number of objects/signals.")
        raise

    return payoff

def sample(agent, k, p_mistake=0) :
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
    n_signals = agent.active_matrix[0].size

    for obj in range(len(agent.active_matrix)) :
        for _ in range(k) :
            try :
                # Sample response
                if p_mistake and np.random.binomial(1, p_mistake) :
                    response = np.random.choice(n_signals)
                else :
                    response = np.random.choice(n_signals, p=agent.active_matrix[obj])
            except ValueError as err :
                print(err)
                break
            # Record response
            assoc[obj][response] += 1

    return assoc

def sample_from_matrix(active_matrix, k, p_mistake=0) :

    if isinstance(active_matrix, np.ndarray) :
        n_objects, n_signals = active_matrix.shape
    else :
        n_objects = len(active_matrix)
        n_signals = len(active_matrix[0])

    assoc = np.zeros((n_objects, n_signals))

    for obj in range(n_objects) :
        for _ in range(k) :
            try :
                # Sample response
                if p_mistake and np.random.binomial(1, p_mistake) :
                    response = np.random.choice(n_signals)
                else :
                    response = np.random.choice(n_signals, p=active_matrix[obj])
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