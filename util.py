import numpy as np

import numbers

def pick_item(list_prob) :
    if not (isinstance(list_prob, list) or isinstance(list_prob, np.ndarray)) :
        raise TypeError('Argument must be a list (or numpy array).')
    elif np.array(list_prob).size == 0 :
        raise ValueError('Argument list must not be empty.')
    elif not all(isinstance(x, numbers.Number) for x in list_prob) :
        raise ValueError('Argument list must contain only numbers.')

    sum_prob = np.sum(list_prob)
    assert (sum_prob == 1) or (sum_prob > 0.99 and round(sum_prob) == 1), \
        "Probability distribution must be strict (items in list should sum up to 1)"
    a = np.random.random_sample(1)
    count, acc = 0, 0

    for e in list_prob :
        acc += e
        if a <= acc :
            return count
        count += 1

# taken from https://stackoverflow.com/questions/13214809/pretty-print-2d-python-list
def pp_matrix(matrix) :
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))