import numpy as np

def pick_item(list_prob) :
    try :
        sum_prob = np.sum(list_prob)
        assert (sum_prob == 1) or (sum_prob > 0.99 and round(sum_prob) == 1) # TODO: Improve validation
        a = np.random.random_sample(1)
        count, acc = 0, 0

        for e in list_prob :
            acc += e
            if a <= acc :
                return count
            count += 1
    except AssertionError :
        print("Assertion error -- probability distribution must be strict (items in list should sum up to 1)")
        return -1