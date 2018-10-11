import numpy as np
import random
import tensorflow as tf


def get_distribution(settings, size):
    lower, upper = settings.range
    if settings.distribution == 'uniform':
        return np.random.uniform(lower, upper, size)
    elif settings.distribution == 'normal':
        mean = (upper + lower) / 2
        return np.random.normal(mean, settings.standard_deviation, size)


def set_random_seed(seed):
    random.randrange(1000)
    # print(seed)
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

