import numpy as np
import random
import tflowtools as tft


def bit_counter():
    number_of_cases = 500
    number_of_bits = 15
    return tft.gen_vector_count_cases(number_of_cases, number_of_bits)


def segment_counter():
    number_of_bits = 25
    number_of_cases = 1000
    min_segments = 0
    max_segments = 8
    return tft.gen_segmented_vector_cases(number_of_bits, number_of_cases, min_segments, max_segments)


def autoencoder_one_hot():
    number_of_bits = 8
    return tft.gen_all_one_hot_cases(number_of_bits)


def autoencoder_dense():
    number_of_bits = 8
    number_of_cases = 200
    lower_density = 0.0
    upper_density = 1.0
    return tft.gen_dense_autoencoder_cases(number_of_cases, number_of_bits, (lower_density, upper_density))


def symmetry():
    number_of_bits = 100
    number_of_cases = 2000
    vectors = tft.gen_symvect_dataset(number_of_bits, number_of_cases)

    return [(x[:-1], [1, 0]) if x[number_of_bits] == 0 else (x[:-1], [0, 1]) for x in vectors]


def parity():
    number_of_bits = 10
    return tft.gen_all_parity_cases(number_of_bits)


def mnist():
    print('Loading mnist dataset...')
    with open('data/mnist/all_flat_mnist_training_cases_text.txt') as file:
        raw_data = [[int(x) for x in line.rstrip().split(' ')] for line in file.readlines()]

    print('Processing cases')
    cases = []
    for index, case in enumerate(raw_data[1:]):
        label = [0]*10
        label[raw_data[0][index]] = 1
        cases.append((case, label))

    return cases


def load_file(filename, separator, normalize=True):
    with open(filename) as file:
        raw_data = [[float(x) for x in line.rstrip().split(separator)] for line in file.readlines()]

    cases = []
    labels = []
    max_label = max(int(case[-1]) for case in raw_data) + 1
    for case in raw_data:
        label = int(case[-1])
        label_array = [0]*max_label
        label_array[label - 1] = 1
        labels.append(label_array)
        cases.append(case[:len(case) - 1])

    if normalize:
        features = np.array(cases).transpose()
        means = np.mean(features, axis=1)
        standard_deviations = np.std(features, axis=1)
        for i, row in enumerate(cases):
            for j, col in enumerate(cases[i]):
                cases[i][j] = (cases[i][j] - means[j]) / standard_deviations[j]

    data_set = [(case, labels[i]) for i, case in enumerate(cases)]
    return data_set


def wine():
    print('Loading wine dataset...')
    return load_file(filename='data/wine.txt', separator=';', normalize=True)


def glass():
    print('Loading glass dataset...')
    return load_file(filename='data/glass.txt', separator=',', normalize=True)


def yeast():
    print('Loading yeast dataset...')
    return load_file(filename='data/yeast.txt', separator=',', normalize=True)


def poker():
    print('Loading poker dataset...')
    return load_file(filename='data/poker/poker-hand-training-true.data', separator=',', normalize=True)


def iris():
    return load_file(filename='data/iris.txt', separator=',', normalize=True)


def get_cases_from_source(data_source):
    if data_source == 'parity':
        return parity()
    elif data_source == 'bit_counter':
        return bit_counter()
    elif data_source == 'segment_counter':
        return segment_counter()
    elif data_source == 'symmetry':
        return symmetry()
    elif data_source == 'autoencoder_one_hot':
        return autoencoder_one_hot()
    elif data_source == 'autoencoder_dense':
        return autoencoder_dense()
    elif data_source == 'mnist':
        return mnist()
    elif data_source == 'wine':
        return wine()
    elif data_source == 'glass':
        return glass()
    elif data_source == 'yeast':
        return yeast()
    elif data_source == 'poker':
        return poker()
    elif data_source == 'iris':
        return iris()
    else:
        pass


class Cases:
    def __init__(self, data_source, case_fraction, validation_fraction, test_fraction):
        self._cases = get_cases_from_source(data_source)

        array_of_cases = np.array(self._cases)
        np.random.shuffle(array_of_cases)
        self._cases = array_of_cases
        if 0.0 < case_fraction < 1.0:
            self._cases = self._cases[:round(self._number_of_cases * case_fraction)]
        self._case_fraction = case_fraction
        self._validation_fraction = validation_fraction
        self._test_fraction = test_fraction
        (self.training_cases,
         self.validation_cases,
         self.test_cases) = self.organize_cases()

    def organize_cases(self):
        # Split cases into training, validation and testing
        training_cases_end_index = round(
            (1 - (self._validation_fraction + self._test_fraction)) * self._number_of_cases
        )
        validation_cases_end_index = round(training_cases_end_index + self._validation_fraction * self._number_of_cases)
        training_cases = self._cases[: training_cases_end_index]
        validation_cases = self._cases[training_cases_end_index: validation_cases_end_index]
        test_cases = self._cases[validation_cases_end_index:]

        return training_cases, validation_cases, test_cases

    def get_minibatch_of_size(self, size):
        number_of_cases = min(size, self.number_of_training_cases)
        return random.sample(list(self.training_cases), number_of_cases)

    @property
    def number_of_training_cases(self):
        return len(self.training_cases)

    @property
    def number_of_validation_cases(self):
        return len(self.validation_cases)

    @property
    def number_of_test_cases(self):
        return len(self.test_cases)

    @property
    def _number_of_cases(self):
        return len(self._cases)

    def get_input_size(self):
        if self._number_of_cases:
            case = self._cases[0]
            return len(case[0])
        else:
            raise Exception('No cases found, can\'t calculate input size.')

    def get_output_size(self):
        if self._number_of_cases:
            case = self._cases[0]
            return len(case[1])
        else:
            raise Exception('No cases found, can\'t calculate output size.')
