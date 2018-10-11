import argparse
from loader import load_settings
import math
from utils import set_random_seed
from plotter import Plotter
import tensorflow as tf
import tflowtools as TFT
from cases import Cases
from sessions import Session
from layers import InputLayer, DenseLayer


class Net:
    def __init__(self, settings):
        self._settings = settings
        self._layers = []
        self._input = None
        self._output = None
        self._target = None
        self._error = None
        self._optimizer = None
        self._trainer = None
        self._learning_rate = self._settings.learning_rate
        self._learning_rate_placeholder = None
        self._training_error_history = []
        self._validation_error_history = []
        self._cases = self._generate_cases()
        self._build()
        self._setup_learning()
        self._session = Session(use_tensorboard=False)
        self._plotter = Plotter()

    def _generate_cases(self):
        return Cases(
            data_source=self._settings.data_source,
            case_fraction=self._settings.case_fraction,
            validation_fraction=self._settings.validation_fraction,
            test_fraction=self._settings.test_fraction
        )

    def finish(self):
        self._session.close()

    def _build(self):
        # Create input layer
        input_layer = InputLayer(layer_settings=self._settings.input_layer, size=(None, self._cases.get_input_size()))
        input_layer.setup(net=self, name='InputLayer')
        input_layer.build_layer()
        self._input = input_layer.get_layer_output_before_activation()
        self._layers.append(input_layer)

        # Create hidden layers
        last_layer = input_layer
        for index, layer_settings in enumerate(self._settings.hidden_layers):
            hidden_layer = DenseLayer(
                layer_settings=layer_settings,
                size=(last_layer.out_size, layer_settings.size),
            )
            hidden_layer.setup(net=self, name='Layer-{}'.format(index + 1))
            hidden_layer.build_layer(layer_input=last_layer.get_layer_output())
            self._layers.append(hidden_layer)
            last_layer = hidden_layer

        # Create output layer
        output_layer_settings = self._settings.output_layer
        output_layer = DenseLayer(
            layer_settings=output_layer_settings,
            size=(last_layer.out_size, self._cases.get_output_size()),
        )
        output_layer.setup(net=self, name='OutLayer')
        output_layer.build_layer(layer_input=last_layer.get_layer_output())
        self._output = output_layer.get_layer_output()
        self._output_before_activation = output_layer.get_layer_output_before_activation()
        self._layers.append(output_layer)

    def _setup_learning(self):
        self._target = tf.placeholder(tf.float64, shape=(None, self._cases.get_output_size()), name='Target')
        self._learning_rate_placeholder = tf.placeholder(tf.float64, name='LearningRate')

        # Set error function
        if self._settings.error_function == 'mse':
            self._error = tf.reduce_mean(tf.square(self._target - self._output), name='MSE')
        elif self._settings.error_function == 'cross_entropy':
            if self._settings.output_layer.activation == 'softmax':
                self._error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self._target,
                    logits=self._output_before_activation,

                ))
            elif self._settings.output_layer.activation == 'sigmoid':
                self._error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self._target,
                    logits=self._output_before_activation,
                ))

        # Set optimizer
        if self._settings.optimizer == 'gradient':
            self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate_placeholder)
        elif self._settings.optimizer == 'adam':
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate_placeholder)
        elif self._settings.optimizer == 'adagrad':
            self._optimizer = tf.train.AdagradOptimizer(learning_rate=self._learning_rate_placeholder)
        elif self._settings.optimizer == 'rmsprop':
            self._optimizer = tf.train.RMSPropOptimizer(learning_rate=self._learning_rate_placeholder)

        self._trainer = self._optimizer.minimize(self._error, name='Backprop')

    def _run(self, operators, grabbed_vars=None, feeder=None):
        if grabbed_vars:
            results = self._session.run([operators, grabbed_vars], feeder=feeder)
            return results[0], results[1]
        else:
            results = self._session.run([operators], feeder=feeder)
            return results[0], None

    def train(self):
        for step_number in range(1, self._settings.steps + 1):
            grab_variables = [self._error]
            minibatch = self._cases.get_minibatch_of_size(self._settings.minibatch_size)
            self._learning_rate *= self._settings.delta_learning_rate
            feeder = {
                self._input: [case[0] for case in minibatch],
                self._target: [case[1] for case in minibatch],
                self._learning_rate_placeholder: self._learning_rate,
            }
            _, grabbed_values = self._run([self._trainer], grab_variables, feeder=feeder)
            if step_number % self._settings.validation_interval == 0:
                self.display_grabbed_values(grabbed_values, step_number)
                feeder = {
                    self._input: [case[0] for case in self._cases.training_cases],
                    self._target: [case[1] for case in self._cases.training_cases],
                }
                grabbed_values = self._run([self._error], feeder=feeder)
                self._training_error_history.append((step_number, grabbed_values[0]))

                if self._cases.number_of_validation_cases:
                    feeder = {
                        self._input: [case[0] for case in self._cases.validation_cases],
                        self._target: [case[1] for case in self._cases.validation_cases],
                    }
                    grabbed_values = self._run([self._error], feeder=feeder)
                    self._validation_error_history.append((step_number, grabbed_values[0]))

    def display_grabbed_values(self, values, step):
        print('Step {0:>5}, Error: {1:.5f}, Learning Rate: {2:.5f}, Training Accuracy: {3:.3f}%'.format(
            step,
            values[0],
            self._learning_rate,
            self._run_test(self._cases.training_cases)
        ))

    def _run_test(self, cases):
        feeder = {
            self._input: [case[0] for case in cases],
            self._target: [case[1] for case in cases],
        }

        if self._settings.testing == 'classification':
            test = tf.equal(tf.argmax(self._output, 1), tf.argmax(self._target, 1))
        elif self._settings.testing == 'equal':
            test = tf.map_fn(lambda x: tf.reduce_all(x), tf.equal(self._output, self._target))
        else:
            test = None

        grabbed_vars = self._run([test], feeder=feeder)

        correct = 0
        for test_result in grabbed_vars[0][0]:
            if test_result:
                correct += 1
        return (correct / len(cases)) * 100

    def test(self):
        print(' ')
        training_accuracy = self._run_test(self._cases.training_cases)
        print('Training accuracy: {0:.2f}%'.format(training_accuracy))
        if self._cases.number_of_test_cases:
            test_accuracy = self._run_test(self._cases.test_cases)
            print('Test accuracy: {0:.2f}%'.format(test_accuracy))

    def _generate_error_plots(self):
        if self._settings.visualization.error:
            x_axis = [x[0] for x in self._training_error_history]
            y_training = [y[1] for y in self._training_error_history]
            self._plotter.add_error_plot(y_axis=y_training, x_axis=x_axis, label='training')

            if self._cases.number_of_validation_cases:
                y_validation = [y[1] for y in self._validation_error_history]
                self._plotter.add_error_plot(y_axis=y_validation, x_axis=x_axis, label='validation')

    def _generate_layer_mappings(self):
        if self._settings.visualization.mappings:
            cases = self._cases.get_minibatch_of_size(self._settings.map_batch_size)
            variables = []
            labels = []
            for layer in self._layers:
                if layer.layer_settings.visualize_layer.mappings:
                    labels.append(layer.get_name())
                    out = layer.get_layer_output()
                    variables.append(out)

            feeder = {
                self._input: [case[0] for case in cases],
                self._target: [case[1] for case in cases],

            }
            _, grabbed_values = self._run([self._error], variables, feeder=feeder)
            for index, grabbed_value in enumerate(grabbed_values):
                self._plotter.add_hinton_plot(grabbed_value, label=labels[index])

    def _get_initial_weights(self):
        if self._settings.visualization.weights.start:
            variables = []
            labels = []
            for layer in self._layers[1:]:
                if layer.layer_settings.visualize_layer.weights.start:
                    labels.append(layer.get_name())
                    variables.append(layer.weights)
            grabbed_values, _ = self._run(variables)
            for index, grabbed_value in enumerate(grabbed_values):
                self._plotter.add_weight_plot_at_start(grabbed_value, label=labels[index])

    def _get_final_weights(self):
        if self._settings.visualization.weights.end:
            variables = []
            labels = []
            for layer in self._layers[1:]:
                if layer.layer_settings.visualize_layer.weights.end:
                    labels.append(layer.get_name())
                    variables.append(layer.weights)
            grabbed_values, _ = self._run(variables)
            for index, grabbed_value in enumerate(grabbed_values):
                self._plotter.add_weight_plot_at_end(grabbed_value, label=labels[index])

    def _get_initial_biases(self):
        if self._settings.visualization.biases.start:
            variables = []
            labels = []
            for layer in self._layers[1:]:
                if layer.layer_settings.visualize_layer.biases.start:
                    labels.append(layer.get_name())
                    variables.append(layer.biases)
            grabbed_values, _ = self._run(variables)
            for index, grabbed_value in enumerate(grabbed_values):
                self._plotter.add_bias_plot_at_start(grabbed_value, label=labels[index])

    def _get_final_biases(self):
        if self._settings.visualization.biases.end:
            variables = []
            labels = []
            for layer in self._layers[1:]:
                if layer.layer_settings.visualize_layer.biases.end:
                    labels.append(layer.get_name())
                    variables.append(layer.biases)
            grabbed_values, _ = self._run(variables)
            for index, grabbed_value in enumerate(grabbed_values):
                self._plotter.add_bias_plot_at_end(grabbed_value, label=labels[index])

    def _generate_dendrograms(self):
        if self._settings.visualization.dendrograms:
            cases = self._cases.get_minibatch_of_size(self._settings.map_batch_size)
            variables = []
            labels = []
            for layer in self._layers:
                if layer.layer_settings.visualize_layer.dendrograms:
                    labels.append(layer.get_name())
                    out = layer.get_layer_output()
                    variables.append(out)

            targets = [case[1] for case in cases]
            feeder = {
                self._input: [case[0] for case in cases],
            }
            grabbed_values, _ = self._run(variables, feeder=feeder)
            for index, grabbed_value in enumerate(grabbed_values):
                target_labels = ["".join([str(t) for t in target]) for target in targets]
                self._plotter.add_dendrogram_plot(grabbed_value, label=labels[index], labels=target_labels)

    def pre_visualize(self):
        self._get_initial_weights()
        self._get_initial_biases()

    def post_visualize(self):
        self._generate_error_plots()
        self._generate_layer_mappings()
        self._generate_dendrograms()
        self._get_final_weights()
        self._get_final_biases()
        self._plotter.plot()


def main():
    settings = load_settings()
    set_random_seed(seed=settings.seed)
    net = Net(settings=settings)
    net.pre_visualize()
    net.train()
    net.test()
    net.post_visualize()
    net.finish()


if __name__ == '__main__':
    main()
