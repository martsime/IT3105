import os

import tensorflow as tf


class Session:
    def __init__(self, directory='probedir', use_tensorboard=True):
        self._directory = directory
        self._use_tensorboard = use_tensorboard
        self._session = tf.Session()
        if use_tensorboard:
            self._clear_tensorboard()
            self._session.probe_stream = tf.summary.FileWriter(
                self._directory,
                self._session.graph,
                max_queue=10,
                flush_secs=120,
            )
            self._session.viewdir = self._directory
        self._session.run(tf.global_variables_initializer())

    def run(self, graph_elements, feeder=None):
        return self._session.run(graph_elements, feed_dict=feeder)

    def close(self):
        if self._use_tensorboard:
            self._session.probe_stream.close()
        self._session.close()
        if self._use_tensorboard:
            self._fireup_tensorboard()

    def _clear_tensorboard(self):
        os.system('rm ' + self._directory + '/events.out.*')

    def _fireup_tensorboard(self):
        os.system('tensorboard --logdir=' + self._directory)
