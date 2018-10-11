import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import numpy as np


def create_hinton_plot(
        matrix,
        max_value=None,
        max_size=1,
        number=1,
        total=1,
        figure=None,
        transpose=True,
        scale=True,
        label='Hinton plot',
):
    colors = ('gray', 'red', 'blue', 'white')
    if transpose:
        matrix = matrix.transpose()
    if not max_value:
        max_value = np.abs(matrix).max()
    if not max_size:
        max_size = 2 ** np.ceil(np.log(max_value) / np.log(2))

    axes = figure.add_subplot(1, total, number)
    axes.clear()
    axes.patch.set_facecolor(colors[0])
    axes.set_aspect('equal', 'box')
    axes.set_title(label)
    axes.xaxis.set_major_locator(plt.NullLocator())
    axes.yaxis.set_major_locator(plt.NullLocator())

    ymax = (matrix.shape[1] - 1) * max_size
    for (x, y), val in np.ndenumerate(matrix):
        color = colors[1] if val > 0 else colors[2]
        if scale:
            size = max(0.01, np.sqrt(min(max_size, max_size * np.abs(val) / max_value)))
        else:
            size = np.sqrt(min(np.abs(val), max_size))
        bottom_left = [x - size / 2, (ymax - y) - size / 2]
        blob = plt.Rectangle(bottom_left, size, size, facecolor=color, edgecolor=colors[3])
        axes.add_patch(blob)
    axes.autoscale_view()
    figure.add_subplot(axes)


def create_matrix_plot(
        matrix,
        figure=None,
        transpose=True,
        label='Matrix',
        format='{:.2f}',
        number=1,
        total=1,
        tsize=8,
        cutoff=0.1,
):
    colors = ('red', 'yellow', 'grey', 'blue')
    if transpose:
        matrix = matrix.transpose()
    axes = figure.add_subplot(1, total, number)
    axes.clear()
    axes.patch.set_facecolor('white')
    axes.set_aspect('equal', 'box')
    axes.set_title(label)
    axes.xaxis.set_major_locator(plt.NullLocator())
    axes.yaxis.set_major_locator(plt.NullLocator())

    if len(matrix.shape) == 1:
        matrix = np.reshape(matrix, (1, matrix.shape[0]))
    ymax = matrix.shape[1] - 1
    for (x, y), val in np.ndenumerate(matrix):
        if val > 0:
            color = colors[0] if val > cutoff else colors[1]
        else:
            color = colors[3] if val < -cutoff else colors[2]
        botleft = [x - 1 / 2, (ymax - y) - 1 / 2]  # (ymax - y) to invert: row 0 at TOP of diagram
        # This is a hack, but I seem to need to add these blank blob rectangles first, and then I can add the text
        # boxes.  If I omit the blobs, I get just one plotted textbox...grrrrrr.
        blob = plt.Rectangle(botleft, 1, 1, facecolor='white', edgecolor='white')
        axes.add_patch(blob)
        axes.text(botleft[0] + 0.5, botleft[1] + 0.5, format.format(val),
                  bbox=dict(facecolor=color, alpha=0.5, edgecolor='white'), ha='center', va='center',
                  color='black', size=tsize)
    axes.autoscale_view()
    figure.add_subplot(axes)


def create_error_plot(error_plots):
    title = 'Error History'
    figure = plt.figure()
    figure.canvas.set_window_title(title)
    for error_plot in error_plots:
        plt.plot(
            error_plot['x_axis'],
            error_plot['y_axis'],
            label=error_plot['label']
        )
    plt.xlabel('Steps')
    plt.ylabel('Error')
    plt.title(title)
    plt.legend()
    plt.draw()


def create_dendrogram(
        features,
        labels,
        metric='euclidean',
        mode='average',
        figure=None,
        orient='top',
        lrot=90.0
):
    axes = figure.gca()
    cluster_history = sch.linkage(features,method=mode,metric=metric)
    sch.dendrogram(cluster_history,labels=labels,orientation=orient,leaf_rotation=lrot)
    plt.tight_layout()
    axes.set_ylabel(metric + ' distance')


def get_figure_with_title(title):
    figure = plt.figure()
    figure.canvas.set_window_title(title)
    figure.suptitle(title)
    return figure


class Plotter:
    def __init__(self):
        self._error_plots = []
        self._hinton_plots = []
        self._start_weights = []
        self._end_weights = []
        self._start_biases = []
        self._end_biases = []
        self._dendrograms = []

    def add_hinton_plot(self, matrix, label='Matrix'):
        self._hinton_plots.append({
            'data': matrix,
            'label': label,
        })

    def add_error_plot(self, y_axis, x_axis, label):
        self._error_plots.append({
            'y_axis': y_axis,
            'x_axis': x_axis,
            'label': label,
        })

    def add_weight_plot_at_start(self, matrix, label='Matrix'):
        self._start_weights.append({
            'data': matrix,
            'label': label,
        })

    def add_weight_plot_at_end(self, matrix, label='Matrix'):
        self._end_weights.append({
            'data': matrix,
            'label': label,
        })

    def add_bias_plot_at_end(self, matrix, label='Matrix'):
        self._end_biases.append({
            'data': matrix,
            'label': label,
        })

    def add_bias_plot_at_start(self, matrix, label='Matrix'):
        self._start_biases.append({
            'data': matrix,
            'label': label,
        })

    def add_dendrogram_plot(self, matrix, labels, label='Matrix'):
        self._dendrograms.append({
            'data': matrix,
            'labels': labels,
            'label': label,
        })

    def plot(self):
        # Error plots
        if len(self._error_plots):
            create_error_plot(self._error_plots)

        # Plot of layer mappings
        total_hinton_plots = len(self._hinton_plots)
        if total_hinton_plots:
            figure = get_figure_with_title('Layer Mappings')
            for index, hinton_plot in enumerate(self._hinton_plots):
                create_hinton_plot(hinton_plot['data'], label=hinton_plot['label'], number=index + 1,
                                   total=total_hinton_plots, figure=figure)

        total_start_weights = len(self._start_weights)
        if total_start_weights:
            figure = get_figure_with_title('Initial Weights')
            for index, plot in enumerate(self._start_weights):
                create_matrix_plot(plot['data'], label=plot['label'], number=index + 1,
                                   total=total_start_weights, figure=figure)

        total_end_weights = len(self._end_weights)
        if total_end_weights:
            figure = get_figure_with_title('Final Weights')
            for index, plot in enumerate(self._end_weights):
                create_matrix_plot(plot['data'], label=plot['label'], number=index + 1,
                                   total=total_end_weights, figure=figure)

        total_start_biases = len(self._start_biases)
        if total_start_biases:
            figure = get_figure_with_title('Initial Biases')
            for index, plot in enumerate(self._start_biases):
                create_matrix_plot(plot['data'], label=plot['label'], number=index + 1,
                                   total=total_start_biases, figure=figure)

        total_end_biases = len(self._end_biases)
        if total_end_biases:
            figure = get_figure_with_title('Final Biases')
            for index, plot in enumerate(self._end_biases):
                create_matrix_plot(plot['data'], label=plot['label'], number=index + 1,
                                   total=total_end_biases, figure=figure)

        total_dendrograms = len(self._dendrograms)
        if total_dendrograms:
            for dendrogram in self._dendrograms:
                figure = get_figure_with_title('Dendrogram-' + dendrogram['label'])
                create_dendrogram(dendrogram['data'], labels=dendrogram['labels'], figure=figure)

        plt.show()
