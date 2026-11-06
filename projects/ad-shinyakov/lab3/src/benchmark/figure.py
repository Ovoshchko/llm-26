import os

from .cluster import LAB_ROOT


class ComparisonFigure:
    def write(self, runs, destination):
        os.environ.setdefault('MPLCONFIGDIR', str(LAB_ROOT / 'data/matplotlib'))
        import matplotlib

        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap

        identifiers = sorted({task['id'] for tasks in runs.values() for task in tasks})
        matrix = [self._row(runs[agent], identifiers) for agent in ('own', 'opencode')]
        figure, axes = plt.subplots(figsize=(14, 2.5), layout='constrained')
        axes.imshow(matrix, cmap=ListedColormap(['#d9d9d9', '#d95f02', '#1b9e77']),
                    norm=BoundaryNorm([-1.5, -.5, .5, 1.5], 3), aspect='auto')
        kinds = {task['id']: task['type'] for tasks in runs.values() for task in tasks}
        boundaries = [index + .5 for index in range(len(identifiers) - 1)
                      if kinds[identifiers[index]] != kinds[identifiers[index + 1]]]
        self._decorate(axes, identifiers, matrix, boundaries)
        figure.savefig(destination, dpi=180)
        plt.close(figure)

    @staticmethod
    def _row(tasks, identifiers):
        indexed = {task['id']: task for task in tasks}
        return [int(indexed[name]['success']) if name in indexed and not indexed[name].get('fixture_error')
                else -1 for name in identifiers]

    @staticmethod
    def _decorate(axes, identifiers, matrix, boundaries):
        axes.set_xticks(range(len(identifiers)), identifiers, fontsize=9)
        axes.set_yticks([0, 1], ['LangGraph', 'OpenCode'])
        axes.set_title('Task success: 1 = passed, 0 = failed, — = unavailable', pad=12)
        for row, values in enumerate(matrix):
            for column, value in enumerate(values):
                axes.text(column, row, str(value) if value >= 0 else '—', ha='center', va='center',
                          color='white' if value >= 0 else 'black', fontsize=10)
        for boundary in boundaries:
            axes.axvline(boundary, color='white', linewidth=2)
