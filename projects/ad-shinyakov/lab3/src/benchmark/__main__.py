import argparse

from .runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description='Последовательная оценка двух агентов на локальном k3d')
    parser.add_argument('--agent', choices=['own', 'opencode', 'both'], default='both')
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--output')
    arguments = parser.parse_args()
    agents = ['own', 'opencode'] if arguments.agent == 'both' else [arguments.agent]
    ExperimentRunner(agents, arguments.tasks, arguments.output).run()


if __name__ == '__main__':
    main()
