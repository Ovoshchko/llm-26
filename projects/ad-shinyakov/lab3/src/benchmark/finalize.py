import argparse
import json
import hashlib
import platform
import re
from importlib.metadata import PackageNotFoundError, version
from dataclasses import asdict
import httpx
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes

from opencode_agent.config import OpenCodeConfig
from opencode_agent.native_tracing import NativeToolTracing
from agent import AgentConfig

from .cluster import LAB_ROOT
from .metrics import RetrievalMetrics, RunMetrics, TaskMetrics, TraceMetrics
from .report import ComparisonReport
from .policy import SuccessPolicy


class ExperimentFinalizer:
    def __init__(self, directory):
        load_dotenv(LAB_ROOT / '.env')
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        self.max_steps = self.manifest['max_steps']
        self.config_id = 'glm4-opencode-server-extractive-v3'
        self.client = get_client()
        self.reviews = self._reviews()
        self.score_ids = {}
        dataset = self.directory / 'dataset.json'
        self.dataset = {task['id']: task for task in json.loads(dataset.read_text())}

    def run(self):
        for agent in ('own', 'opencode'):
            tasks = []
            for path in sorted((self.directory / agent).glob('a??.json')):
                task = json.loads(path.read_text())
                if not task.get('fixture_error'):
                    self._task(agent, task)
                    self._task_summary(agent, task)
                    path.write_text(json.dumps(task, ensure_ascii=False, indent=2) + '\n')
                tasks.append(task)
            self._aggregate(agent, tasks)
        self.client.flush()
        complete = all(len(list((self.directory / agent).glob('a??.json'))) == self.manifest['dataset_size']
                       for agent in ('own', 'opencode'))
        if complete:
            print(ComparisonReport(self.directory).write())
            self._results()
            self._environment(self.directory)
            if self.reviews:
                (LAB_ROOT / 'evaluation/reviewed_scores.json').write_text(
                    (self.directory / 'reviewed_scores.json').read_text())
        else:
            print(f'Частичный прогон: {self.directory}')

    def _results(self):
        results = []
        fields = ('answer', 'trace_id', 'trace_url', 'latency_seconds', 'steps', 'stop_reason',
                  'timeout', 'step_limit_reached', 'context_tokens', 'context_tokens_before', 'compressed', 'metrics',
                  'history_dependent', 'evaluation', 'automatic_evaluation', 'state_checks')
        for agent in ('own', 'opencode'):
            for path in sorted((self.directory / agent).glob('a??.json')):
                task = json.loads(path.read_text())
                turns = []
                for index, turn in enumerate(task.get('turns', []), 1):
                    values = {key: turn[key] for key in fields if key in turn}
                    if not turn.get('evaluation'):
                        values.pop('answer', None)
                    turns.append({'turn': index, **values})
                results.append({'agent': agent, 'id': task['id'], 'type': task['type'],
                                'difficulty': task['difficulty'], 'success': task.get('success'),
                                'metrics': TaskMetrics().calculate(task) if not task.get('fixture_error') else {},
                                'evaluation_trace_url': task.get('evaluation_trace_url'),
                                'fixture_error': task.get('fixture_error'), 'turns': turns})
        destination = LAB_ROOT / 'evaluation/results/tasks.json'
        destination.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')

    def _task_summary(self, agent, task):
        metrics = TaskMetrics().calculate(task)
        turns = task['turns']
        session_id = turns[0].get('langfuse_session_id') or turns[0].get('session_id')
        with self.client.start_as_current_observation(name=f'evaluation.task.{agent}', as_type='evaluator',
                input={'task_id': task['id'], 'turn_traces': [turn.get('trace_id') for turn in turns]}) as span:
            with propagate_attributes(session_id=session_id, tags=['evaluation', agent, self.directory.name],
                    metadata={'run_id': self.directory.name, 'task_id': task['id'],
                              'config_id': self.config_id}):
                span.update(output=metrics)
                task['evaluation_trace_url'] = self.client.get_trace_url()
                self.client.create_score(trace_id=span.trace_id, name='task_success',
                                         value=int(task['success']), data_type='NUMERIC')
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.client.create_score(trace_id=span.trace_id, name=name, value=value, data_type='NUMERIC')

    def _reviews(self):
        path = self.directory / 'reviewed_scores.json'
        if path.is_file():
            values = json.loads(path.read_text())
            if values['run_id'] != self.directory.name:
                raise ValueError('Разметка относится к другому прогону')
            return values['scores']
        return {}

    def _task(self, agent, task):
        audit = self.directory / agent / f'{task["id"]}-state.jsonl'
        entries = [json.loads(line) for line in audit.read_text().splitlines()] if audit.is_file() else []
        history = {}
        if agent == 'opencode':
            session_id = task['turns'][-1].get('session_id')
            if session_id:
                path = self.directory / agent / f'{task["id"]}-history.json'
                history = json.loads(path.read_text())
        for index, turn in enumerate(task['turns'], 1):
            if turn.get('trace_id'):
                turn['metrics'].update(TraceMetrics().read(turn['trace_id']))
            self._tool_counts(agent, turn)
            current = [entry for entry in entries if entry['turn'] == index]
            turn['metrics'].update(RetrievalMetrics().calculate(
                current, self.dataset[task['id']]['metadata']['reference_documents']))
            if agent == 'opencode':
                self._native_tools(turn)
                turn['compressed'] = self._compacted(history, turn)
                previous = task['turns'][index - 2] if index > 1 else {}
                turn['metrics'].update(self._compaction_metrics(history, turn, previous))
                turn['step_limit_reached'] = sum(event['type'] == 'step_finish'
                                                  for event in turn.get('events', [])) >= self.max_steps
            self._review(agent, task['id'], index, turn)
        task['success'] = SuccessPolicy.task(task['turns'])


    def _native_tools(self, turn):
        events = turn.get('events', [])
        if not any(event['type'] == 'tool_use' and not event['part']['tool'].startswith('lab_')
                   for event in events):
            return
        observations = self.client.api.observations.get_many(
            trace_id=turn['trace_id'], name='opencode.turn', limit=1, fields='core,basic').data
        root = next(value for value in observations if value.name == 'opencode.turn')
        NativeToolTracing().record(turn['trace_id'], root.id, events)

    @staticmethod
    def _tool_counts(agent, turn):
        if agent == 'own':
            messages = [message for message in turn.get('messages', []) if message.get('type') == 'tool']
            turn['metrics'].update(tool_calls=len(messages), tool_errors=sum(message.get('status') == 'error' for message in messages))
        else:
            parts = [event['part'] for event in turn.get('events', []) if event.get('type') == 'tool_use']
            turn['metrics'].update(tool_calls=len(parts), tool_errors=sum(part['state']['status'] == 'error' for part in parts))

    @classmethod
    def _compacted(cls, history, turn):
        timestamp = max((event.get('timestamp', 0) for event in turn.get('events', [])), default=0)
        if not timestamp or not history:
            return None
        return any(cls._summary_complete(message, timestamp) for message in cls._summaries(history))

    @staticmethod
    def _summaries(history):
        return [message for message in history.get('messages', [])
                if message['info'].get('role') == 'assistant' and message['info'].get('summary') is True]

    @staticmethod
    def _summary_complete(message, timestamp):
        return (message['info'].get('time', {}).get('completed', timestamp + 1) <= timestamp
                and not message['info'].get('error')
                and any(part.get('type') == 'text' and part.get('text') for part in message['parts']))

    @classmethod
    def _compaction_metrics(cls, history, turn, previous):
        end = max((event.get('timestamp', 0) for event in turn.get('events', [])), default=0)
        start = max((event.get('timestamp', 0) for event in previous.get('events', [])), default=0)
        attempts = [message for message in cls._summaries(history)
                    if start < message['info'].get('time', {}).get('created', 0) <= end]
        successes = sum(bool(cls._summary_complete(message, end)) for message in attempts)
        return {'compression_attempts': len(attempts), 'compression_successes': successes,
                'compression_errors': len(attempts) - successes}

    def _review(self, agent, task_id, index, turn):
        review = self.reviews.get(f'{agent}/{task_id}/{index}')
        if review:
            if review['trace_id'] != turn.get('trace_id'):
                raise ValueError(f'Разметка относится к другому трейсу: {agent}/{task_id}/{index}')
            if review['success'] and not all(check['passed'] for check in turn.get('state_checks', [])):
                raise ValueError(f'Успех противоречит обязательным проверкам: {agent}/{task_id}/{index}')
            turn.setdefault('automatic_evaluation', turn['evaluation'])
            turn['evaluation'] = {**turn['evaluation'], **{key: value for key, value in review.items() if key != 'trace_id'}}
        if turn.get('evaluation') and turn.get('trace_id'):
            if review:
                self._score(turn, 'task_success_automatic', int(turn['automatic_evaluation']['success']))
                self._score(turn, 'task_success_reviewed', int(turn['evaluation']['success']))
            self._score(turn, 'task_success', int(turn['evaluation']['success']))
            for name in ('outcome_achieved', 'key_steps_completed', 'critical_failure'):
                if name in turn['evaluation']:
                    self._score(turn, name, int(turn['evaluation'][name]))
            for key, value in turn['metrics'].items():
                if isinstance(value, (int, float)):
                    self._score(turn, key, value)
            if turn['history_dependent']:
                self._score(turn, 'history_success', int(turn['evaluation']['success']))
                if turn.get('compressed') is not None:
                    suffix = 'after_compression' if turn['compressed'] else 'before_compression'
                    self._score(turn, f'history_success_{suffix}', int(turn['evaluation']['success']))

    def _score(self, turn, name, value):
        trace_id = turn['trace_id']
        if trace_id not in self.score_ids:
            scores = self.client.api.scores_v3.get_many_v3(trace_id=trace_id, limit=100).data
            self.score_ids[trace_id] = {score.name: score.id for score in scores}
        identifier = self.score_ids[trace_id].get(name, str(uuid5(NAMESPACE_URL, f'{trace_id}/{name}/final')))
        evaluation = turn['automatic_evaluation'] if name == 'task_success_automatic' else turn['evaluation']
        self.client.create_score(trace_id=turn['trace_id'], score_id=identifier, name=name, value=value,
                                 data_type='NUMERIC', comment=evaluation['reason'])

    def _aggregate(self, agent, tasks):
        metrics = RunMetrics().calculate(tasks)
        with self.client.start_as_current_observation(name=f'evaluation.final.{agent}', as_type='evaluator',
                                                       input={'run': self.directory.name, 'agent': agent}) as span:
            with propagate_attributes(tags=['evaluation', agent, self.directory.name],
                    metadata={'run_id': self.directory.name,
                              'config_id': self.config_id}):
                span.update(output=metrics)
                metrics['trace_url'] = self.client.get_trace_url()
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.client.create_score(trace_id=span.trace_id, name=name, value=value, data_type='NUMERIC')
        (self.directory / f'{agent}-summary.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n')

    @staticmethod
    def _environment(directory):
        result = {'architecture': platform.machine(), 'python': platform.python_version(),
                  'packages': ExperimentFinalizer._packages(),
                  'agents': ExperimentFinalizer._agent_settings(),
                  'models': ExperimentFinalizer._models(),
                  'source_sha256': ExperimentFinalizer._source_hashes()}
        resources = directory / 'resource-observation.json'
        if resources.is_file():
            result['resources'] = json.loads(resources.read_text())
        (LAB_ROOT / 'evaluation/results/environment.json').write_text(json.dumps(result, indent=2) + '\n')

    @staticmethod
    def _packages():
        packages = {}
        for name in ('langgraph', 'langgraph-checkpoint-sqlite', 'langchain-core', 'langchain-ollama',
                     'langfuse', 'fastmcp', 'transformers', 'sentence-transformers', 'qdrant-client',
                     'jinja2', 'numpy', 'matplotlib'):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        return packages

    @staticmethod
    def _source_hashes():
        paths = [*LAB_ROOT.glob('src/**/*.py'), *LAB_ROOT.glob('prompts/*'),
                 LAB_ROOT / 'opencode/package-lock.json', LAB_ROOT / 'opencode/Modelfile',
                 *LAB_ROOT.glob('opencode/*.md'), LAB_ROOT / 'opencode/opencode.json', LAB_ROOT / 'requirements.txt',
                 *LAB_ROOT.glob('infra/*'), *LAB_ROOT.glob('evaluation/fixture_app/*'),
                 LAB_ROOT / 'evaluation/judge.md', LAB_ROOT / 'evaluation/scenarios.json']
        paths.append(LAB_ROOT / 'evaluation/criteria.json')
        return {str(path.relative_to(LAB_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths if path.is_file()}

    @staticmethod
    def _agent_settings():
        config = OpenCodeConfig()
        settings = config._settings('http://localhost:11434/v1')
        return {'own': asdict(AgentConfig()), 'opencode': {
            **asdict(config), 'limits': settings['provider']['ollama']['models'][config.model]['limit'],
            'compaction': settings['compaction']}}

    @staticmethod
    def _models():
        models = {}
        with httpx.Client(timeout=10) as client:
            for name in ('glm4-agent:9b', 'glm4-opencode:9b'):
                response = client.post('http://localhost:11434/api/show', json={'model': name})
                response.raise_for_status()
                model = response.json()
                digest = re.search(r'sha256[-:]([0-9a-f]{64})', model.get('modelfile', ''))
                models[name] = {'weights_sha256': digest.group(1) if digest else None,
                                'template': model.get('template'), 'parameters': model.get('parameters'),
                                'details': model.get('details')}
        return models


def main():
    parser = argparse.ArgumentParser(description='Обновить метрики из Langfuse и собрать итоговый отчёт')
    parser.add_argument('directory')
    arguments = parser.parse_args()
    ExperimentFinalizer(arguments.directory).run()


if __name__ == '__main__':
    main()
