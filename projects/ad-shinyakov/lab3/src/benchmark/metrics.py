import json
from statistics import mean
from time import sleep

from langfuse import get_client


class RetrievalMetrics:
    def calculate(self, entries, references, k=3):
        documents = []
        for entry in entries:
            result = entry.get('result')
            if entry['name'] != 'search_docs' or not isinstance(result, dict):
                continue
            for passage in result.get('passages', []):
                path = self._document(passage)
                if path and path not in documents:
                    documents.append(path)
        ranked = documents[:k]
        relevant = set(references)
        hits = [position for position, path in enumerate(ranked, 1) if path in relevant]
        return {'retrieval_recall_at_3': len(hits) / len(relevant) if relevant else None,
                'retrieval_precision_at_3': len(hits) / k if relevant else None,
                'retrieval_mrr': 1 / hits[0] if hits else (0 if relevant else None),
                'retrieved_documents': ranked}

    @staticmethod
    def _document(passage):
        path = (passage.get('document_path') or passage.get('metadata', {}).get('document_path')
                or passage.get('source_path'))
        return path.removeprefix('content/en/') if path else None


class ToolMetrics:
    aliases = {'inspect_workload': {'resources_get', 'resources_list', 'pods_list_in_namespace', 'events_list', 'rollout_status'},
               'get_pod_logs': {'pods_log'}}

    def missing(self, entries, required):
        names = {entry['name'] for entry in entries if not entry.get('error')
                 or isinstance(entry.get('result'), dict) and entry['result'].get('accepted')}
        return [name for name in required if not names.intersection(self.aliases.get(name, {name}))]

    def calculate(self, entries, expected):
        names = {entry['name'] for entry in entries}
        matched = [bool(names.intersection(self.aliases.get(name, {name}))) for name in expected]
        return {'tool_correctness': sum(matched) / len(matched) if matched else None,
                'tool_calls': len(entries),
                'tool_errors': sum(bool(entry.get('error')) for entry in entries)}


class TraceMetrics:
    def read(self, trace_id):
        client = get_client()
        observations = []
        for attempt in range(5):
            observations = self._observations(client, trace_id, fields='core,basic,usage')
            if any(observation.model_dump(mode='json').get('isRootObservation') for observation in observations):
                break
            sleep(1)
        values = [observation.model_dump(mode='json') for observation in observations]
        generations = [value for value in values if value['type'] == 'GENERATION']
        metrics = {'generations': len(generations),
                'generations_without_usage': sum(not any((value.get('usageDetails') or {}).get(key, 0)
                                                         for key in ('input', 'output')) for value in generations),
                'input_tokens': sum((value.get('usageDetails') or {}).get('input', 0) for value in generations),
                'output_tokens': sum((value.get('usageDetails') or {}).get('output', 0) for value in generations),
                'cost_usd': sum((value.get('costDetails') or {}).get('total', 0) for value in generations),
                'max_input_tokens': max(((value.get('usageDetails') or {}).get('input', 0)
                                         for value in generations), default=0)}
        metrics.update(self._compression(client, trace_id, values))
        return metrics

    @staticmethod
    def _observations(client, trace_id, **filters):
        values = []
        cursor = None
        while True:
            page = client.api.observations.get_many(trace_id=trace_id, limit=100, cursor=cursor, **filters)
            values.extend(page.data)
            cursor = page.meta.cursor
            if not cursor:
                return values

    @staticmethod
    def _compression(client, trace_id, observations):
        attempts = [observation for observation in observations if observation.get('name') == 'context.compress']
        successes = 0
        errors = 0
        if attempts:
            values = client.api.observations.get_many(trace_id=trace_id, name='context.compress',
                                                       limit=100, fields='core,basic,io').data
            for observation in values:
                output = observation.output
                result = json.loads(output) if isinstance(output, str) else (output or {})
                successes += bool(result.get('task_memory')) and not result.get('stop_reason')
                errors += bool(result.get('stop_reason'))
        return {'compression_attempts': len(attempts), 'compression_successes': successes, 'compression_errors': errors}


class RunMetrics:
    def calculate(self, results):
        checked = [turn for task in results for turn in task.get('turns', []) if turn.get('evaluation')]
        evaluated = [task for task in results if not task.get('fixture_error')]
        return {'tasks': len(results), 'fixture_errors': len(results) - len(evaluated),
                'success_rate': self._rate(evaluated),
                'turn_success_rate': self._turn_rate(checked),
                'strict_success_rate': mean(all(turn['evaluation']['success'] for turn in task['turns']
                                                 if turn.get('evaluation')) for task in evaluated) if evaluated else None,
                'single_success_rate': self._rate([task for task in evaluated if task['type'] == 'single']),
                'dialogue_success_rate': self._rate([task for task in evaluated if task['type'] != 'single']),
                'history_accuracy': self._turn_rate([turn for turn in checked if turn['history_dependent']]),
                'history_before_compression': self._turn_rate([turn for turn in checked
                                                              if turn['history_dependent'] and turn.get('compressed') is False]),
                'history_after_compression': self._turn_rate([turn for turn in checked
                                                             if turn['history_dependent'] and turn.get('compressed') is True]),
                'mean_task_latency_seconds': mean(sum(turn['latency_seconds'] for turn in task['turns'])
                                                  for task in evaluated) if evaluated else None,
                'mean_task_cost_usd': mean(sum(turn['metrics']['cost_usd'] for turn in task['turns'])
                                           for task in evaluated) if evaluated else None,
                'input_tokens': sum(turn['metrics']['input_tokens'] for task in evaluated for turn in task['turns']),
                'output_tokens': sum(turn['metrics']['output_tokens'] for task in evaluated for turn in task['turns']),
                'compression_attempts': sum(turn['metrics'].get('compression_attempts', 0)
                                            for task in evaluated for turn in task['turns']),
                'compression_successes': sum(turn['metrics'].get('compression_successes', 0)
                                             for task in evaluated for turn in task['turns']),
                'generations_without_usage': sum(turn['metrics'].get('generations_without_usage', 0)
                                                  for task in evaluated for turn in task['turns']),
                'error_rate': mean(any(turn.get('stop_reason') or turn.get('timeout')
                                       or turn.get('step_limit_reached') or turn['metrics']['tool_errors']
                                       for turn in task['turns']) for task in evaluated) if evaluated else None}

    @staticmethod
    def _rate(tasks):
        return mean(task['success'] for task in tasks) if tasks else None

    @staticmethod
    def _turn_rate(turns):
        return mean(turn['evaluation']['success'] for turn in turns) if turns else None


class TaskMetrics:
    def calculate(self, task):
        metrics = RunMetrics().calculate([task])
        turns = task['turns']
        checked = [turn for turn in turns if turn.get('evaluation')]
        for key in ('generations', 'tool_calls', 'tool_errors'):
            metrics[key] = sum(turn['metrics'].get(key, 0) for turn in turns)
        metrics['max_input_tokens'] = max((turn['metrics']['max_input_tokens'] for turn in turns), default=0)
        for key in ('tool_correctness', 'retrieval_recall_at_3', 'retrieval_precision_at_3', 'retrieval_mrr'):
            values = [turn['metrics'][key] for turn in checked if turn['metrics'].get(key) is not None]
            metrics[key] = mean(values) if values else None
        return metrics
