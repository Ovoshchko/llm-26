import hashlib
import json
import os
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes

from opencode_agent import OpenCodeSession
from opencode_agent.config import OpenCodeConfig
from opencode_agent.server import OpenCodeServer
from agent import AgentConfig

from .checks import StateChecks
from .cluster import LAB_ROOT
from .fixtures import ScenarioFixture
from .judge import AnswerJudge
from .metrics import RetrievalMetrics, RunMetrics, ToolMetrics, TraceMetrics
from .own_agent import OwnAgentSession
from .recording import ToolRecording
from .report import ComparisonReport
from .policy import SuccessPolicy


class ExperimentRunner:
    def __init__(self, agents, task_ids=None, destination=None, opencode_config=None):
        load_dotenv(LAB_ROOT / '.env')
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        self.client = get_client()
        self.agents = agents
        self.opencode_config = opencode_config or OpenCodeConfig()
        self.tasks = json.loads((LAB_ROOT / 'evaluation/tasks.json').read_text())
        self.dataset_size = len(self.tasks)
        if task_ids:
            unknown = set(task_ids) - {task['id'] for task in self.tasks}
            if unknown:
                raise ValueError(f'Неизвестные задачи: {sorted(unknown)}')
            self.tasks = [task for task in self.tasks if task['id'] in task_ids]
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.destination = Path(destination).resolve() if destination else LAB_ROOT / 'data/evaluation' / stamp
        self.destination.mkdir(parents=True, exist_ok=True)
        self.fixture = ScenarioFixture()

    def run(self):
        if not self.client.auth_check():
            raise RuntimeError('Не удалось подключиться к Langfuse')
        self._manifest()
        try:
            for agent in self.agents:
                results = self._run_agent(agent)
                self._aggregate(agent, results)
        finally:
            os.environ.pop('LAB_EVALUATION_STATE', None)
            self.fixture.restore()
            self.client.flush()
        print(f'RESULTS {self.destination}', flush=True)
        if set(self.agents) == {'own', 'opencode'} and len(self.tasks) == self.dataset_size:
            print(f'REPORT {ComparisonReport(self.destination).write()}', flush=True)

    def _run_agent(self, agent):
        context = OpenCodeServer(self.opencode_config) if agent == 'opencode' else nullcontext(None)
        with context as server:
            return self._run_tasks(agent, server)

    def _run_tasks(self, agent, server):
        directory = self.destination / agent
        directory.mkdir(exist_ok=True)
        results = []
        for task in self.tasks:
            path = directory / f'{task["id"]}.json'
            if path.is_file():
                previous = json.loads(path.read_text())
                if not previous.get('fixture_error'):
                    results.append(previous)
                    continue
            print(f'START {agent} {task["id"]} {task["metadata"]["scenario_id"]}', flush=True)
            try:
                result = self._task(agent, task, directory, server)
            except Exception as error:
                result = {'id': task['id'], 'agent': agent, 'fixture_error': str(error), 'turns': []}
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            results.append(result)
            print(f'DONE {agent} {task["id"]} success={result.get("success")} '
                  f'fixture_error={result.get("fixture_error")}', flush=True)
        return results

    def _task(self, agent, task, directory, server):
        initial = self.fixture.setup(task['metadata']['scenario_id'])
        active = directory / f'{task["id"]}-state.json'
        active.unlink(missing_ok=True)
        audit = active.with_suffix('.jsonl')
        audit.unlink(missing_ok=True)
        os.environ['LAB_EVALUATION_STATE'] = str(active)
        metadata = {'run_id': self.destination.name, 'task_id': task['id'],
                    'config_id': 'glm4-opencode-server-extractive-v3'}
        session = (OwnAgentSession(metadata) if agent == 'own'
                   else OpenCodeSession(self.opencode_config, metadata=metadata, server=server))
        context = session if agent == 'own' else nullcontext(session)
        judge = AnswerJudge('glm4-agent:9b' if agent == 'own' else 'glm4-opencode:9b')
        with context as dialogue:
            turns = self._dialogue(agent, task, directory, active, dialogue, judge)
            if agent == 'opencode':
                (directory / f'{task["id"]}-history.json').write_text(
                    json.dumps(dialogue.export(), ensure_ascii=False, indent=2) + '\n')
        return {'id': task['id'], 'agent': agent, 'type': task['metadata']['type'],
                'difficulty': task['metadata']['difficulty'], 'scenario': task['metadata']['scenario_id'],
                'initial_state': initial, 'final_state': self.fixture.cluster.snapshot(),
                'turns': turns, 'success': SuccessPolicy.task(turns), 'evaluation_policy': SuccessPolicy.version}

    def _dialogue(self, agent, task, directory, active, dialogue, judge):
        turns = []
        for turn, question in enumerate(task['input'], 1):
            self.fixture.before_turn(turn)
            self._active(active, task, turn)
            result = self._turn(dialogue, question)
            self._evaluate_turn(task, turn, result, judge)
            turns.append(result)
            (directory / f'{task["id"]}-partial.json').write_text(json.dumps(turns, ensure_ascii=False, indent=2))
            print(f'TURN {agent} {task["id"]} {turn}/{len(task["input"])} '
                  f'{result.get("evaluation", {}).get("success")} {result.get("trace_url", "")}', flush=True)
        return turns

    def _evaluate_turn(self, task, turn, result, judge):
        entries = ToolRecording().entries()
        current = [entry for entry in entries if entry['turn'] == turn]
        result['metrics'] = self._metrics(task, turn, result, current)
        result['history_dependent'] = turn in task['metadata']['history_dependent_turns']
        if str(turn) in task['expected_output']:
            checks = StateChecks(self.fixture).evaluate(task['state_checks'].get(str(turn), []), entries)
            result['state_checks'] = checks
            result['evaluation'] = judge.evaluate(task, turn, result, checks, current)
            self._scores(result)

    @staticmethod
    def _active(path, task, turn):
        previous = json.loads(path.read_text()) if path.is_file() else {}
        path.write_text(json.dumps({'scenario': task['metadata']['scenario_id'], 'turn': turn,
                                    'fault_injected': previous.get('fault_injected', False)}))

    @staticmethod
    def _turn(dialogue, question):
        try:
            return dialogue.ask(question)
        except Exception as error:
            return {'answer': '', 'stop_reason': 'exception', 'error': str(error), 'latency_seconds': 0}

    def _metrics(self, task, turn, result, entries):
        expected = task['metadata']['expected_tools'].get(str(turn), [])
        values = TraceMetrics().read(result['trace_id']) if result.get('trace_id') else {
            'generations': 0, 'input_tokens': 0, 'output_tokens': 0, 'cost_usd': 0, 'max_input_tokens': 0}
        values.update(ToolMetrics().calculate(entries, expected))
        values.update(RetrievalMetrics().calculate(entries, task['metadata']['reference_documents']))
        return values

    def _scores(self, result):
        if not result.get('trace_id'):
            return
        scores = {key: value for key, value in result['metrics'].items()
                  if isinstance(value, (int, float))}
        scores['task_success'] = int(result['evaluation']['success'])
        for name in ('outcome_achieved', 'key_steps_completed', 'critical_failure'):
            if name in result['evaluation']:
                scores[name] = int(result['evaluation'][name])
        for name, value in scores.items():
            self.client.create_score(trace_id=result['trace_id'], name=name, value=value,
                                     data_type='NUMERIC', comment=result['evaluation']['reason'])

    def _aggregate(self, agent, results):
        metrics = RunMetrics().calculate(results)
        path = self.destination / f'{agent}-summary.json'
        with self.client.start_as_current_observation(name=f'evaluation.run.{agent}', as_type='evaluator',
                                                       input={'run': self.destination.name, 'agent': agent}) as span:
            with propagate_attributes(tags=['evaluation', agent, self.destination.name]):
                span.update(output=metrics)
                metrics['trace_url'] = self.client.get_trace_url()
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.client.create_score(trace_id=span.trace_id, name=name, value=value, data_type='NUMERIC')
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n')

    def _manifest(self):
        path = self.destination / 'manifest.json'
        identity = {'dataset_sha256': hashlib.sha256((LAB_ROOT / 'evaluation/tasks.json').read_bytes()).hexdigest(),
                    'dataset_size': self.dataset_size,
                    'tasks': [task['id'] for task in self.tasks], 'agents': self.agents,
                    'embedding_manifest': json.loads((LAB_ROOT.parent / 'lab2/data/embeddings/ced08184c9da0803/manifest.json').read_text()),
                    'fixture_image': self.fixture.image, 'input_budget': 8192, 'model_window': 32768,
                    'output_budget': AgentConfig().max_output_tokens, 'max_steps': AgentConfig().max_steps,
                    'timeout_seconds': AgentConfig().timeout_seconds, 'evaluation_policy': SuccessPolicy.version,
                    'context_strategy': 'extractive-v2', 'opencode_structured_tools': self.opencode_config.structured_tools,
                    'opencode_transport': 'persistent-http',
                    'criteria_sha256': hashlib.sha256((LAB_ROOT / 'evaluation/criteria.json').read_bytes()).hexdigest(),
                    'opencode_prompt_sha256': hashlib.sha256((LAB_ROOT / 'opencode/system.md').read_bytes()).hexdigest()}
        identity['opencode_additional_prompts'] = {
            name: hashlib.sha256((LAB_ROOT / 'opencode' / name).read_bytes()).hexdigest()
            for name in ('compaction.md', 'reply.md')}
        if path.is_file() and json.loads(path.read_text()) != identity:
            raise ValueError('Параметры существующего прогона отличаются; выбери другой каталог')
        path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + '\n')
        (self.destination / 'dataset.json').write_text(json.dumps(self.tasks, ensure_ascii=False, indent=2) + '\n')
