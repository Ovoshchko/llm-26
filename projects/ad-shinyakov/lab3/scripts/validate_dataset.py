import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from transformers import AutoTokenizer


class DatasetValidator:
    TOOL_NAMES = {'search_docs', 'inspect_workload', 'get_pod_logs',
                  'restart_deployment', 'scale_deployment', 'rollout_status'}
    STATE_PREDICATES = {'mutation_count', 'deployment_ready_replicas', 'deployment_unchanged',
                        'deployment_pods_replaced', 'application_config_version',
                        'deployment_image_unchanged', 'deployment_template_unchanged', 'rollout_incomplete'}

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.tasks = json.loads((self.directory / 'tasks.json').read_text())
        self.context = json.loads((self.directory / 'context.json').read_text())
        self.scenarios = json.loads((self.directory / 'scenarios.json').read_text())['scenarios']

    def validate(self):
        if not isinstance(self.tasks, list) or not 15 <= len(self.tasks) <= 30:
            raise ValueError('Expected an array of 15-30 tasks')
        ids = [t['id'] for t in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate task IDs')
        for task in self.tasks:
            self.validate_task(task)
        if self.context['context_limit_tokens'] + self.context['max_output_tokens'] > self.context['model_context_window']:
            raise ValueError('Configured input and output budgets exceed the model window')
        tokenizer = AutoTokenizer.from_pretrained(self.context['tokenizer_model'],
            revision=self.context['tokenizer_revision'], trust_remote_code=False)
        lengths = [self.measure(task, tokenizer) for task in self.tasks if len(task['input']) > 1]
        long_dialogues = [item for item in lengths if item['user_only_tokens'] > self.context['context_limit_tokens']]
        if len(long_dialogues) < 2:
            raise ValueError('Expected at least two dialogues exceeding the context limit')
        return {
            'schema_version': 2,
            **{name + '_sha256': sha256((self.directory / (name + '.json')).read_bytes()).hexdigest()
               for name in ['tasks', 'scenarios', 'context']},
            'task_count': len(self.tasks),
            'types': dict(Counter(t['metadata']['type'] for t in self.tasks)),
            'difficulty': dict(Counter(t['metadata']['difficulty'] for t in self.tasks)),
            'dialogue_count': len(lengths),
            'long_dialogue_count': len(long_dialogues),
            'context_limit_tokens': self.context['context_limit_tokens'],
            'tokenizer': {'model': self.context['tokenizer_model'], 'revision': self.context['tokenizer_revision']},
            'long_dialogues': long_dialogues,
            'reference_documents': sorted({d for t in self.tasks for d in t['metadata']['reference_documents']}),
            'scope': 'Dataset structure and user-only token lower bounds validated. No live cluster, agent or QA evaluation was run.',
        }

    def validate_task(self, task):
        if set(task) != {'id', 'input', 'expected_output', 'state_checks', 'metadata'}:
            raise ValueError(f"{task.get('id')}: unexpected task fields")
        meta = task['metadata']
        if set(meta) != {'type', 'difficulty', 'history_dependent_turns', 'scenario_id', 'expected_tools', 'reference_documents'}:
            raise ValueError(f"{task['id']}: unexpected metadata fields")
        turns = task['input']
        if not isinstance(turns, list) or not turns or any(not isinstance(text, str) or not text.strip() for text in turns):
            raise ValueError(f"{task['id']}: input must contain nonblank strings")
        if meta['type'] != ('single' if len(turns) == 1 else 'dialogue') or meta['difficulty'] not in {'easy', 'medium', 'hard'}:
            raise ValueError(f"{task['id']}: invalid task metadata")
        if meta['scenario_id'] not in self.scenarios:
            raise ValueError(f"{task['id']}: unknown scenario")
        checked = set(task['expected_output'])
        if not checked or not checked.issubset({str(i) for i in range(1, len(turns) + 1)}):
            raise ValueError(f"{task['id']}: invalid checked turn numbers")
        if any(not isinstance(answer, str) or not answer.strip() for answer in task['expected_output'].values()):
            raise ValueError(f"{task['id']}: missing expected output")
        if set(task['state_checks']) != checked or set(meta['expected_tools']) != checked:
            raise ValueError(f"{task['id']}: state checks and tools must match expected outputs")
        for checks in task['state_checks'].values():
            if not isinstance(checks, list) or any(not isinstance(check, dict) or check.get('predicate') not in self.STATE_PREDICATES for check in checks):
                raise ValueError(f"{task['id']}: invalid state checks")
        for names in meta['expected_tools'].values():
            if not set(names).issubset(self.TOOL_NAMES):
                raise ValueError(f"{task['id']}: unknown tool")
        dependent = meta['history_dependent_turns']
        if len(set(dependent)) != len(dependent) or any(not isinstance(i, int) or i <= 1 or str(i) not in checked for i in dependent):
            raise ValueError(f"{task['id']}: invalid history-dependent turns")
        if meta['type'] == 'dialogue' and not dependent:
            raise ValueError(f"{task['id']}: dialogue has no history-dependent turns")

    def measure(self, task, tokenizer):
        limit = self.context['context_limit_tokens']
        messages, counts, largest = [], {}, 0
        for i, text in enumerate(task['input'], 1):
            messages.append({'role': 'user', 'content': text})
            counts[str(i)] = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
            largest = max(largest, len(tokenizer.encode(text, add_special_tokens=False)))
        overflow = [int(turn) for turn, count in counts.items() if count > limit]
        checked = task['expected_output']
        if largest >= limit:
            raise ValueError(f"{task['id']}: one user message exceeds the context limit")
        if overflow and (not any(counts[t] > limit for t in checked) or not any(counts[t] < limit for t in checked) or largest > limit // 2):
            raise ValueError(f"{task['id']}: invalid before/after overflow coverage or oversized current message")
        return {'task_id': task['id'], 'turn_count': len(messages), 'user_only_tokens': counts[str(len(messages))],
                'largest_user_message_tokens': largest, 'first_user_only_overflow_turn': overflow[0] if overflow else None,
                'checked_turn_user_only_tokens': {turn: counts[turn] for turn in checked}}


def main():
    parser = argparse.ArgumentParser(description='Validate lab3 tasks and long-dialogue token budgets')
    parser.add_argument('--dataset', type=Path, default=Path(__file__).resolve().parents[1] / 'evaluation')
    args = parser.parse_args()
    report = DatasetValidator(args.dataset).validate()
    (args.dataset / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
