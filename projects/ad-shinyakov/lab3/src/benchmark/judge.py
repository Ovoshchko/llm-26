import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

from .cluster import LAB_ROOT
from .policy import SuccessPolicy
from .metrics import ToolMetrics


class Verdict(BaseModel):
    outcome_achieved: bool
    key_steps_completed: bool
    critical_failure: bool
    category: Literal['none', 'tool_selection', 'tool_arguments', 'reasoning', 'history',
                      'verification', 'context_limit', 'limits']
    reason: str = Field(min_length=1, max_length=600)


class AnswerJudge:
    def __init__(self, model='glm4-agent:9b'):
        self.client = ChatOllama(model=model, temperature=0, num_ctx=32768,
                                 num_predict=512, client_kwargs={'timeout': 120}).with_structured_output(
                                     Verdict, method='json_schema')
        self.prompt = (LAB_ROOT / 'evaluation/judge.md').read_text()
        self.criteria = json.loads((LAB_ROOT / 'evaluation/criteria.json').read_text())['tasks']

    def evaluate(self, task, turn, result, checks, entries):
        payload = {'task': task['input'][0], 'question': task['input'][turn - 1],
                   'expected': task['expected_output'][str(turn)], 'answer': result['answer'],
                   'rubric': self.criteria[task['id']][str(turn)],
                   'state_checks': checks, 'tools': self._tools(entries)}
        with get_client().start_as_current_observation(name='evaluation.judge', as_type='evaluator',
                                                       input={'task_id': task['id'], 'turn': turn}) as span:
            verdict = self.client.invoke([SystemMessage(self.prompt),
                                          HumanMessage(json.dumps(payload, ensure_ascii=False))],
                                         config={'callbacks': [CallbackHandler()]})
            value = self.validate(verdict.model_dump(), payload['rubric'], checks, result, entries)
            span.update(output=value)
            return value

    @staticmethod
    def validate(verdict, rubric, checks, result, entries):
        value = dict(verdict)
        missing = ToolMetrics().missing(entries, rubric.get('required_tools', []))
        if missing:
            value.update(key_steps_completed=False, category='tool_selection',
                         reason='Не выполнены ключевые вызовы: ' + ', '.join(missing))
        return SuccessPolicy().turn(value, checks, result)

    @staticmethod
    def _tools(entries):
        tools = []
        for entry in entries[-8:]:
            result = json.dumps(entry.get('result'), ensure_ascii=False)
            tools.append({'name': entry['name'], 'arguments': entry['arguments'], 'error': entry.get('error'),
                          'result': result[:3500], 'result_truncated': len(result) > 3500})
        return tools
