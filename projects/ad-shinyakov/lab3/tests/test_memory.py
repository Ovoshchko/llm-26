import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent import AgentConfig, AgentNodes, ContextManager, DialogueSession, OllamaModel
from agent.memory import MemoryCompressor


class CharacterCounter:
    def count(self, messages, tools=()):
        return sum(len(message.content) for message in messages)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.config = AgentConfig(context_limit_tokens=3000)
        self.counter = CharacterCounter()
        self.context = ContextManager(self.config, self.counter, 'system')
        self.history = [HumanMessage('Проверь api. Не меняй реплики.'),
                        AIMessage('', tool_calls=[{'id': 'a', 'name': 'logs', 'args': {'name': 'api'}}]),
                        ToolMessage('FATAL CONFIG_ENDPOINT is required\n' + 'архив ' * 1000, tool_call_id='a'),
                        AIMessage('Выдуманный факт: всё исправлено.'), HumanMessage('В чём причина?')]

    def test_extracts_user_and_observations_without_model_or_assistant_claims(self):
        result = self.context.prepare(self.history)
        self.assertIsNone(result['stop_reason'])
        self.assertLessEqual(result['context_tokens'], self.config.input_budget)
        records = result['task_memory']['data']['records']
        text = json.dumps(records, ensure_ascii=False)
        self.assertIn('Не меняй реплики', text)
        self.assertIn('CONFIG_ENDPOINT', text)
        self.assertNotIn('всё исправлено', text)
        self.assertEqual(records[1]['name'], 'logs')
        self.assertEqual(records[1]['arguments'], {'name': 'api'})
        self.assertEqual(result['model_messages'][-1], self.history[-1])
        self.assertEqual(len(self.history), 5)
        self.assertFalse(hasattr(self.context.compressor, 'client'))

    def test_last_clarification_survives_many_large_archived_reports(self):
        history = [HumanMessage('Основная задача worker, цель четыре. Control не менять.'),
                   HumanMessage('Уточняю: теперь три вместо четырёх.')]
        for i in range(15):
            history.extend([HumanMessage('Архив другого приложения.\n```text\n' + str(i) * 5000 + '\n```'),
                            AIMessage('Это исторические данные.')])
        history.append(HumanMessage('Примени последнее значение.'))
        result = self.context.prepare(history)
        self.assertIsNone(result['stop_reason'])
        text = result['model_messages'][1].content
        self.assertIn('теперь три вместо четырёх', text)
        self.assertIn('Control не менять', text)
        self.assertLessEqual(result['context_tokens'], self.config.input_budget)

    def test_current_tool_exchange_is_never_split(self):
        history = [*self.history, AIMessage('', tool_calls=[{'id': 'b', 'name': 'status', 'args': {}}]),
                   ToolMessage('Ready 0/2', tool_call_id='b')]
        result = self.context.prepare(history)
        self.assertEqual(result['model_messages'][-3:], history[-3:])

    def test_genuinely_oversized_current_turn_returns_context_limit(self):
        result = self.context.prepare([HumanMessage('x' * 4000)])
        self.assertEqual(result['stop_reason'], 'context_limit')
        self.assertEqual(result['model_messages'], [])

    def test_memory_can_shrink_again_when_current_tool_result_grows(self):
        first = self.context.prepare(self.history)
        history = [*self.history, AIMessage('', tool_calls=[{'id': 'b', 'name': 'status', 'args': {}}]),
                   ToolMessage('x' * 1400, tool_call_id='b')]
        result = self.context.prepare(history, first['task_memory'])
        self.assertIsNone(result['stop_reason'])
        self.assertEqual(result['model_messages'][-3:], history[-3:])
        self.assertIn('Не меняй реплики', result['model_messages'][1].content)

    def test_old_llm_memory_is_rebuilt_from_real_history(self):
        old = {'data': {'goal': [{'text': 'Выдуманный worker'}]}, 'covered_messages': 4}
        result = self.context.prepare(self.history, old)
        self.assertIsNone(result['stop_reason'])
        self.assertEqual(result['task_memory']['strategy'], 'extractive-v2')
        self.assertNotIn('Выдуманный worker', result['model_messages'][1].content)

    def test_sqlite_keeps_raw_history_and_deterministic_memory(self):
        model = OllamaModel(self.config)
        model.client = Mock()
        model.client.invoke.return_value = AIMessage('Ответ')
        nodes = AgentNodes(self.config, context_manager=self.context, model=model)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'dialogue.sqlite'
            with DialogueSession(nodes, database=path, thread_id='test') as session:
                result = session.graph.invoke({'messages': self.history},
                                              {'configurable': {'thread_id': 'test'}})
            with DialogueSession(nodes, database=path, thread_id='test') as session:
                restored = session.ask('Какое ограничение?')
                self.assertEqual(restored['task_memory'], result['task_memory'])
                self.assertEqual(restored['messages'][2].content, self.history[2].content)
                self.assertIn('Не меняй реплики', restored['model_messages'][1].content)


if __name__ == '__main__':
    unittest.main()
