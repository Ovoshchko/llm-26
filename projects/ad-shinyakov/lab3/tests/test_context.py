import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent import AgentConfig, AgentGraph, AgentNodes, ContextManager
from agent.tokenizer import ChatTokenCounter


class CharacterCounter:
    def count(self, messages, tools=()):
        return 1 + sum(1 + len(message.content) for message in messages)


class AnswerNodes(AgentNodes):
    def call_model(self, state):
        return {'messages': [AIMessage('answer')], 'steps': state['steps'] + 1}


class ContextTests(unittest.TestCase):
    def manager(self, limit=1000):
        compressor = Mock()
        compressor.strategy = 'extractive-v2'
        compressor.compress.side_effect = AssertionError('Unexpected compression')
        return ContextManager(AgentConfig(context_limit_tokens=limit),
                              token_counter=CharacterCounter(), system_prompt='system', compressor=compressor)

    def test_complete_history_and_exact_budget(self):
        history = [HumanMessage('question'), AIMessage('answer'), HumanMessage('next')]
        budget = CharacterCounter().count([SystemMessage('system'), *history])
        result = self.manager(budget).prepare(history)
        self.assertIsNone(result['stop_reason'])
        self.assertEqual(result['model_messages'], [SystemMessage('system'), *history])
        self.assertEqual(result['context_tokens'], budget)
        self.assertEqual(result['context_dropped_messages'], 0)
        self.assertEqual(len(history), 3)

    def test_tool_exchanges_are_kept_whole_in_current_turn(self):
        history = [HumanMessage('now'),
                   AIMessage('', tool_calls=[{'id': 'a', 'name': 'inspect', 'args': {}},
                                             {'id': 'b', 'name': 'logs', 'args': {}}]),
                   ToolMessage('logs', tool_call_id='b'), ToolMessage('status', tool_call_id='a')]
        result = self.manager().prepare(history)
        self.assertEqual(result['model_messages'], [SystemMessage('system'), *history])
        self.assertIsNone(result['stop_reason'])

    def test_overflow_stops_graph_before_model(self):
        manager = self.manager(10)
        result = AgentGraph(AnswerNodes(context_manager=manager)).build().invoke(
            {'messages': [HumanMessage('question' * 10)], 'model_messages': [SystemMessage('stale')]})
        self.assertEqual(result['stop_reason'], 'context_limit')
        self.assertEqual(result['model_messages'], [])
        self.assertEqual(result['steps'], 0)
        self.assertEqual(len(result['messages']), 1)

    def test_large_current_tool_result_stops_instead_of_truncating(self):
        history = [HumanMessage('now'),
                   AIMessage('', tool_calls=[{'id': 'a', 'name': 'logs', 'args': {}}]),
                   ToolMessage('log' * 100, tool_call_id='a')]
        result = self.manager(40).prepare(history)
        self.assertEqual(result['stop_reason'], 'context_limit')
        self.assertEqual(result['model_messages'], [])
        self.assertEqual(result['context_tokens'], result['context_tokens_before'])

    def test_provided_system_message_is_preserved(self):
        system = SystemMessage('custom')
        history = [system, HumanMessage('old' * 100), HumanMessage('now')]
        result = self.manager().prepare(history)
        self.assertEqual(result['model_messages'], history)
        self.assertEqual(history[0], system)

    def test_checkpointer_retains_full_history_after_compression(self):
        manager = self.manager()
        manager.compressor.compress.side_effect = None
        manager.compressor.compress.return_value = {
            'strategy': 'extractive-v2',
            'data': {'goal': [{'text': 'old', 'source_id': 'message-0', 'quote': 'old'}],
                     'constraints': [], 'evidence': [], 'actions': [], 'pending': []},
            'covered_messages': 1,
        }
        graph = AgentGraph(AnswerNodes(context_manager=manager)).build(InMemorySaver())
        config = {'configurable': {'thread_id': 'context-test'}}
        history = [HumanMessage('old' * 1000), HumanMessage('now')]
        first = graph.invoke({'messages': history}, config)
        self.assertEqual(len(first['messages']), 3)
        self.assertEqual(first['context_dropped_messages'], 1)
        self.assertNotIn(history[0], first['model_messages'])
        second = graph.invoke({'messages': [HumanMessage('next')]}, config)
        self.assertEqual(len(second['messages']), 5)
        self.assertEqual(second['steps'], 1)
        self.assertIsNone(second['stop_reason'])
        self.assertEqual(second['model_messages'][-1].content, 'next')
        self.assertEqual(second['task_memory'], first['task_memory'])
        manager.compressor.compress.assert_called_once()

    def test_invalid_history_stops_without_compression(self):
        manager = self.manager()
        for history in ([], [SystemMessage('system')], [AIMessage('answer')]):
            with self.subTest(history=history):
                result = manager.prepare(history)
                self.assertEqual(result['stop_reason'], 'invalid_history')
                self.assertEqual(result['model_messages'], [])
        manager.compressor.compress.assert_not_called()

    def test_reserves_generation_budget(self):
        config = AgentConfig(context_limit_tokens=100, model_context_window=80, max_output_tokens=20)
        self.assertEqual(config.input_budget, 60)
        for name in ('context_limit_tokens', 'model_context_window', 'max_output_tokens'):
            for value in (0, -1, True, 1.5):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    AgentConfig(**{name: value})
        with self.assertRaises(ValueError):
            AgentConfig(model_context_window=1024, max_output_tokens=1024)


class TokenCounterTests(unittest.TestCase):
    def test_loads_native_tokenizer_once_and_counts_rendered_prompt(self):
        tools = [{'type': 'function', 'function': {'name': 'logs', 'parameters': {'type': 'object'}}}]
        messages = [SystemMessage('rules'), HumanMessage('question'),
                    AIMessage('', tool_calls=[{'id': 'a', 'name': 'logs', 'args': {'pod': 'web'}}]),
                    ToolMessage('output', tool_call_id='a')]
        with patch('agent.tokenizer.AutoTokenizer.from_pretrained') as load:
            load.return_value.apply_chat_template.return_value = [1, 2, 3]
            counter = ChatTokenCounter()
            self.assertIsNone(counter.tokenizer)
            self.assertEqual(counter.count(messages, tools), 3)
            self.assertEqual(counter.count(messages, tools), 3)
            load.assert_called_once_with(counter.model, revision=counter.revision, trust_remote_code=False,
                                         local_files_only=os.environ.get('HF_HUB_OFFLINE') == '1')
            args, kwargs = load.return_value.apply_chat_template.call_args
            self.assertEqual(kwargs, {'tokenize': True, 'add_generation_prompt': True})
            self.assertEqual(args[0][0]['tools'], tools)
            self.assertEqual(args[0][-2], {'role': 'assistant', 'metadata': 'logs', 'content': '{"pod": "web"}'})
            self.assertEqual(args[0][-1]['role'], 'observation')
            self.assertEqual(args[0][-1]['content'], 'output')


if __name__ == '__main__':
    unittest.main()
