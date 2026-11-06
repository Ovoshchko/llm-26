import unittest
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agent import AgentConfig, AgentGraph, AgentNodes, ContextManager, OllamaModel
from agent.routing import after_limits


class Counter:
    def count(self, messages, tools=()):
        return len(messages)


class ToolRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.executed = []

        @tool(description='Прочитать состояние приложения.')
        def inspect(name: str) -> dict:
            self.executed.append(name)
            if name == 'broken':
                raise RuntimeError('App not found; use web')
            return {'name': name, 'ready': True}

        self.tool = inspect
        self.config = AgentConfig()
        self.model = OllamaModel(self.config, [inspect])
        self.model.client = Mock()
        self.context = ContextManager(self.config, token_counter=Counter(), tools=self.model.tools)
        self.nodes = AgentNodes(self.config, model=self.model, context_manager=self.context, tools=[inspect])

    def call(self, identifier, name):
        return {'id': identifier, 'name': 'inspect', 'args': {'name': name}}

    def test_pending_tool_runs_before_model(self):
        self.model.client.invoke.return_value = AIMessage('Готово')
        history = [HumanMessage('Проверь web'), AIMessage('', tool_calls=[self.call('a', 'web')])]
        result = AgentGraph(self.nodes).build().invoke({'messages': history})
        self.assertEqual(self.executed, ['web'])
        model_input = self.model.client.invoke.call_args.args[0]
        results = [message for message in model_input if isinstance(message, ToolMessage)]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tool_call_id, 'a')
        self.assertEqual(result['steps'], 1)
        self.assertEqual(result['pending_tool_calls'], [])
        self.assertIsNone(result['stop_reason'])

    def test_partial_parallel_exchange_only_runs_missing_call(self):
        self.model.client.invoke.return_value = AIMessage('Готово')
        history = [HumanMessage('Проверь оба приложения'),
                   AIMessage('', tool_calls=[self.call('a', 'already-done'), self.call('b', 'web')]),
                   ToolMessage('ready', tool_call_id='a')]
        result = AgentGraph(self.nodes).build().invoke({'messages': history})
        self.assertEqual(self.executed, ['web'])
        results = [m for m in result['messages'] if isinstance(m, ToolMessage)]
        self.assertEqual([m.tool_call_id for m in results], ['a', 'b'])

    def test_tool_error_is_returned_and_model_can_retry_with_new_arguments(self):
        self.model.client.invoke.side_effect = [
            AIMessage('', tool_calls=[self.call('a', 'broken')]),
            AIMessage('', tool_calls=[self.call('b', 'web')]), AIMessage('Готово')]
        result = AgentGraph(self.nodes).build().invoke({'messages': [HumanMessage('Проверь web')]})
        self.assertEqual(self.executed, ['broken', 'web'])
        results = [m for m in result['messages'] if isinstance(m, ToolMessage)]
        self.assertEqual([m.status for m in results], ['error', 'success'])
        self.assertIn('App not found', results[0].content)
        self.assertEqual(result['steps'], 3)
        self.assertIsNone(result['stop_reason'])

    def test_missing_tool_results_in_middle_are_not_replayed(self):
        history = [HumanMessage('Проверь web'), AIMessage('', tool_calls=[self.call('a', 'web')]),
                   HumanMessage('Уже другая задача')]
        result = AgentGraph(self.nodes).build().invoke({'messages': history})
        self.assertEqual(result['stop_reason'], 'invalid_history')
        self.assertEqual(self.executed, [])
        self.model.client.invoke.assert_not_called()

    def test_resume_checkpoint_does_not_repeat_saved_result(self):
        self.model.client.invoke.return_value = AIMessage('Готово')
        graph = AgentGraph(self.nodes).build(InMemorySaver())
        config = {'configurable': {'thread_id': 'resume-tools'}}
        history = [HumanMessage('Проверь оба'),
                   AIMessage('', tool_calls=[self.call('a', 'done'), self.call('b', 'web')]),
                   ToolMessage('ready', tool_call_id='a')]
        graph.update_state(config, {'messages': history})
        first = graph.invoke({'messages': []}, config)
        second = graph.invoke({'messages': [HumanMessage('Что получилось?')]}, config)
        self.assertEqual(self.executed, ['web'])
        self.assertEqual(first['steps'], 1)
        self.assertEqual(second['steps'], 1)
        self.assertEqual(len([m for m in second['messages'] if isinstance(m, ToolMessage)]), 2)

    def test_retry_loop_is_bounded_by_steps(self):
        self.model.client.invoke.side_effect = lambda *args, **kwargs: AIMessage(
            '', tool_calls=[self.call(f'call-{self.model.client.invoke.call_count}', 'broken')])
        result = AgentGraph(self.nodes).build().invoke({'messages': [HumanMessage('Проверь web')]})
        self.assertEqual(result['stop_reason'], 'step_limit')
        self.assertEqual(result['steps'], self.config.max_steps)
        self.assertEqual(len(self.executed), self.config.max_steps)

    def test_default_recursion_limit_allows_recovery_then_all_steps(self):
        self.model.client.invoke.side_effect = lambda *args, **kwargs: AIMessage(
            '', tool_calls=[self.call(f'call-{self.model.client.invoke.call_count}', 'web')])
        history = [HumanMessage('Проверь web'), AIMessage('', tool_calls=[self.call('pending', 'web')])]
        result = AgentGraph(self.nodes).build().invoke({'messages': history})
        self.assertEqual(result['stop_reason'], 'step_limit')
        self.assertEqual(len(self.executed), self.config.max_steps + 1)

    def test_stop_reason_has_priority_over_pending_calls(self):
        state = {'stop_reason': 'timeout', 'pending_tool_calls': [self.call('a', 'web')]}
        self.assertEqual(after_limits(state), 'stop')

    def test_corrupt_histories_stop_before_tool_or_model(self):
        call = AIMessage('', tool_calls=[self.call('a', 'web')])
        cases = [[], [ToolMessage('orphan', tool_call_id='a')],
                 [HumanMessage('x'), call, ToolMessage('wrong', tool_call_id='b')],
                 [HumanMessage('x'), call, ToolMessage('ok', tool_call_id='a'), ToolMessage('duplicate', tool_call_id='a')]]
        for history in cases:
            with self.subTest(history=history):
                result = AgentGraph(self.nodes).build().invoke({'messages': history})
                self.assertEqual(result['stop_reason'], 'invalid_history')
                self.assertEqual(self.executed, [])
                self.model.client.invoke.assert_not_called()


if __name__ == '__main__':
    unittest.main()
