import unittest
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent import AgentConfig, AgentGraph, AgentNodes
from agent.routing import after_limits
from agent.context import ContextManager


class LoopNodes(AgentNodes):
    def __init__(self, config, tool_seconds=0):
        self.now = 0.0
        self.tool_seconds = tool_seconds
        self.calls = []
        super().__init__(config, clock=lambda: self.now)

    def prepare_context(self, state):
        self.calls.append('context')
        return {}

    def call_model(self, state):
        self.calls.append('model')
        step = state['steps'] + 1
        if state['messages'][-1].content == 'finish':
            response = AIMessage(content='Готово')
        else:
            response = AIMessage(content='', tool_calls=[{'id': f'call-{step}', 'name': 'test', 'args': {}}])
        return {'steps': step, 'messages': [response]}

    def call_tools(self, state):
        self.calls.append('tools')
        self.now += self.tool_seconds
        return {'messages': [ToolMessage(content='result', tool_call_id=state['messages'][-1].tool_calls[0]['id'])]}


class LimitsTests(unittest.TestCase):
    def test_boundaries_and_timeout_precedence(self):
        now = [100.0]
        nodes = AgentNodes(AgentConfig(max_steps=2, timeout_seconds=3), clock=lambda: now[0])
        state = {**nodes.begin_turn({'messages': []}), 'messages': [HumanMessage('test')]}
        for steps, elapsed, reason in [(0, 0, None), (1, 2.99, None), (2, 2.99, 'step_limit'),
                                       (0, 3, 'timeout'), (2, 3, 'timeout')]:
            with self.subTest(steps=steps, elapsed=elapsed):
                now[0] = 100 + elapsed
                update = nodes.check_state({**state, 'steps': steps})
                self.assertEqual(update['stop_reason'], reason)
                self.assertEqual(after_limits({**state, **update}), 'stop' if reason else 'context')

    def test_existing_stop_reason_is_preserved(self):
        nodes = AgentNodes(clock=lambda: 10)
        state = {**nodes.begin_turn({'messages': []}), 'stop_reason': 'tool_error'}
        self.assertEqual(nodes.check_state(state), {})
        self.assertEqual(after_limits(state), 'stop')

    def test_step_limit_prevents_another_model_and_tool_call(self):
        nodes = LoopNodes(AgentConfig(max_steps=2, timeout_seconds=10))
        state = AgentGraph(nodes).build().invoke({'messages': [('user', 'loop')]})
        self.assertEqual(state['stop_reason'], 'step_limit')
        self.assertEqual(state['steps'], 2)
        self.assertEqual(nodes.calls, ['context', 'model', 'tools'] * 2)

    def test_timeout_prevents_next_iteration(self):
        nodes = LoopNodes(AgentConfig(max_steps=5, timeout_seconds=3), tool_seconds=3)
        state = AgentGraph(nodes).build().invoke({'messages': [('user', 'loop')]})
        self.assertEqual(state['stop_reason'], 'timeout')
        self.assertEqual(state['steps'], 1)
        self.assertEqual(nodes.calls, ['context', 'model', 'tools'])

    def test_default_limit_is_not_preempted_by_langgraph_recursion_limit(self):
        nodes = LoopNodes(AgentConfig())
        state = AgentGraph(nodes).build().invoke({'messages': [('user', 'loop')]})
        self.assertEqual(state['stop_reason'], 'step_limit')
        self.assertEqual(state['steps'], AgentConfig().max_steps)
        self.assertEqual(nodes.calls.count('model'), AgentConfig().max_steps)

    def test_next_turn_resets_limits_and_preserves_history(self):
        nodes = LoopNodes(AgentConfig(max_steps=1, timeout_seconds=3), tool_seconds=3)
        graph = AgentGraph(nodes).build(checkpointer=InMemorySaver())
        config = {'configurable': {'thread_id': 'dialogue'}}
        first = graph.invoke({'messages': [('user', 'loop')]}, config)
        self.assertEqual(first['stop_reason'], 'timeout')
        old_ids = [m.id for m in first['messages']]
        nodes.now = 100
        second = graph.invoke({'messages': [('user', 'finish')]}, config)
        self.assertIsNone(second['stop_reason'])
        self.assertEqual(second['steps'], 1)
        self.assertEqual(second['started_at'], 100)
        self.assertEqual([m.id for m in second['messages'][:len(old_ids)]], old_ids)
        self.assertEqual(second['messages'][-1].content, 'Готово')

    def test_unknown_tool_returns_error_to_model(self):
        class Counter:
            def count(self, messages, tools=()):
                return len(messages)

        context = ContextManager(AgentConfig(), token_counter=Counter())
        model = Mock(tools=[])
        model.invoke.side_effect = [
            AIMessage('', tool_calls=[{'id': '1', 'name': 'test', 'args': {}}]), AIMessage('Недоступный инструмент')]
        result = AgentGraph(AgentNodes(context_manager=context, model=model)).build().invoke(
            {'messages': [HumanMessage(content='test')]})
        self.assertEqual(result['messages'][-2].status, 'error')
        self.assertEqual(result['steps'], 2)

    def test_invalid_configuration(self):
        for max_steps in [0, -1, True, 1.5]:
            with self.subTest(max_steps=max_steps), self.assertRaises(ValueError):
                AgentConfig(max_steps=max_steps)
        for timeout in [0, -1, True, float('inf'), float('nan')]:
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                AgentConfig(timeout_seconds=timeout)


if __name__ == '__main__':
    unittest.main()
