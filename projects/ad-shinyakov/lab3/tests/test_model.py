import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agent import AgentConfig, AgentGraph, AgentNodes, ContextManager
from agent.model import OllamaModel


@tool(description='Прочитать состояние Deployment без изменений.')
def inspect_workload(namespace: str, name: str) -> dict:
    return {'namespace': namespace, 'name': name, 'ready_replicas': 1}


class Counter:
    def count(self, messages, tools=()):
        return len(messages) + len(tools)


class ModelTests(unittest.TestCase):
    def test_adapter_binds_tools_and_passes_generation_limits(self):
        config = AgentConfig(model_context_window=4096, max_output_tokens=256, timeout_seconds=12)
        with patch('agent.model.ChatOllama') as client:
            model = OllamaModel(config, [inspect_workload])
            client.return_value.bind_tools.return_value.invoke.assert_not_called()
            response = AIMessage('done')
            bound = client.return_value.bind_tools.return_value
            bound.invoke.return_value = response
            messages = [HumanMessage('test')]
            runnable_config = {'tags': ['model-test']}
            self.assertIs(model.invoke(messages, runnable_config), response)
            self.assertIs(model.invoke(messages, runnable_config), response)
            client.assert_called_once_with(
                model=config.model, base_url='http://localhost:11434', temperature=0,
                num_ctx=4096, num_predict=256, client_kwargs={'timeout': 12}, format=model.response_format.schema())
            client.return_value.bind_tools.assert_called_once_with(model.tools)
            self.assertEqual(model.tools[0]['function']['name'], 'inspect_workload')
            bound.invoke.assert_called_with(messages, config=runnable_config)

    def test_model_without_tools_does_not_bind_empty_list(self):
        with patch('agent.model.ChatOllama') as client:
            client.return_value.invoke.return_value = AIMessage('answer')
            result = OllamaModel(AgentConfig()).invoke([HumanMessage('hi')])
            self.assertEqual(result.content, 'answer')
            client.return_value.bind_tools.assert_not_called()

    def test_node_uses_prepared_context_and_preserves_model_metadata(self):
        model = Mock(tools=[])
        response = AIMessage('answer', usage_metadata={'input_tokens': 12, 'output_tokens': 3, 'total_tokens': 15})
        model.invoke.return_value = response
        nodes = AgentNodes(model=model)
        prepared = [SystemMessage('rules'), HumanMessage('now')]
        result = nodes.call_model({'messages': [HumanMessage('old')], 'model_messages': prepared, 'steps': 2}, {})
        model.invoke.assert_called_once_with(prepared, config={})
        self.assertEqual(result['steps'], 3)
        self.assertIs(result['messages'][0], response)
        self.assertEqual(result['messages'][0].usage_metadata['total_tokens'], 15)

    def test_unprepared_context_never_reaches_model(self):
        model = Mock(tools=[])
        with self.assertRaisesRegex(ValueError, 'Prepare model_messages'):
            AgentNodes(model=model).call_model({'messages': [HumanMessage('test')], 'steps': 0}, {})
        model.invoke.assert_not_called()

    def test_provisional_text_is_discarded_when_model_calls_tools(self):
        model = OllamaModel(AgentConfig())
        response = AIMessage('Непроверенное состояние', tool_calls=[
            {'id': 'a', 'name': 'resources_get', 'args': {'name': 'api'}}])
        model.client = Mock()
        model.client.invoke.return_value = response
        result = model.invoke([HumanMessage('Проверь api')])
        self.assertEqual(result.content, '')
        self.assertEqual(result.tool_calls, response.tool_calls)
        self.assertEqual(result.id, response.id)
        self.assertEqual(response.content, 'Непроверенное состояние')

    def test_structured_final_answer_is_displayed_as_text(self):
        model = OllamaModel(AgentConfig(), [inspect_workload])
        model.client = Mock()
        response = AIMessage('{"answer":"Готовы две реплики."}', response_metadata={'model': 'glm4-agent:9b'})
        model.client.invoke.return_value = response
        result = model.invoke([HumanMessage('Проверь api')])
        self.assertEqual(result.content, 'Готовы две реплики.')
        self.assertEqual(result.response_metadata, response.response_metadata)
        self.assertFalse(result.tool_calls)

    def test_json_command_in_content_is_not_converted_to_a_call(self):
        model = OllamaModel(AgentConfig(), [inspect_workload])
        model.client = Mock()
        model.client.invoke.return_value = AIMessage('{"name":"inspect_workload","arguments":{"name":"api"}}')
        with self.assertRaises(ValueError):
            model.invoke([HumanMessage('Проверь api')])

    def test_complete_graph_cycle_with_library_tool_executor(self):
        model = OllamaModel(AgentConfig(), [inspect_workload])
        fake_client = Mock()
        fake_client.invoke.side_effect = [
            AIMessage('', tool_calls=[{'id': 'call-1', 'name': 'inspect_workload',
                                      'args': {'namespace': 'agent-lab', 'name': 'web'}}]),
            AIMessage('Готова одна реплика.'),
        ]
        model.client = fake_client
        context = ContextManager(AgentConfig(), token_counter=Counter(), tools=model.tools)
        graph = AgentGraph(AgentNodes(model=model, context_manager=context, tools=[inspect_workload])).build()
        state = graph.invoke({'messages': [HumanMessage('Проверь web')]})
        self.assertIsNone(state['stop_reason'])
        self.assertEqual(state['steps'], 2)
        self.assertEqual(state['messages'][-1].content, 'Готова одна реплика.')
        observation = state['messages'][-2]
        self.assertIsInstance(observation, ToolMessage)
        self.assertEqual(observation.tool_call_id, 'call-1')
        self.assertIn('ready_replicas', observation.content)
        second_input = fake_client.invoke.call_args.args[0]
        self.assertEqual(second_input[-1], observation)
        self.assertEqual(context.tools, model.tools)

    def test_context_schema_mismatch_is_rejected(self):
        model = OllamaModel(AgentConfig(), [inspect_workload])
        context = ContextManager(AgentConfig(), token_counter=Counter())
        with self.assertRaisesRegex(ValueError, 'same tool schemas'):
            AgentNodes(model=model, context_manager=context)

    def test_invalid_response_is_not_treated_as_final_answer(self):
        for response in [HumanMessage('bad'), AIMessage('', invalid_tool_calls=[
                {'name': 'inspect_workload', 'args': '{bad', 'id': 'a', 'error': 'invalid JSON'}])]:
            with self.subTest(response=response):
                model = OllamaModel(AgentConfig())
                model.client = Mock()
                model.client.invoke.return_value = response
                with self.assertRaises((TypeError, ValueError)):
                    model.invoke([HumanMessage('test')])

    def test_network_error_is_not_retried_or_fabricated(self):
        model = OllamaModel(AgentConfig())
        model.client = Mock()
        model.client.invoke.side_effect = ConnectionError('Ollama unavailable')
        with self.assertRaises(ConnectionError):
            model.invoke([HumanMessage('test')])
        self.assertEqual(model.client.invoke.call_count, 1)


if __name__ == '__main__':
    unittest.main()
