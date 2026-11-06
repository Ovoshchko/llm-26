import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from anyio.from_thread import start_blocking_portal
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError
from agent import AgentConfig, AgentGraph, AgentNodes, ContextManager, OllamaModel

from agent.tools.config import ToolConfig
from agent.tools.deployments import DeploymentActions
from agent.tools.mcp import KubernetesMCP
from agent.tools.retrieval import DocumentationSearch
from agent.tools.resource_summary import ResourceSummary
from langchain_core.tools import StructuredTool


class AgentToolTests(unittest.TestCase):
    def test_resource_summary_preserves_status_and_spec_without_duplicate_metadata(self):
        source = ('kind: Deployment\nmetadata:\n  name: api\n  resourceVersion: "7"\n'
                  '  annotations:\n    kubectl.kubernetes.io/last-applied-configuration: old snapshot\n'
                  '    kubectl.kubernetes.io/restartedAt: now\n'
                  'spec:\n  replicas: 2\nstatus:\n  readyReplicas: 2\n')
        result = ResourceSummary().format('resources_get', source)
        self.assertIn('"readyReplicas":2', result)
        self.assertIn('"replicas":2', result)
        self.assertIn('"resourceVersion":"7"', result)
        self.assertIn('restartedAt', result)
        self.assertNotIn('old snapshot', result)
        self.assertLess(result.index('"status"'), result.index('"spec"'))
        self.assertEqual(ResourceSummary().format('pods_log', source), source)

    def test_scale_validation_blocks_invalid_arguments_before_execution(self):
        client = Mock()
        action = DeploymentActions(client)
        tool = StructuredTool.from_function(action.scale, name='scale_deployment', description='scale')
        for arguments in [{'name': 'api', 'replicas': 6}, {'name': 'api', 'replicas': 0},
                          {'name': 'api', 'replicas': 2, 'namespace': 'kube-system'},
                          {'name': 'api; rm', 'replicas': 2}]:
            with self.subTest(arguments=arguments), self.assertRaises(ValidationError):
                tool.invoke(arguments)
        client.call.assert_not_called()

    @staticmethod
    def resource(status=None):
        return {'apiVersion': 'apps/v1', 'kind': 'Deployment',
                'metadata': {'name': 'api', 'namespace': 'agent-lab', 'generation': 3, 'resourceVersion': '7'},
                'spec': {'replicas': 2, 'template': {'spec': {'containers': [{'name': 'api', 'image': 'app:v1'}]}}},
                'status': status or {'observedGeneration': 3, 'replicas': 2, 'updatedReplicas': 2,
                                     'readyReplicas': 2, 'availableReplicas': 2}}

    def test_changes_use_mcp_and_require_a_separate_rollout_check(self):
        mcp = Mock()
        mcp.call.side_effect = lambda tool_name, **arguments: json.dumps(self.resource()) if tool_name == 'resources_get' else 'accepted'
        actions = DeploymentActions(mcp)
        for result in (actions.restart('api'), actions.scale('api', 2)):
            self.assertTrue(result['accepted'])
            self.assertTrue(result['requires_rollout_check'])
            self.assertNotIn('complete', result)
        self.assertTrue(actions.rollout('api')['complete'])
        mcp.call.assert_any_call('resources_scale', apiVersion='apps/v1', kind='Deployment',
                                 name='api', namespace='agent-lab', scale=2)
        update = next(call for call in mcp.call.call_args_list if call.args[0] == 'resources_create_or_update')
        resource = json.loads(update.kwargs['resource'])
        self.assertNotIn('status', resource)
        self.assertEqual(resource['metadata']['resourceVersion'], '7')
        self.assertEqual(resource['spec']['template']['spec'], self.resource()['spec']['template']['spec'])
        self.assertIn('kubectl.kubernetes.io/restartedAt', resource['spec']['template']['metadata']['annotations'])

    def test_rollout_requires_current_generation_and_all_replicas(self):
        mcp = Mock()
        actions = DeploymentActions(mcp)
        for change in ({'observedGeneration': 2}, {'replicas': 3}, {'updatedReplicas': 1},
                       {'readyReplicas': 1}, {'availableReplicas': 1}, {}):
            with self.subTest(change=change):
                resource = self.resource()
                resource['status'].update(change)
                mcp.call.return_value = json.dumps(resource)
                self.assertEqual(actions.rollout('api')['complete'], not bool(change))

    def test_failed_mcp_write_is_not_reported_as_accepted_or_retried(self):
        mcp = Mock()
        mcp.call.side_effect = [json.dumps(self.resource()), TimeoutError('Unknown write result')]
        with self.assertRaisesRegex(TimeoutError, 'Unknown write result'):
            DeploymentActions(mcp).restart('api')
        self.assertEqual(mcp.call.call_count, 2)

    def test_retrieval_preserves_sources_and_bounds_passage_size(self):
        search = DocumentationSearch(ToolConfig())
        search.retriever = Mock()
        search.retriever.retrieve.return_value = [
            {'chunk_id': 'a', 'text': 'x' * 4000, 'score': 0.8, 'source_url': 'https://kubernetes.io/docs/'}]
        result = search.search('question', 3)
        passage = result['passages'][0]
        self.assertEqual(passage['chunk_id'], 'a')
        self.assertEqual(passage['score'], 0.8)
        self.assertEqual(len(passage['text']), 2000)
        self.assertTrue(passage['truncated'])
        search.retriever.retrieve.assert_called_once_with('question', 3)

    def test_mcp_bridge_returns_plain_text_and_errors_reach_tool_node(self):
        remote = SimpleNamespace(name='resources_get', args_schema={
            'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']},
            coroutine=AsyncMock(return_value=([{'type': 'text', 'text': 'ready: 2'}], None)))
        bridge = KubernetesMCP(ToolConfig())
        with start_blocking_portal() as portal:
            bridge.portal = portal
            bridge.remote_tools = {'resources_get': remote}
            tool = bridge._wrap(remote)
            result = tool.invoke({'name': 'api'})
            self.assertEqual(result, 'ready: 2')
            self.assertEqual(tool.metadata['mcp_server'], 'kubernetes')
            remote.coroutine.side_effect = TimeoutError('MCP unavailable')
            model = OllamaModel(AgentConfig(), [tool])
            model.client = Mock()
            model.client.invoke.side_effect = [AIMessage('', tool_calls=[
                {'name': 'resources_get', 'args': {'name': 'api'}, 'id': 'a'}]), AIMessage('Ошибка обработана')]
            counter = Mock()
            counter.count.side_effect = lambda messages, tools: len(messages)
            context = ContextManager(AgentConfig(), token_counter=counter, tools=model.tools)
            graph = AgentGraph(AgentNodes(model=model, context_manager=context, tools=[tool])).build()
            result = graph.invoke({'messages': [HumanMessage('Проверь api')]})
            self.assertEqual(result['messages'][-2].status, 'error')
            self.assertIn('MCP unavailable', result['messages'][-2].content)
            self.assertEqual(result['messages'][-1].content, 'Ошибка обработана')

    def test_mcp_limits_model_output_without_rejecting_internal_write_acknowledgements(self):
        payload = 'x' * 10000
        bridge = KubernetesMCP(ToolConfig())
        bridge.remote_tools = {
            tool_name: SimpleNamespace(coroutine=AsyncMock(return_value=(
                [{'type': 'text', 'text': payload}], None)))
            for tool_name in ('resources_get', 'resources_create_or_update')}
        with start_blocking_portal() as portal:
            bridge.portal = portal
            with self.assertRaisesRegex(ValueError, 'context budget'):
                bridge.call('resources_get', name='api')
            self.assertEqual(bridge.call('resources_create_or_update', resource='{}'), payload)


if __name__ == '__main__':
    unittest.main()
