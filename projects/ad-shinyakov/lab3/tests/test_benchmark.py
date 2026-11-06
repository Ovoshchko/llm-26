import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from benchmark.metrics import RetrievalMetrics, ToolMetrics, TraceMetrics
from benchmark.recording import ToolRecording
from benchmark.finalize import ExperimentFinalizer
from opencode_agent.gateway import ModelRequest
from opencode_agent.session import OpenCodeSession


class OpenCodeProtocolTests(unittest.TestCase):
    def test_schema_keeps_required_tool_arguments(self):
        parameters = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}
        schema = ModelRequest._schema([{'function': {'name': 'lab_get', 'parameters': parameters}}])
        call = schema['oneOf'][1]
        self.assertEqual(call['properties']['name'], {'const': 'lab_get'})
        self.assertEqual(call['properties']['arguments'], parameters)
        self.assertFalse(call['additionalProperties'])

    def test_answer_does_not_include_intermediate_model_text(self):
        events = [{'type': 'text', 'part': {'text': 'Проверяю'}},
                  {'type': 'text', 'part': {'text': '{"answer":"Готовы 2 реплики"}'}}]
        self.assertEqual(OpenCodeSession.answer(events), 'Готовы 2 реплики')

    def test_json_command_is_never_executed_by_answer_decoder(self):
        text = '{"name":"restart_deployment","arguments":{}}'
        self.assertEqual(OpenCodeSession.answer([{'type': 'text', 'part': {'text': text}}]), text)

    def test_api_history_preserves_tool_events_and_ignores_user_text(self):
        messages = [{'info': {'role': 'user'}, 'parts': [{'type': 'text', 'text': 'question'}]},
                    {'info': {'role': 'assistant', 'sessionID': 'ses', 'agent': 'kubernetes',
                              'time': {'created': 10, 'completed': 20}},
                     'parts': [{'type': 'tool', 'tool': 'lab_get', 'state': {'status': 'completed'}},
                               {'type': 'text', 'text': 'Ответ'}]}]
        events = OpenCodeSession.events(messages)
        self.assertEqual([event['type'] for event in events], ['tool_use', 'text'])
        self.assertEqual(events[0]['timestamp'], 20)
        self.assertEqual(OpenCodeSession.answer(events), 'Ответ')

    def test_stream_usage_is_read_from_final_event(self):
        stream = (b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                  b'data: {"choices":[],"usage":{"prompt_tokens":42,"completion_tokens":3}}\n\n'
                  b'data: [DONE]\n\n')
        self.assertEqual(ModelRequest._completion(stream)['usage']['prompt_tokens'], 42)


class EvaluationMetricTests(unittest.TestCase):
    def test_retrieval_deduplicates_documents(self):
        entries = [{'name': 'search_docs', 'result': {'passages': [
            {'document_path': 'a.md'}, {'document_path': 'a.md'}, {'document_path': 'b.md'}]}}]
        result = RetrievalMetrics().calculate(entries, ['b.md', 'c.md'])
        self.assertEqual(result['retrieval_recall_at_3'], .5)
        self.assertEqual(result['retrieval_precision_at_3'], 1 / 3)
        self.assertEqual(result['retrieval_mrr'], .5)

    def test_missing_retrieval_is_zero_with_references(self):
        self.assertEqual(RetrievalMetrics().calculate([], ['a.md'])['retrieval_recall_at_3'], 0)
        self.assertIsNone(RetrievalMetrics().calculate([], [])['retrieval_recall_at_3'])

    def test_lab2_source_path_matches_document_reference(self):
        entries = [{'name': 'search_docs', 'result': {'passages': [
            {'source_path': 'content/en/docs/concepts/configuration/configmap.md'}]}}]
        result = RetrievalMetrics().calculate(entries, ['docs/concepts/configuration/configmap.md'])
        self.assertEqual(result['retrieval_recall_at_3'], 1)
        self.assertEqual(result['retrieval_mrr'], 1)

    def test_tool_aliases_match_mcp_names(self):
        entries = [{'name': 'resources_get', 'result': {}, 'error': None}]
        result = ToolMetrics().calculate(entries, ['inspect_workload', 'get_pod_logs'])
        self.assertEqual(result['tool_correctness'], .5)

    @patch('benchmark.metrics.get_client')
    def test_trace_metrics_use_api_serialized_fields(self, client):
        observation = Mock()
        observation.model_dump.return_value = {'isRootObservation': True, 'type': 'GENERATION',
                                               'usageDetails': {'input': 100, 'output': 5},
                                               'costDetails': {'total': 0}}
        missing = Mock()
        missing.model_dump.return_value = {'type': 'GENERATION', 'usageDetails': {'input': 0, 'output': 0}}
        client.return_value.api.observations.get_many.return_value.data = [observation, missing]
        client.return_value.api.observations.get_many.return_value.meta.cursor = None
        result = TraceMetrics().read('trace')
        self.assertEqual(result['input_tokens'], 100)
        self.assertEqual(result['output_tokens'], 5)
        self.assertEqual(result['generations_without_usage'], 1)
        observation.model_dump.assert_called_with(mode='json')

    @patch('benchmark.metrics.get_client')
    def test_trace_metrics_include_generations_beyond_first_page(self, client):
        first, second = Mock(), Mock()
        first.model_dump.return_value = {'type': 'GENERATION', 'usageDetails': {'input': 100, 'output': 5}}
        second.model_dump.return_value = {'isRootObservation': True, 'type': 'GENERATION',
                                          'usageDetails': {'input': 200, 'output': 10}}
        page1, page2 = Mock(), Mock()
        page1.data, page1.meta.cursor = [first], 'next-page'
        page2.data, page2.meta.cursor = [second], None
        client.return_value.api.observations.get_many.side_effect = [page1, page2]
        result = TraceMetrics().read('trace')
        self.assertEqual(result['generations'], 2)
        self.assertEqual(result['input_tokens'], 300)
        self.assertEqual(result['output_tokens'], 15)
        self.assertEqual(client.return_value.api.observations.get_many.call_args.kwargs['cursor'], 'next-page')


class FaultInjectionTests(unittest.TestCase):
    @patch('benchmark.recording.get_client')
    def test_timeout_after_apply_records_one_accepted_mutation(self, client):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            path.write_text(json.dumps({'scenario': 'timeout_after_restart', 'turn': 1}))
            with patch.dict(os.environ, {'LAB_EVALUATION_STATE': str(path)}):
                recording = ToolRecording()
                action = Mock(return_value={'accepted': True})
                with self.assertRaises(TimeoutError):
                    recording.execute('restart_deployment', {'name': 'api'}, action)
                entries = recording.entries()
                self.assertEqual(len(entries), 1)
                self.assertEqual(entries[0]['result'], {'accepted': True})
                self.assertTrue(entries[0]['error'])
                action.assert_called_once()
                self.assertEqual(recording.execute('restart_deployment', {}, action), {'accepted': True})


class ReviewValidationTests(unittest.TestCase):
    def test_review_cannot_override_failed_state_check(self):
        finalizer = ExperimentFinalizer.__new__(ExperimentFinalizer)
        finalizer.reviews = {'own/a01/1': {'trace_id': 'trace', 'success': True}}
        turn = {'trace_id': 'trace', 'evaluation': {'success': False}, 'state_checks': [{'passed': False}]}
        with self.assertRaisesRegex(ValueError, 'обязательным проверкам'):
            finalizer._review('own', 'a01', 1, turn)

    def test_review_is_bound_to_trace_id(self):
        finalizer = ExperimentFinalizer.__new__(ExperimentFinalizer)
        finalizer.reviews = {'own/a01/1': {'trace_id': 'old-trace', 'success': True}}
        with self.assertRaisesRegex(ValueError, 'другому трейсу'):
            finalizer._review('own', 'a01', 1, {'trace_id': 'new-trace'})

    def test_unknown_compaction_does_not_count_as_before(self):
        from benchmark.metrics import RunMetrics

        turn = {'evaluation': {'success': True}, 'history_dependent': True, 'compressed': None,
                'latency_seconds': 1, 'metrics': {'cost_usd': 0, 'input_tokens': 1, 'output_tokens': 1, 'tool_errors': 0}}
        task = {'type': 'dialogue', 'success': True, 'turns': [turn]}
        metrics = RunMetrics().calculate([task])
        self.assertIsNone(metrics['history_before_compression'])

    def test_review_updates_primary_score_and_preserves_automatic_verdict(self):
        finalizer = ExperimentFinalizer.__new__(ExperimentFinalizer)
        finalizer.reviews = {'own/a01/1': {'trace_id': 'trace', 'success': False,
                                           'reason': 'Вызова логов не было', 'method': 'trace_review'}}
        scores = {}
        finalizer._score = lambda turn, name, value: scores.update({name: value})
        turn = {'trace_id': 'trace', 'evaluation': {'success': True, 'reason': 'Успех'},
                'metrics': {}, 'history_dependent': False}
        finalizer._review('own', 'a01', 1, turn)
        self.assertEqual(scores['task_success'], 0)
        self.assertEqual(scores['task_success_reviewed'], 0)
        self.assertEqual(scores['task_success_automatic'], 1)
        self.assertTrue(turn['automatic_evaluation']['success'])

    def test_compaction_is_detected_at_the_correct_turn(self):
        history = {'messages': [{'info': {'role': 'assistant', 'summary': True,
                                         'time': {'created': 100, 'completed': 105}},
                                  'parts': [{'type': 'text', 'text': 'Сохранённая задача'}]}]}
        self.assertFalse(ExperimentFinalizer._compacted(history, {'events': [{'timestamp': 90}]}))
        self.assertTrue(ExperimentFinalizer._compacted(history, {'events': [{'timestamp': 110}]}))
        counts = ExperimentFinalizer._compaction_metrics(history, {'events': [{'timestamp': 110}]}, {})
        self.assertEqual(counts['compression_successes'], 1)
        counts = ExperimentFinalizer._compaction_metrics(history, {'events': [{'timestamp': 120}]},
                                                         {'events': [{'timestamp': 110}]})
        self.assertEqual(counts['compression_attempts'], 0)
        history['messages'][0]['info'].update(role='user', summary={'title': 'Диагностика'})
        self.assertFalse(ExperimentFinalizer._compacted(history, {'events': [{'timestamp': 110}]}))

    def test_started_or_failed_compaction_is_not_successful_compression(self):
        message = {'info': {'role': 'assistant', 'summary': True, 'time': {'created': 100}},
                   'parts': []}
        history = {'messages': [message]}
        turn = {'events': [{'timestamp': 200}]}
        self.assertFalse(ExperimentFinalizer._compacted(history, turn))
        message['info']['time']['completed'] = 150
        message['info']['error'] = {'name': 'TimeoutError'}
        message['parts'] = [{'type': 'text', 'text': 'Неполный ответ'}]
        self.assertFalse(ExperimentFinalizer._compacted(history, turn))


if __name__ == '__main__':
    unittest.main()
