import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
import unittest
from unittest.mock import patch

from opencode_agent.config import OpenCodeConfig
from opencode_agent.session import OpenCodeSession
from opencode_agent.tracing import RemoteObservation, TurnContext
from benchmark.recording import ToolRecording


class ServerContextTests(unittest.TestCase):
    def test_new_turn_never_inherits_previous_generation_parent(self):
        with TemporaryDirectory() as directory:
            context = TurnContext(Path(directory) / 'turn.json')
            context.write({'trace_id': 'first', 'parent_span_id': 'root1', 'generation_id': 'gen1'})
            self.assertEqual(RemoteObservation._parent(context.read(), 'tool')['parent_span_id'], 'gen1')
            context.write({'trace_id': 'second', 'parent_span_id': 'root2'})
            with patch.dict(os.environ, {'OPENCODE_TURN_FILE': str(context.path)}):
                self.assertEqual(TurnContext.current()['trace_id'], 'second')
                self.assertEqual(RemoteObservation._parent(TurnContext.current(), 'tool')['parent_span_id'], 'root2')
            context.clear()
            self.assertEqual(context.read(), {})

    @patch('benchmark.recording.get_client')
    def test_explicit_current_task_overrides_stale_process_environment(self, client):
        with TemporaryDirectory() as directory:
            first, second = [Path(directory) / name for name in ('old.json', 'current.json')]
            for path in (first, second):path.write_text('{"scenario":"healthy","turn":2}')
            with patch.dict(os.environ, {'LAB_EVALUATION_STATE': str(first)}):
                ToolRecording(second).execute('resources_get', {}, lambda: 'ready')
            self.assertFalse(first.with_suffix('.jsonl').exists())
            self.assertEqual(ToolRecording(second).entries()[0]['result'], 'ready')

    def test_compaction_counts_toward_native_limit_but_not_primary_steps(self):
        session = OpenCodeSession.__new__(OpenCodeSession)
        session.config = OpenCodeConfig(max_steps=2)
        session.session_id = 'session'
        events = [{'type': 'step_finish', 'agent': 'compaction'},
                  {'type': 'step_finish', 'agent': 'kubernetes'},
                  {'type': 'text', 'part': {'text': 'Ответ'}}]
        result = session._result('question', events, {'timeout': False}, monotonic())
        self.assertEqual(result['steps'], 1)
        self.assertTrue(result['step_limit_reached'])

    def test_provider_failure_never_returns_earlier_progress_as_final_answer(self):
        session = OpenCodeSession.__new__(OpenCodeSession)
        session.config = OpenCodeConfig()
        session.session_id = 'session'
        result = session._result('question', [{'type': 'text', 'part': {'text': 'Начинаю'}}],
                                 {'timeout': False, 'error': 'provider unavailable'}, monotonic())
        self.assertEqual(result['answer'], '')
        self.assertEqual(result['stop_reason'], 'provider_error')


if __name__ == '__main__':
    unittest.main()
