import json
import os
from contextlib import nullcontext
from pathlib import Path

from langfuse import get_client


class ToolRecording:
    def __init__(self, path=None):
        path = path or os.environ.get('LAB_EVALUATION_STATE')
        self.path = Path(path) if path else None
        if self.path and not self.path.is_absolute():
            self.path = Path(__file__).resolve().parents[5] / self.path

    def execute(self, name, arguments, action):
        client = get_client()
        observation = client.start_as_current_observation(name='retrieval', as_type='retriever', input=arguments,
                                                          metadata={'source': 'lab2'}) if name == 'search_docs' else nullcontext()
        result = None
        try:
            with observation as span:
                result = action()
                if span:
                    span.update(output=result)
            self._fault(name)
            self._record(name, arguments, result)
            return result
        except Exception as error:
            self._record(name, arguments, result, str(error))
            raise

    def _record(self, name, arguments, result, error=None):
        if self.path:
            state = json.loads(self.path.read_text())
            entry = {'turn': state['turn'], 'name': name, 'arguments': arguments,
                     'result': result, 'error': error}
            with self.path.with_suffix('.jsonl').open('a') as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def _fault(self, name):
        if not self.path or name != 'restart_deployment':
            return
        state = json.loads(self.path.read_text())
        if state['scenario'] == 'timeout_after_restart' and not state.get('fault_injected'):
            state['fault_injected'] = True
            self.path.write_text(json.dumps(state))
            raise TimeoutError('Injected timeout after Kubernetes accepted restart; inspect before retrying')

    def entries(self):
        if self.path is None:
            return []
        path = self.path.with_suffix('.jsonl')
        return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []
