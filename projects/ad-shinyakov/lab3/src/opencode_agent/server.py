import os
import signal
import socket
import subprocess
from contextlib import ExitStack
from threading import RLock
from time import monotonic, sleep

import httpx

from .config import LAB_ROOT, OpenCodeConfig
from .gateway import ModelGateway
from .tracing import TurnContext


class OpenCodeServer:
    def __init__(self, config=None):
        self.config = config or OpenCodeConfig()
        self.stack = ExitStack()
        self.lock = RLock()
        self.turns = TurnContext(self.config.runtime / 'turn.json')

    def __enter__(self):
        try:
            gateway = self.stack.enter_context(ModelGateway(self.turns, self.config))
            self.config.prepare(gateway.url)
            self._start()
            self._wait_ready()
            self.request('GET', '/mcp')
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        self.stack.close()

    def _start(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'
        environment = self.config.environment()
        environment.pop('LAB_EVALUATION_STATE', None)
        environment['OPENCODE_TURN_FILE'] = str(self.turns.path)
        log = self.stack.enter_context((self.config.runtime / 'server.log').open('a'))
        self.process = subprocess.Popen([str(self.config.binary), 'serve', '--hostname', '127.0.0.1',
                                         '--port', str(port)], cwd=self.config.runtime / 'workspace',
                                        env=environment, stdout=log, stderr=log, start_new_session=True)
        self.stack.callback(self._stop)
        self.client = self.stack.enter_context(httpx.Client(base_url=self.url, timeout=self.config.timeout_seconds))

    def _wait_ready(self):
        deadline = monotonic() + 45
        while self.process.poll() is None and monotonic() < deadline:
            try:
                if self.client.get('/global/health', timeout=1).is_success:
                    return
            except httpx.HTTPError:
                pass
            sleep(.1)
        raise RuntimeError('OpenCode server не запустился; см. data/opencode/server.log')

    def _stop(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self.turns.clear()

    def request(self, method, path, **values):
        response = self.client.request(method, path, **values)
        response.raise_for_status()
        return response.json() if response.content else None

    def activate(self, context):
        context = {**context, 'evaluation_state': os.environ.get('LAB_EVALUATION_STATE'),
                   'context_file': str(self.turns.path)}
        self.turns.write(context)

    def create_session(self, title):
        return self.request('POST', '/session', json={'title': title})['id']

    def messages(self, session_id):
        messages = self.request('GET', f'/session/{session_id}/message')
        return sorted(messages, key=lambda message: message['info']['id'])

    def send(self, session_id, question):
        body = {'agent': 'kubernetes', 'model': {'providerID': 'ollama', 'modelID': self.config.model},
                'system': (LAB_ROOT / 'opencode/reply.md').read_text(),
                'parts': [{'type': 'text', 'text': question}]}
        return self.request('POST', f'/session/{session_id}/message', json=body)

    def abort(self, session_id):
        return self.request('POST', f'/session/{session_id}/abort', timeout=10)
