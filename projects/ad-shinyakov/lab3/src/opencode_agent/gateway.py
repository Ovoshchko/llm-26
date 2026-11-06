import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import httpx
from .config import OpenCodeConfig

from .tracing import RemoteObservation


class ModelGateway:
    def __init__(self, turns, config=None):
        self.turns = turns
        self.config = config or OpenCodeConfig()

    def __enter__(self):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), ModelRequest)
        self.server.turns = self.turns
        self.server.config = self.config
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1'
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class ModelRequest(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if body.get('tools') and self.server.config.structured_tools:
            body['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'agent_response', 'strict': True, 'schema': self._schema(body['tools'])}}
        with RemoteObservation(self.server.turns.read(), 'ollama.chat', 'generation',
                               input=body['messages'], model=body['model'],
                               metadata={'has_tools': bool(body.get('tools')),
                                         'structured_tools': self.server.config.structured_tools},
                               model_parameters={'temperature': body.get('temperature', 0),
                                                 'max_tokens': body.get('max_tokens', 2048),
                                                 'num_ctx': 32768}) as span:
            try:
                output = self._forward(body)
            except (BrokenPipeError, ConnectionResetError):
                span.update(level='ERROR', status_message='OpenCode closed the connection')
                return
            response = self._completion(output)
            usage = response.get('usage', {})
            span.update(output=response, usage_details={'input': usage.get('prompt_tokens', 0),
                                                        'output': usage.get('completion_tokens', 0)},
                        cost_details={'total': 0})

    def _forward(self, body):
        collected = bytearray()
        with httpx.Client(timeout=self.server.config.timeout_seconds) as client:
            with client.stream('POST', 'http://localhost:11434/v1/chat/completions', json=body) as response:
                self.send_response(response.status_code)
                self.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.close_connection = True
                try:
                    for chunk in response.iter_bytes():
                        collected.extend(chunk)
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
        return bytes(collected)

    @staticmethod
    def _completion(output):
        text = output.decode()
        if not text.startswith('data:'):
            return json.loads(text)
        events = [json.loads(line[5:]) for line in text.splitlines()
                  if line.startswith('data:') and line[5:].strip() != '[DONE]']
        return {'stream': events, 'usage': next((event['usage'] for event in reversed(events)
                                                if event.get('usage')), {})}

    @staticmethod
    def _schema(tools):
        choices = [{'type': 'object', 'properties': {'answer': {'type': 'string', 'minLength': 1}},
                    'required': ['answer'], 'additionalProperties': False}]
        for tool in tools:
            function = tool['function']
            choices.append({'type': 'object', 'properties': {
                'name': {'const': function['name']}, 'arguments': function['parameters']},
                'required': ['name', 'arguments'], 'additionalProperties': False})
        return {'oneOf': choices}

    def log_message(self, format, *args):
        return
