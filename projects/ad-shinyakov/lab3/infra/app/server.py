import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


class Handler(BaseHTTPRequestHandler):
    config_version = os.environ.get('CONFIG_VERSION', 'v1')

    def do_GET(self):
        if urlsplit(self.path).path not in ('/', '/health', '/ready'):
            self.send_error(404)
            return
        body = json.dumps({'status': 'ok', 'config_version': self.config_version}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    print(f'config_version={Handler.config_version}', flush=True)
    ThreadingHTTPServer(('0.0.0.0', 8080), Handler).serve_forever()
