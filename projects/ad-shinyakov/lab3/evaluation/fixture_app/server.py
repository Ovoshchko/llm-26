import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from time import sleep


class Handler(BaseHTTPRequestHandler):
    version = os.environ.get('CONFIG_VERSION', 'v1')
    ready = os.environ.get('ALLOW_READY', 'true') == 'true'

    def do_GET(self):
        if self.path not in ('/', '/ready', '/health'):
            self.send_error(404)
            return
        body = json.dumps({'config_version': self.version}).encode()
        self.send_response(503 if self.path == '/ready' and not self.ready else 200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        if args[1] != '200':
            print(format % args, flush=True)


def report_config():
    while True:
        print(f'config_version={Handler.version} ready={Handler.ready}', flush=True)
        sleep(5)


if __name__ == '__main__':
    if os.environ.get('REQUIRE_ENDPOINT') == 'true' and not os.environ.get('CONFIG_ENDPOINT'):
        print('FATAL: CONFIG_ENDPOINT is required', flush=True)
        sys.exit(1)
    print('HTTP server started: health endpoint=/ready', flush=True)
    Thread(target=report_config, daemon=True).start()
    ThreadingHTTPServer(('0.0.0.0', 8080), Handler).serve_forever()
