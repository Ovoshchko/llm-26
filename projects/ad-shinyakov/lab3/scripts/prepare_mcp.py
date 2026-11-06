import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.request import urlopen

import yaml


class MCPSetup:
    version = '0.1.2'
    digest = 'e450c82dc2d6dec5a4a7fba875d6fb84100e2a68635663044b295c1996bbb82b'

    def __init__(self):
        self.root = Path(__file__).resolve().parents[1]
        self.binary = self.root / 'data/bin/kubernetes-mcp-server'
        self.kubeconfig = self.root / 'data/cluster/kubeconfig.yaml'

    def run(self):
        self._install_binary()
        self._write_credentials()
        print(f'MCP {self.version}: {self.binary}')
        print('ServiceAccount kubeconfig подготовлен, токен действует 24 часа.')

    def _install_binary(self):
        if self.binary.is_file() and hashlib.sha256(self.binary.read_bytes()).hexdigest() == self.digest:
            return
        url = (f'https://github.com/containers/kubernetes-mcp-server/releases/download/v{self.version}/'
               'kubernetes-mcp-server-darwin-arm64')
        with urlopen(url, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != self.digest:
            raise ValueError('MCP binary checksum mismatch')
        self.binary.parent.mkdir(parents=True, exist_ok=True)
        self.binary.write_bytes(content)
        self.binary.chmod(0o700)

    def _write_credentials(self):
        config = yaml.safe_load(self.kubeconfig.read_text())
        context = next(item['context'] for item in config['contexts'] if item['name'] == 'k3d-agent-lab')
        cluster = next(item for item in config['clusters'] if item['name'] == context['cluster'])
        if not cluster['cluster']['server'].startswith('https://127.0.0.1:'):
            raise ValueError('Only the local lab cluster is allowed')
        token = self._create_token()
        output = {'apiVersion': 'v1', 'kind': 'Config', 'clusters': [cluster],
                  'users': [{'name': 'agent', 'user': {'token': token}}],
                  'contexts': [{'name': 'k3d-agent-lab', 'context': {
                      'cluster': context['cluster'], 'user': 'agent', 'namespace': 'agent-lab'}}],
                  'current-context': 'k3d-agent-lab'}
        destination = self.root / 'data/cluster/agent-kubeconfig.yaml'
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(output, stream)
        destination.chmod(0o600)

    def _create_token(self):
        result = subprocess.run(
            ['kubectl', '--kubeconfig', str(self.kubeconfig), '--context', 'k3d-agent-lab',
             '--namespace', 'agent-lab', '--request-timeout=10s', 'create', 'token', 'agent', '--duration=24h'],
            capture_output=True, text=True, check=True, timeout=15)
        return result.stdout.strip()


if __name__ == '__main__':
    MCPSetup().run()
