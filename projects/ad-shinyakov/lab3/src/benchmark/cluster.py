import json
import subprocess
from pathlib import Path

import yaml


LAB_ROOT = Path(__file__).resolve().parents[2]


class LocalCluster:
    def __init__(self):
        self.kubeconfig = LAB_ROOT / 'data/cluster/kubeconfig.yaml'
        configuration = yaml.safe_load(self.kubeconfig.read_text())
        if configuration['current-context'] != 'k3d-agent-lab':
            raise ValueError('Для эксперимента требуется изолированный k3d-agent-lab')
        if any(not entry['cluster']['server'].startswith('https://127.0.0.1:')
               for entry in configuration['clusters']):
            raise ValueError('Эксперимент разрешён только на loopback-кластере')

    def run(self, *arguments, value=None, timeout=30):
        command = ['kubectl', '--kubeconfig', str(self.kubeconfig), '--context', 'k3d-agent-lab',
                   '--request-timeout=10s', '-n', 'agent-lab', *arguments]
        result = subprocess.run(command, input=value, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        return result.stdout

    def get(self, kind, name=None, selector=None):
        arguments = ['get', kind]
        if name:
            arguments.append(name)
        if selector:
            arguments.extend(['-l', selector])
        return json.loads(self.run(*arguments, '-o', 'json'))

    def apply(self, resource):
        self.run('apply', '-f', '-', value=json.dumps(resource))

    def patch(self, kind, name, patch):
        self.run('patch', kind, name, '--type=merge', '-p', json.dumps(patch))

    def reset(self):
        self.run('delete', 'deployment,pod,service', '--all', '--wait=true', '--timeout=60s', timeout=75)
        self.run('delete', 'events', '--all', '--wait=false')
        maps = self.get('configmap')['items']
        for resource in maps:
            if resource['metadata']['name'] != 'kube-root-ca.crt':
                self.run('delete', 'configmap', resource['metadata']['name'])

    def ready(self, name):
        self.run('rollout', 'status', f'deployment/{name}', '--timeout=90s', timeout=105)

    def logs(self, name):
        pods = self.get('pods', selector=f'app={name}')['items']
        return {pod['metadata']['name']: self.run('logs', pod['metadata']['name'], '--tail=30')
                for pod in pods if pod['status'].get('phase') == 'Running'}

    def snapshot(self):
        deployments = self.get('deployment')['items']
        pods = self.get('pod')['items']
        return {resource['metadata']['name']: {
            'spec': resource['spec'], 'generation': resource['metadata']['generation'],
            'status': resource.get('status', {}),
            'pod_uids': [pod['metadata']['uid'] for pod in pods
                         if pod['metadata'].get('labels', {}).get('app') == resource['metadata']['name']],
        } for resource in deployments}
