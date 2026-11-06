import json
from copy import deepcopy
from time import monotonic, sleep

import yaml

from .cluster import LAB_ROOT, LocalCluster


class ScenarioFixture:
    image = 'agent-lab/evaluation:dev'

    def __init__(self, cluster=None):
        self.cluster = cluster or LocalCluster()
        self.scenarios = json.loads((LAB_ROOT / 'evaluation/scenarios.json').read_text())['scenarios']
        documents = list(yaml.safe_load_all((LAB_ROOT / 'infra/workload.yaml').read_text()))
        self.template = next(resource for resource in documents if resource['kind'] == 'Deployment')
        self.initial = {}
        self.scenario = None

    def setup(self, name):
        self.scenario = name
        self.cluster.reset()
        if name == 'pending_cpu':
            self._reserve_cpu()
        specification = self.scenarios[name]
        for deployment in specification['deployments']:
            observations = specification['initial_observations'][deployment]
            self._deploy(deployment, observations)
        self._special_setup(name)
        self._wait_preconditions(name)
        self.initial = self.cluster.snapshot()
        return self.initial

    def _deploy(self, name, observations):
        self.cluster.apply(self._deployment(name, observations))
        self.cluster.apply(self._service(name))

    def _deployment(self, name, observations):
        resource = deepcopy(self.template)
        resource['metadata']['name'] = name
        resource['spec']['replicas'] = observations.get('desired_replicas', 2)
        resource['spec']['selector']['matchLabels'] = {'app': name}
        resource['spec']['template']['metadata'] = {'labels': {'app': name}}
        pod = resource['spec']['template']['spec']
        pod['terminationGracePeriodSeconds'] = 1
        container = self._container(pod, name, observations)
        self._faults(pod, container)
        return resource

    def _container(self, pod, name, observations):
        container = pod['containers'][0]
        container.update(name='app', image=self.image)
        config_name = observations.get('env_from_configmap', f'{name}-config')
        self._config(config_name, 'v1')
        container['envFrom'] = [{'configMapRef': {'name': config_name}}]
        if observations.get('previous_logs'):
            container['env'] = [{'name': 'REQUIRE_ENDPOINT', 'value': 'true'}]
        if observations.get('readiness_path'):
            container['readinessProbe']['httpGet']['path'] = observations['readiness_path']
        return container

    def _faults(self, pod, container):
        if self.scenario == 'pending_cpu':
            container['resources']['requests']['cpu'] = '2000m'
            container['resources']['limits']['cpu'] = '2000m'
        if self.scenario == 'init_wait':
            pod['initContainers'] = [{'name': 'wait-config', 'image': self.image,
                                     'imagePullPolicy': 'Never', 'command': ['python', '-u', '-c',
                                     'import time\nwhile True:\n print("Waiting for required config file", flush=True)\n time.sleep(3)']}]

    @staticmethod
    def _service(name):
        return {'apiVersion': 'v1', 'kind': 'Service', 'metadata': {'name': name, 'namespace': 'agent-lab'},
                'spec': {'selector': {'app': name}, 'ports': [{'port': 80, 'targetPort': 8080}]}}

    def _config(self, name, version, ready='true'):
        self.cluster.apply({'apiVersion': 'v1', 'kind': 'ConfigMap',
                            'metadata': {'name': name, 'namespace': 'agent-lab'},
                            'data': {'CONFIG_VERSION': version, 'ALLOW_READY': ready}})

    def _special_setup(self, name):
        if name == 'service_selector':
            self.cluster.apply({'apiVersion': 'v1', 'kind': 'Service',
                                'metadata': {'name': 'web-service', 'namespace': 'agent-lab'},
                                'spec': {'selector': {'app': 'other'}, 'ports': [{'port': 80, 'targetPort': 8080}]}})

    def _reserve_cpu(self):
        node = self.cluster.get('node')['items'][0]
        capacity = self._millicpu(node['status']['allocatable']['cpu'])
        pods = json.loads(self.cluster.run('get', 'pods', '-A', '-o', 'json'))['items']
        used = sum(self._millicpu(container.get('resources', {}).get('requests', {}).get('cpu', '0'))
                   for pod in pods if pod.get('spec', {}).get('nodeName')
                   for container in pod['spec']['containers'])
        request = capacity - used - 500
        if request <= 0:
            raise RuntimeError('Недостаточно CPU для детерминированного сценария Pending')
        self.cluster.apply({'apiVersion': 'v1', 'kind': 'Pod',
                            'metadata': {'name': 'cpu-reservation', 'namespace': 'agent-lab'},
                            'spec': {'containers': [{'name': 'reserve', 'image': self.image,
                                      'imagePullPolicy': 'Never', 'resources': {'requests': {'cpu': f'{request}m'}}}]}})

    @staticmethod
    def _millicpu(value):
        return int(value[:-1]) if value.endswith('m') else int(float(value) * 1000)

    def _wait_preconditions(self, name):
        deployments = self.scenarios[name]['deployments']
        healthy = [deployment for deployment in deployments
                   if self.scenarios[name]['initial_observations'][deployment].get('ready_replicas', 2) > 0]
        for deployment in healthy:
            self.cluster.ready(deployment)
        if name == 'config_env_stale':
            self._config('api-config', 'v2')
        if name == 'blocked_rollout':
            self._config('api-config', 'v1', ready='false')
        if len(healthy) != len(deployments):
            self._wait_broken(name)

    def _wait_broken(self, name):
        deadline = monotonic() + 60
        while monotonic() < deadline:
            pods = self.cluster.get('pods')['items']
            if self._broken_ready(name, pods):
                return
            sleep(1)
        raise RuntimeError(f'Не выполнены предусловия сценария {name}')

    @staticmethod
    def _broken_ready(name, pods):
        statuses = [status for pod in pods for status in pod['status'].get('containerStatuses', [])]
        waiting = [status.get('state', {}).get('waiting', {}).get('reason') for status in statuses]
        if name in ('missing_env', 'long_diagnosis'):
            return 'CrashLoopBackOff' in waiting
        if name == 'pending_cpu':
            return any('Insufficient cpu' in condition.get('message', '')
                       for pod in pods for condition in pod['status'].get('conditions', []))
        if name == 'init_wait':
            return any(status.get('state', {}).get('running') for pod in pods
                       for status in pod['status'].get('initContainerStatuses', []))
        return bool(statuses) and all(status.get('state', {}).get('running') and not status.get('ready')
                                      for status in statuses)

    def before_turn(self, turn):
        if self.scenario == 'pod_replaced' and turn == 2:
            self._config('api-config', 'v2')
            self.cluster.run('delete', 'pod', '-l', 'app=api', '--wait=true', '--timeout=30s', timeout=45)
            self.cluster.ready('api')

    def restore(self):
        self.cluster.reset()
        self.cluster.run('apply', '-f', str(LAB_ROOT / 'infra/workload.yaml'))
        self.cluster.ready('api')
