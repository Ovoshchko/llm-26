import json
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import Field


ResourceName = Annotated[str, Field(min_length=1, max_length=253, pattern=r'^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$')]
Namespace = Literal['agent-lab']


class DeploymentActions:
    def __init__(self, mcp):
        self.mcp = mcp

    def _get(self, name, namespace):
        return json.loads(self.mcp.call('resources_get', apiVersion='apps/v1', kind='Deployment',
                                        name=name, namespace=namespace))

    def restart(self, name: ResourceName, namespace: Namespace = 'agent-lab') -> dict:
        resource = self._get(name, namespace)
        resource.pop('status', None)
        metadata = resource['spec']['template'].setdefault('metadata', {})
        metadata.setdefault('annotations', {})['kubectl.kubernetes.io/restartedAt'] = datetime.now(timezone.utc).isoformat()
        self.mcp.call('resources_create_or_update', resource=json.dumps(resource))
        return {'namespace': namespace, 'name': name, 'accepted': True, 'requires_rollout_check': True}

    def scale(self, name: ResourceName, replicas: Annotated[int, Field(ge=1, le=5)],
              namespace: Namespace = 'agent-lab') -> dict:
        self.mcp.call('resources_scale', apiVersion='apps/v1', kind='Deployment', name=name,
                      namespace=namespace, scale=replicas)
        return {'namespace': namespace, 'name': name, 'replicas': replicas,
                'accepted': True, 'requires_rollout_check': True}

    def rollout(self, name: ResourceName, namespace: Namespace = 'agent-lab') -> dict:
        resource = self._get(name, namespace)
        status = resource.get('status', {})
        replicas = resource['spec'].get('replicas', 1)
        complete = (status.get('observedGeneration', 0) >= resource['metadata']['generation']
                    and all(status.get(key, 0) == replicas for key in (
                        'replicas', 'updatedReplicas', 'readyReplicas', 'availableReplicas')))
        return {'namespace': namespace, 'name': name, 'complete': complete, 'status': status}
