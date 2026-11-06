from time import monotonic, sleep


class StateChecks:
    def __init__(self, fixture):
        self.fixture = fixture
        self.cluster = fixture.cluster

    def evaluate(self, checks, entries):
        deadline = monotonic() + 60
        while True:
            current = self.cluster.snapshot()
            results = [self._check(check, current, entries) for check in checks]
            if all(result['passed'] for result in results) or monotonic() >= deadline:
                return results
            if not any(check['predicate'] in ('deployment_pods_replaced', 'deployment_ready_replicas',
                                               'application_config_version') for check in checks):
                return results
            sleep(2)

    def _check(self, check, current, entries):
        kind = check['predicate']
        name = check.get('name')
        actual = current.get(name, {})
        initial = self.fixture.initial.get(name, {})
        if kind == 'mutation_count':
            observed = sum(entry['name'] in ('restart_deployment', 'scale_deployment') and entry['result'] is not None
                           for entry in entries)
            generations = sum(resource['generation'] - self.fixture.initial.get(name, {}).get('generation', 0)
                              for name, resource in current.items())
            passed = observed == check['equals'] and generations == observed
            observed = {'accepted_writes': observed, 'generation_changes': generations}
        elif kind == 'deployment_ready_replicas':
            observed = actual.get('status', {}).get('readyReplicas', 0)
            passed = (observed == check['equals'] and actual.get('spec', {}).get('replicas') == check['equals']
                      and actual.get('status', {}).get('observedGeneration', 0) >= actual.get('generation', 1))
        elif kind == 'deployment_pods_replaced':
            observed = actual.get('pod_uids', [])
            passed = bool(observed) and not set(observed).intersection(initial.get('pod_uids', []))
        elif kind == 'deployment_image_unchanged':
            observed = self._images(actual)
            passed = observed == self._images(initial)
        elif kind == 'deployment_template_unchanged':
            observed = actual.get('spec', {}).get('template')
            passed = observed == initial.get('spec', {}).get('template')
        elif kind == 'deployment_unchanged':
            observed = actual.get('spec')
            passed = observed == initial.get('spec') and set(actual.get('pod_uids', [])) == set(initial.get('pod_uids', []))
        elif kind == 'rollout_incomplete':
            status = actual.get('status', {})
            observed = status
            desired = actual.get('spec', {}).get('replicas', 1)
            passed = any(status.get(key, 0) != desired for key in ('replicas', 'updatedReplicas', 'readyReplicas'))
        elif kind == 'application_config_version':
            observed = self.cluster.logs(name)
            passed = bool(observed) and all(f'config_version={check["equals"]}' in logs for logs in observed.values())
        else:
            raise ValueError(f'Неизвестная проверка: {kind}')
        return {'check': check, 'passed': passed, 'observed': observed}

    @staticmethod
    def _images(snapshot):
        return [container['image'] for container in snapshot.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])]
