import json

import yaml


class ResourceSummary:
    def format(self, name: str, text: str) -> str:
        if name != 'resources_get':
            return text
        data = yaml.safe_load(text)
        if not isinstance(data, dict) or 'kind' not in data:
            return text
        metadata = self._metadata(data.get('metadata', {}))
        ordered = {'kind': data['kind'], 'metadata': metadata, 'status': data.get('status', {})}
        ordered.update({key: value for key, value in data.items() if key not in ordered})
        return json.dumps(ordered, ensure_ascii=False, default=str, separators=(',', ':'))

    @staticmethod
    def _metadata(metadata):
        result = {key: value for key, value in metadata.items() if key != 'managedFields'}
        result['annotations'] = {key: value for key, value in result.get('annotations', {}).items()
                                 if key != 'kubectl.kubernetes.io/last-applied-configuration'}
        return result
