import json
import os
from hashlib import sha256

import httpx


class NativeToolTracing:
    def record(self, trace_id, parent_id, events):
        spans = [self._span(trace_id, parent_id, event) for event in events
                 if event['type'] == 'tool_use' and not event['part']['tool'].startswith('lab_')]
        if not spans:
            return
        endpoint = os.environ.get('LANGFUSE_BASE_URL', 'http://localhost:3000').rstrip('/')
        payload = {'resourceSpans': [{'scopeSpans': [{'scope': {'name': 'lab3.opencode'}, 'spans': spans}]}]}
        response = httpx.post(f'{endpoint}/api/public/otel/v1/traces', json=payload,
                              auth=(os.environ['LANGFUSE_PUBLIC_KEY'], os.environ['LANGFUSE_SECRET_KEY']),
                              headers={'x-langfuse-ingestion-version': '4'}, timeout=30)
        response.raise_for_status()
        if int(response.json().get('partialSuccess', {}).get('rejectedSpans', 0)):
            raise RuntimeError('Langfuse rejected native tool spans')

    @staticmethod
    def _span(trace_id, parent_id, event):
        part = event['part']
        state = part['state']
        timing = state.get('time', {})
        attributes = {
            'langfuse.observation.type': 'tool',
            'langfuse.observation.input': json.dumps(state.get('input'), ensure_ascii=False),
            'langfuse.observation.output': json.dumps(state.get('output', state.get('error')), ensure_ascii=False),
            'langfuse.observation.level': 'ERROR' if state['status'] == 'error' else 'DEFAULT',
            'langfuse.observation.metadata': json.dumps({'opencode_native': True, 'part_id': part['id']}),
        }
        return {'traceId': trace_id, 'spanId': sha256(part['id'].encode()).hexdigest()[:16],
                'parentSpanId': parent_id, 'name': part['tool'], 'kind': 1,
                'startTimeUnixNano': str(timing.get('start', event['timestamp']) * 1_000_000),
                'endTimeUnixNano': str(timing.get('end', event['timestamp']) * 1_000_000),
                'attributes': [{'key': key, 'value': {'stringValue': value}} for key, value in attributes.items()]}
