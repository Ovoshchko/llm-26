import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub.constants import HF_HUB_CACHE


LAB_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = json.loads((LAB_ROOT / 'opencode/opencode.json').read_text())


@dataclass(frozen=True)
class OpenCodeConfig:
    model: str = DEFAULT_SETTINGS['model'].split('/', 1)[1]
    timeout_seconds: int = 600
    max_steps: int = DEFAULT_SETTINGS['agent']['kubernetes']['steps']
    structured_tools: bool = True

    @property
    def runtime(self):
        return LAB_ROOT / 'data/opencode'

    @property
    def binary(self):
        return LAB_ROOT / 'opencode/node_modules/.bin/opencode'

    def prepare(self, base_url='http://localhost:11434/v1'):
        if not self.binary.is_file():
            raise FileNotFoundError('Установи OpenCode: npm ci --prefix opencode')
        self.runtime.mkdir(parents=True, exist_ok=True)
        (self.runtime / 'workspace').mkdir(exist_ok=True)
        path = self.runtime / 'opencode.json'
        path.write_text(json.dumps(self._settings(base_url), ensure_ascii=False, indent=2) + '\n')

    def environment(self):
        values = {key: value for key, value in dotenv_values(LAB_ROOT / '.env').items() if value is not None}
        environment = {**os.environ, **values}
        environment.update({
            'OPENCODE_CONFIG': str(self.runtime / 'opencode.json'),
            'OPENCODE_DISABLE_AUTOUPDATE': 'true',
            'OPENCODE_DISABLE_PROJECT_CONFIG': 'true',
            'PYTHONPATH': str(LAB_ROOT / 'src'),
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'HF_HUB_CACHE': HF_HUB_CACHE,
        })
        for name in ('CONFIG', 'DATA', 'CACHE', 'STATE'):
            environment[f'XDG_{name}_HOME'] = str(self.runtime / name.lower())
        return environment

    def _settings(self, base_url):
        directory = LAB_ROOT / 'opencode'
        settings = json.loads((directory / 'opencode.json').read_text())
        for agent in settings['agent'].values():
            prompt = agent.get('prompt', '')
            if prompt.startswith('{file:') and prompt.endswith('}'):
                agent['prompt'] = (directory / prompt[6:-1]).read_text()
        settings['provider']['ollama']['options']['baseURL'] = base_url
        settings['mcp']['lab']['command'][0] = sys.executable
        original = settings['model'].split('/', 1)[1]
        model = settings['provider']['ollama']['models'].pop(original)
        model['name'] = self.model
        settings['provider']['ollama']['models'][self.model] = model
        settings['model'] = settings['small_model'] = f'ollama/{self.model}'
        settings['agent']['kubernetes']['steps'] = self.max_steps
        return settings
