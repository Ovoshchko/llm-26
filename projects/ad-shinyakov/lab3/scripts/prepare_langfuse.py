import os
from pathlib import Path
from secrets import token_hex, token_urlsafe


class LangfuseSetup:
    def __init__(self):
        self.destination = Path(__file__).resolve().parents[1] / '.env'

    def run(self):
        if self.destination.exists():
            print(f'Существующая конфигурация сохранена: {self.destination}')
            return
        values = self._values()
        descriptor = os.open(self.destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(''.join(f'{key}={value}\n' for key, value in values.items()))
        print(f'Создана конфигурация: {self.destination}')
        print('Учётные данные интерфейса: LANGFUSE_ADMIN_EMAIL и LANGFUSE_ADMIN_PASSWORD в .env')

    @staticmethod
    def _values():
        return {
            'LANGFUSE_BASE_URL': 'http://localhost:3000',
            'LANGFUSE_PUBLIC_KEY': f'pk-lf-{token_hex(16)}',
            'LANGFUSE_SECRET_KEY': f'sk-lf-{token_hex(32)}',
            'LANGFUSE_ADMIN_EMAIL': 'admin@agent-lab.local',
            'LANGFUSE_ADMIN_PASSWORD': token_urlsafe(24),
            'NEXTAUTH_SECRET': token_urlsafe(32),
            'SALT': token_urlsafe(32),
            'ENCRYPTION_KEY': token_hex(32),
            'POSTGRES_PASSWORD': token_hex(24),
            'CLICKHOUSE_PASSWORD': token_hex(24),
            'REDIS_AUTH': token_hex(24),
            'MINIO_ROOT_PASSWORD': token_hex(24),
        }


if __name__ == '__main__':
    LangfuseSetup().run()
