# Langfuse

## Развёртывание

Команды выполняются из lab3 после установки requirements.txt:

```bash
python scripts/prepare_langfuse.py
docker compose up -d
```

Скрипт создаёт локальный .env с независимыми случайными секретами. Повторный запуск
сохраняет существующую конфигурацию. Файл исключён из Git и доступен только владельцу.
Шаблон имён переменных находится в [.env.example](../.env.example).

Интерфейс: [http://localhost:3000](http://localhost:3000). Для входа используются
LANGFUSE_ADMIN_EMAIL и LANGFUSE_ADMIN_PASSWORD из .env. При первом запуске Langfuse
автоматически создаёт организацию Agent Lab, проект Kubernetes Agent и API-ключи.
Публичный и секретный ключи проекта — LANGFUSE_PUBLIC_KEY и LANGFUSE_SECRET_KEY.
Адрес для SDK задаётся через LANGFUSE_BASE_URL; команды загружают .env через python-dotenv.

Compose содержит шесть сервисов: web, worker, PostgreSQL, ClickHouse, Redis и MinIO.
Проверенная версия Langfuse — 4.76.0; образы web и worker закреплены по digest.
PostgreSQL хранит настройки и пользователей, ClickHouse — наблюдения,
Redis обслуживает очередь, MinIO — объекты событий и медиа. Worker обрабатывает
поступающие данные. На хост опубликованы только localhost:3000 и localhost:9090.
Данные сервисов сохраняются в отдельных Docker volumes.

```bash
docker compose ps
docker compose logs --tail=50 langfuse-web langfuse-worker
docker compose stop
docker compose start
```

Команда stop сохраняет проект и трейсы. При удалении volumes потребуется повторная
инициализация. Изменение ключей в .env после создания проекта не заменяет уже
выданные ключи в базе; новые ключи следует создавать в настройках проекта.

## Диалог с трассировкой

При работающих Ollama, Kubernetes MCP и Qdrant из lab3:

```bash
PYTHONPATH=src python -m benchmark.dialogue
```

Команда запускает собственного агента и выводит ссылку на трейс после ответа.
Разметка находится в src/benchmark/own_agent.py: узлы графа, вызовы модели,
аргументы и результаты инструментов, сжатие, retrieval и причины остановки.
Базовый CLI src/agent остаётся независимым от Langfuse.
OpenCode логируется через src/opencode_agent/gateway.py и tracing.py.

## Сохранённые примеры

Проверенный запуск:

- [Вызов GLM](http://localhost:3000/project/kubernetes-agent/traces/1e0920562a1d2fb5e8f8e272a8e1bb7e):
  2 наблюдения, 34 входных и 54 выходных токена.
- [Агент: Kubernetes и retrieval](http://localhost:3000/project/kubernetes-agent/traces/64c36ac2aa8996dfac4bcce8c4cb6c74):
  28 наблюдений, 3 вызова модели, resources_get и search_docs; ошибок нет,
  последний вход — 3790 токенов при лимите 8192.

Ссылки действуют на этом локальном экземпляре при сохранении Docker volumes.

## Сравнение агентов

В Tracing выберите тег comparison-18, затем own-agent или ready-agent.
Корневые наблюдения пользовательских ходов называются own-agent.turn и opencode.turn.
Sessions объединяет реплики одного диалога; metadata содержит task_id и run_id.
Примеры и диагностические прогоны имеют другие теги и в итоговое сравнение не входят.

Внутри собственного трейса context.compress и context.memory показывают сжатие
и его результат; retrieval — найденные документы и scores. У OpenCode ollama.chat
содержит реальный запрос и ответ, а дочерние spans — инструменты соответствующего шага.

В Scores основной task_success после финализации содержит проверенный результат.
task_success_automatic сохраняет исходную оценку GLM, task_success_reviewed —
результат разбора по трассе. Итог задачи целиком находится в evaluation.task.own/opencode,
агрегат полного прогона — в evaluation.final.own/opencode. Эти служебные evaluator
не включаются в токены и время выполнения агента.

Развёртывание основано на [официальном Docker Compose](https://langfuse.com/self-hosting/deployment/docker-compose).
Автоматическое создание проекта использует
[headless initialization](https://langfuse.com/self-hosting/administration/headless-initialization).
Передача session_id выполняется через
[attribute propagation](https://langfuse.com/docs/observability/features/sessions).
