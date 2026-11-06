# OpenCode

Готовый агент использует собственный цикл OpenCode 1.19.5 и расположен отдельно
от агента LangGraph. Общие инструменты доступны через stdio MCP.

```bash
npm ci --prefix opencode
bash opencode/prepare_model.sh
PYTHONPATH=src python -m opencode_agent "Проверь api в agent-lab. Ничего не меняй."
PYTHONPATH=src python -m opencode_agent --session SESSION_ID "А сколько реплик было?"
```

## Настройки

[opencode.json](opencode.json) содержит модель, primary agent, MCP, права,
лимиты и compaction. [system.md](system.md) задаёт роль и правила действий,
[compaction.md](compaction.md) — сохранение цели и фактов при сокращении истории,
[reply.md](reply.md) — формат ответа на очередную реплику.
Python подставляет адрес прокси, интерпретатор и содержимое prompt-файлов
в рабочую копию конфига в data/opencode. Настройки агента остаются в JSON.

Алиас glm4-opencode:9b использует веса GLM-4 9B и шаблон вызовов с именами lab_*.
Лимиты: 16 шагов, 600 секунд, выход 2048. Окно Ollama — 32768;
input=8192, context=10240, reserved=2048 управляют штатными compaction и prune.
У собственного агента используется детерминированная компрессия без LLM.
Штатная compaction OpenCode тоже расходует его бюджет шагов. Поле steps считает
генерации основного агента, а step_limit_reached учитывает все генерации цикла.

Разрешены MCP-инструменты lab_* и штатный todowrite. Доступ к shell
и файлам запрещён. XDG-каталоги и сессии изолированы в data/opencode.
Режим structured_tools ограничивает формат tool calling JSON Schema для GLM.
Действия исполняет OpenCode; текстовые команды из ответа не исполняются обвязкой.

## Запуск и трассировка

| Файл в src/opencode_agent | Назначение |
|---|---|
| config.py | Загрузка JSON и изолированная среда |
| server.py | Один постоянный HTTP-сервер и MCP-процесс |
| session.py | Сессии и отправка реплик через API |
| gateway.py | Проксирование Ollama и запись генераций в Langfuse |
| mcp.py | Публикация общего контракта инструментов |
| tracing.py | Trace на реплику и вложенность MCP-вызовов |
| native_tracing.py | Штатные инструменты: OTLP с исходными временными метками |

Сравнение использует один сервер на все задачи OpenCode. Каждая задача получает
отдельную сессию; история между её репликами сохраняется. Полная история читается
через API, включая штатную compaction. Ключи Langfuse загружаются из .env.

Трассировка текущего сравнения использует прокси и MCP spans. Для самостоятельного
подключения Langfuse через конфиг доступен официальный
[opencode-observability-plugin](https://github.com/langfuse/opencode-observability-plugin).
Для OpenCode 1.x подключение выглядит так:

```json
{"plugin": ["@langfuse/opencode-observability-plugin@latest"]}
```

Плагину нужны LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY и LANGFUSE_BASE_URL
в окружении. В текущий конфиг сравнения он не включён: сначала необходимо
отключить существующую запись тех же генераций, чтобы избежать дублирования.

Настройка агента описана в [документации OpenCode](https://opencode.ai/docs/agents/).
