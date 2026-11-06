import json
from collections import Counter
from pathlib import Path
from statistics import mean

from .cluster import LAB_ROOT
from .metrics import RunMetrics
from .figure import ComparisonFigure


class ComparisonReport:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        self.outcome_policy = self.manifest.get('evaluation_policy') == 'outcome-v2'

    def write(self):
        runs = {agent: self._tasks(agent) for agent in ('own', 'opencode')}
        summaries = {agent: RunMetrics().calculate(tasks) for agent, tasks in runs.items()}
        for agent in summaries:
            path = self.directory / f'{agent}-summary.json'
            if path.is_file():
                summaries[agent]['trace_url'] = json.loads(path.read_text()).get('trace_url')
        lines = ['# Сравнение диалоговых агентов', '',
                 'Собственный LangGraph сравнивается с OpenCode на одном наборе заданий.',
                 'Модель — одни веса GLM-4 9B Q4_0; различаются оркестрация, промпты и управление историей.',
                 'Инструменты, поиск и исходное состояние стенда общие; сценарий восстанавливается перед каждой задачей.', '',
                 '[Схема LangGraph и стратегия контекста](../docs/agent.md);',
                 '[схема OpenCode](../opencode/README.md); [состав и критерии датасета](README.md).',
                 ('Память собственного агента — детерминированные извлечения сообщений пользователя и инструментов.'
                  if self.outcome_policy else 'Память собственного агента выделяет цель, ограничения, evidence и действия, сохраняя полный текущий ход.'),
                 'Память содержит ссылки на исходные сообщения; полная история хранится отдельно.',
                 'Эксперимент проверяет, насколько модель сохраняет нужные сведения при ограниченном бюджете.', '',
                 self._scope(runs),
                 'Промежуточные ходы длинных диалогов также выполняются и входят в затраты.',
                 ('Успех задачи: последний результат и отсутствие критических нарушений за весь диалог. '
                  'Качество промежуточных ходов учитывается отдельно.' if self.outcome_policy else
                  'Задача успешна, только если успешны все её проверяемые ходы.'),
                 'В таблицах доли указаны от 0 до 1.',
                 *self._methodology(runs), '',
                 '| Агент | Задач | Success | По ходам | Одиночные | Диалоги | История | Среднее время задачи, с | Токены вход / выход |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for agent, summary in summaries.items():
            lines.append(self._row(agent, summary))
        lines.extend(['', '| Агент | Средняя стоимость задачи, USD | Доля задач с ошибкой инструмента или остановкой | Генераций без usage |',
                      '|---|---:|---:|---:|'])
        for agent, summary in summaries.items():
            lines.append(f'| {agent} | {self._number(summary["mean_task_cost_usd"])} | '
                         f'{self._number(summary["error_rate"])} | {summary["generations_without_usage"]} |')
        lines.extend(self._details(runs))
        lines.extend(['', '![Результаты задач](results/task-success.png)', ''])
        for agent, tasks in runs.items():
            lines.extend(self._breakdown(agent, tasks))
            lines.extend(self._failures(agent, tasks))
        lines.extend(self._conclusions(summaries, runs))
        destination = LAB_ROOT / 'evaluation/comparison.md'
        destination.write_text('\n'.join(lines) + '\n')
        results = LAB_ROOT / 'evaluation/results'
        results.mkdir(exist_ok=True)
        ComparisonFigure().write(runs, results / 'task-success.png')
        for agent, summary in summaries.items():
            (results / f'{agent}-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        (results / 'manifest.json').write_text((self.directory / 'manifest.json').read_text())
        return destination

    def _scope(self, runs):
        values = []
        for agent, tasks in runs.items():
            turns = [turn for task in tasks for turn in task.get('turns', [])]
            checked = sum(bool(turn.get('evaluation')) for turn in turns)
            values.append(f'{agent}: {len(tasks)} задач, {len(turns)} ходов, {checked} проверяемых ответов')
        return f'Прогон `{self.directory.name}`. ' + '; '.join(values) + '.'

    @staticmethod
    def _methodology(runs):
        checked = [turn for tasks in runs.values() for task in tasks for turn in task.get('turns', [])
                   if turn.get('evaluation')]
        reviewed = [turn for turn in checked if turn['evaluation'].get('method') == 'trace_review']
        changed = sum(turn['evaluation']['success'] != turn['automatic_evaluation']['success'] for turn in reviewed)
        return ['Первичный судья — GLM с JSON Schema; состояние проверяется независимо через Kubernetes.',
                f'По трассам перепроверено {len(reviewed)} из {len(checked)} ответов; исправлено {changed} вердиктов success.',
                'Разметка находится в [reviewed_scores.json](reviewed_scores.json), исходные оценки — в',
                '[results/tasks.json](results/tasks.json). Это разбор тем же ассистентом, а не независимая экспертная разметка.']

    def _details(self, runs):
        lines = ['', '## Конфигурация и затраты', '',
                 'Корпус lab2: 2950 чанков Markdown по 800 токенов с overlap=100, без заголовков в тексте.',
                 'Эмбеддинги Qwen3-Embedding-0.6B, 1024 измерения, нормализация; Qdrant cosine,',
                 'HNSW M=16, ef_construction=128, ef_search=128. Поиск: top_k=3, порог 0.5.',
                 f'Окно {self.manifest["model_window"]}, выход {self.manifest["output_budget"]}, '
                 f'максимум {self.manifest["max_steps"]} шагов, таймаут {self.manifest["timeout_seconds"]} с.', '',
                 'Бюджет собственного агента — 8192 токена с сохранением памяти.',
                 f'OpenCode: input=8192, context={8192 + self.manifest["output_budget"]}, '
                 f'reserved={self.manifest["output_budget"]}; auto-compaction и prune штатные.',
                 'Compaction расходует общий штатный бюджет шагов OpenCode; поле steps считает основные генерации.',
                 ('Подсчёт и момент сжатия различаются. Веса одинаковы; примеры tool calling в шаблонах адаптированы к именам инструментов.'
                  if self.outcome_policy else 'Подсчёт и момент сжатия у фреймворков различаются. Алиасы моделей используют одинаковые веса и шаблон;'),
                 'снимки параметров и версии пакетов находятся в [environment.json](results/environment.json).', '',
                 '| Агент | Tool correctness | Вызовов инструментов на ход | Вызовов LLM на ход | Макс. вход модели | Recall@3 | Precision@3 | MRR |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|']
        for agent, tasks in runs.items():
            turns = [turn for task in tasks for turn in task.get('turns', []) if turn.get('evaluation')]
            averages = [self._average(turns, key) for key in ('tool_correctness', 'tool_calls', 'generations')]
            maximum = max((turn['metrics']['max_input_tokens'] for task in tasks
                           for turn in task.get('turns', [])), default=0)
            retrieval = [self._average(turns, key) for key in ('retrieval_recall_at_3', 'retrieval_precision_at_3', 'retrieval_mrr')]
            lines.append('| ' + ' | '.join([agent, *(self._number(value) for value in averages), str(maximum),
                                           *(self._number(value) for value in retrieval)]) + ' |')
        lines.extend(['', 'Вызовы инструментов и LLM усреднены по проверяемым ходам.',
                      'Максимум входа берётся по всем ходам и учитывает также сжатие, которому доступно полное окно модели.', '',
                      '## Результаты задач', '', '| Задача | Свой агент | OpenCode |', '|---|---|---|'])
        identifiers = sorted({task['id'] for tasks in runs.values() for task in tasks})
        for identifier in identifiers:
            cells = []
            for agent in ('own', 'opencode'):
                task = next((task for task in runs[agent] if task['id'] == identifier), {})
                if not task or task.get('fixture_error'):
                    cells.append('ошибка стенда / нет результата')
                    continue
                turn = next((turn for turn in reversed(task['turns']) if turn.get('evaluation')), task['turns'][-1])
                cells.append(f'[{"успех" if task["success"] else "ошибка"}]({turn.get("trace_url", "")})')
            lines.append(f'| {identifier} | ' + ' | '.join(cells) + ' |')
        return lines

    @staticmethod
    def _average(turns, key):
        values = [turn['metrics'][key] for turn in turns if turn['metrics'].get(key) is not None]
        return mean(values) if values else None

    def _tasks(self, agent):
        return [json.loads(path.read_text()) for path in sorted((self.directory / agent).glob('a??.json'))]

    @staticmethod
    def _number(value):
        return '—' if value is None else f'{value:.3f}'

    def _row(self, agent, summary):
        values = [agent, str(summary['tasks']), *(self._number(summary[key]) for key in (
            'success_rate', 'turn_success_rate', 'single_success_rate', 'dialogue_success_rate', 'history_accuracy', 'mean_task_latency_seconds')),
                  f'{summary["input_tokens"]} / {summary["output_tokens"]}']
        return '| ' + ' | '.join(values) + ' |'

    def _breakdown(self, agent, tasks):
        lines = ['', f'## {agent}: разбивка', '', '| Группа | Задач | Success |', '|---|---:|---:|']
        for field in ('difficulty', 'type'):
            groups = sorted({task[field] for task in tasks if not task.get('fixture_error')})
            for group in groups:
                selected = [task for task in tasks if task.get(field) == group and not task.get('fixture_error')]
                lines.append(f'| {field}={group} | {len(selected)} | {mean(task["success"] for task in selected):.3f} |')
        return lines

    def _failures(self, agent, tasks):
        failures = [(task, turn) for task in tasks for turn in task.get('turns', [])
                    if turn.get('evaluation') and not turn['evaluation']['success']]
        counts = Counter(turn['evaluation']['category'] for _, turn in failures)
        lines = ['', f'## {agent}: ошибки', '',
                 'Категории: ' + ', '.join(f'{name} — {count}' for name, count in counts.most_common()) + '.', '']
        chosen = self._select_failures(failures)
        for task, turn in chosen:
            lines.extend([f'### {task["id"]}: {turn["evaluation"]["category"]}', '',
                          turn['evaluation']['reason'], '',
                          f'Ответ: {turn["answer"][:1000] or "Финальный ответ не сформирован: " + str(turn.get("stop_reason"))}', '',
                          f'[Трейс]({turn.get("trace_url", "")})', ''])
        if len(chosen) < 5:
            lines.append(f'В этом прогоне найдено {len(chosen)} независимых неудачных задач; остальные не добавлялись искусственно.')
        return lines

    @staticmethod
    def _select_failures(failures):
        preferred = ['context_limit', 'history', 'recovery', 'policy', 'tool_arguments', 'reasoning', 'verification', 'tool_selection']
        ordered = sorted(failures, key=lambda item: (not item[1].get('compressed'), item[0]['id']))
        chosen = []
        covered = set()
        for category in preferred:
            match = next((item for item in ordered if item[1]['evaluation']['category'] == category
                          and item[0]['id'] not in covered), None)
            if match:
                chosen.append(match)
                covered.add(match[0]['id'])
            if len(chosen) == 5:
                return chosen
        for item in failures:
            if item[0]['id'] not in covered:
                chosen.append(item)
                covered.add(item[0]['id'])
            if len(chosen) == 5:
                break
        return chosen

    def _conclusions(self, summaries, runs):
        valid = {agent: value for agent, value in summaries.items() if value['success_rate'] is not None}
        best = max(valid, key=lambda agent: valid[agent]['success_rate']) if valid else 'нет полного прогона'
        lines = ['', '## Выводы', '', f'На этом наборе наибольшая доля успешных задач: **{best}**.', '',
                 'Стоимость API локальной Ollama равна нулю. Время оборудования и электроэнергия не оценивались;',
                 'затраты сравниваются по времени выполнения и токенам всех вызовов, включая сжатие и служебные вызовы агента.', '',
                 '| Агент | История до сжатия | История после сжатия | Сжатий успешно / попыток | Входных токенов на успешную задачу | Секунд на успешную задачу |',
                 '|---|---:|---:|---:|---:|---:|']
        for agent, summary in summaries.items():
            successes = sum(task.get('success', False) for task in runs[agent])
            tokens = summary['input_tokens'] / successes if successes else None
            seconds = summary['mean_task_latency_seconds'] * summary['tasks'] / successes if successes else None
            lines.append(f'| {agent} | {self._number(summary["history_before_compression"])} | '
                         f'{self._number(summary["history_after_compression"])} | '
                         f'{summary["compression_successes"]} / {summary["compression_attempts"]} | '
                         f'{self._number(tokens)} | {self._number(seconds)} |')
        lines.extend(['', ('У собственного агента память содержит извлечения пользовательских сообщений и результатов инструментов.'
                          if self.outcome_policy else 'У собственного агента структура памяти хранит цель, ограничения, факты с цитатами и результаты действий.'),
                      'OpenCode использует свой стандартный цикл и механизм compaction. Различия качества анализируются по',
                      'наблюдаемым вызовам и приведённым выше ошибкам, включая восстановление после ошибки аргументов,',
                      'выбор инструмента, сохранение цели и подтверждение изменений.', '',
                      'Независимая проверка состояния может дождаться готовности после ответа агента. Это подтверждает',
                      'результат изменения. Ключевые действия оцениваются по рубрике и реальным вызовам инструментов.',
                      'Retrieval-метрики считаются на уровне документов: Recall@3, Precision@3 и MRR по уникальным',
                      'документам в порядке получения в пределах одного хода. Среднее считается по проверяемым ходам',
                      'с эталонными документами; отсутствие поиска даёт ноль. Значения отражают также решение вызвать поиск,',
                      'а не только качество ранжирования среди выполненных запросов.', '',
                      'Латентность собственного агента включает выполнение внутри процесса диалога; OpenCode дополнительно',
                      'использует постоянный HTTP-сервер и MCP. Входные токены учитывают также сжатие и служебные генерации.',
                      'Оборванные генерации без usage перечислены в агрегатах; их неизвестные токены не оценивались.', '',
                      'Затраты на успешную задачу включают расходы всего прогона, в том числе неудачных задач.',
                      'Для нормирования на один достигнутый процентный пункт success rate эти значения умножаются на N/100,',
                      'где N — число задач агента в прогоне.',
                      'Это отношение затрат к полученному качеству в одном прогоне, а не измеренная цена будущего улучшения.',
                      'Конкретные различия архитектур и сохранения памяти разобраны в [analysis.md](analysis.md).', '',
                      'Приоритет улучшений: обязательная проверка после accepted, сохранение последней версии цели',
                      'при сжатии и восстановление после таймаута через чтение состояния. Изменения агентов во время',
                      'сравнения не выполнялись; оценки не используются для автоматического исправления тестовой нагрузки.'])
        return lines
