# Сравнение retrieval и ответов RAG

В каталоге сохранены оценочный набор и выбранные результаты двух видов сравнений.
Новые локальные запуски попадают в игнорируемый `local_runs/`.

```text
evaluation/
├── dataset/                         вопросы, эталонные ответы и разметка
├── retrieval/
│   ├── embedding_dimensions/        1024 / 512 / 256
│   ├── search_methods/              dense / BM25 / hybrid RRF
│   ├── query_instructions/          Kubernetes / общая / без инструкции
│   └── score_threshold.json         проверка порога 0.5
└── rag/
    ├── single_pass/                 однопроходный RAG
    ├── answer_review/               повторная проверка финального ответа
    ├── planned_multihop/            поиск по плану подзадач
    ├── evidence_review/             отдельная проверка достаточности цитат
    └── tuned_multihop/              ответы после изменения промптов
```

## Retrieval

[Набор](dataset/README.md): 20 русских вопросов к английской документации Kubernetes,
10 single-hop и 10 multi-hop. Разметка фиксирует необходимые цитаты и релевантные чанки.

| Сравнение | Отчёт | Основной результат |
|---|---|---|
| Размерности эмбеддингов | [embedding_dimensions](retrieval/embedding_dimensions/report.md) | Выбрана 1024 по допуску потери Recall/MRR и полному покрытию multi-hop |
| Методы поиска | [search_methods](retrieval/search_methods/report.md) | Hybrid выше по Recall@10, dense выше по MRR@10 |
| Инструкции запросов | [query_instructions](retrieval/query_instructions/report.md) | Лучше инструкция про Kubernetes |
| Порог близости | [score_threshold.json](retrieval/score_threshold.json) | Порог 0.5 не изменил метрики на этих 20 вопросах |

Для каждого сравнения сохранены report.md и исходный results.json.
В embedding_dimensions также сохранены конфигурации каждого варианта.
Воспроизводимые бинарные векторы запросов NPY не входят в Git; их хеши в JSON
сохраняют сведения об исходном запуске. Пути внутри исторических JSON относятся
к моменту измерения; сами результаты не переписывались.

Основной эксперимент пункта 5 — embedding_dimensions. Чанки 800/100, модель Qwen
и HNSW фиксированы; сравнения размеров чанков или разных моделей здесь нет.

## Ответы RAG

[Сводное сравнение](rag/README.md) содержит результаты, ограничения и ссылки на
полные ответы. Сохранены и ошибки, и отказы. Это диагностические сценарии,
а не полная оценка QA на 20 вопросах. Модель генерации — glm4:9b во всех вариантах.

## Повторить измерения

Команды выполняются из `projects/ad-shinyakov/lab2` при запущенном Qdrant.
Корпус, чанки, эмбеддинги и содержимое Qdrant пересоздаются по основному README;
они хранятся локально и исключены из Git.

```bash
python scripts/compare_configurations.py \
  --embeddings data/embeddings/ced08184c9da0803 \
  --chunks data/chunks/3cccc393ec5ee49d

python scripts/evaluate_retrieval.py \
  --embeddings data/embeddings/ced08184c9da0803

python scripts/smoke_rag.py \
  --embeddings data/embeddings/72ae1550489b20f9
```

Последняя команда обращается к запущенному RAG API. Все три скрипта сохраняют
новые результаты в evaluation/local_runs; `--output` позволяет указать другой путь.
Сохранённые варианты в retrieval и rag автоматически не перезаписываются.
