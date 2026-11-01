from os.path import relpath
from pathlib import Path

def write_configuration_report(report, output):
    dataset = Path(__file__).resolve().parents[2] / "evaluation/dataset"
    questions_md = Path(relpath(dataset / "questions.md", output.resolve())).as_posix()
    questions_json = Path(relpath(dataset / "questions.json", output.resolve())).as_posix()
    variants = report["variants"]
    first = variants[0]
    embedding = first["configuration"]["embedding"]
    splitter = first["configuration"]["splitter"]
    selected = next(v for v in variants if v["name"] == report["recommended"])
    maximum = max(variants, key=lambda v: (v["metrics"]["all"]["recall@10"], v["metrics"]["all"]["mrr@10"]))
    lines = [
        "# Сравнение конфигураций этапов 3–4", "",
        "Основной эксперимент пункта 5: три конфигурации эмбеддингов при фиксированных сплиттере, индексе и способе retrieval.",
        "Меняется только размерность: 1024, 512, 256. Инструкция вопроса, исходные документы, чанки и разметка одинаковые.",
        "20 вопросов: 10 single-hop и 10 multi-hop. Retrieval — dense через Qdrant, без BM25, агента или генерации ответа.", "",
        "## Зафиксированные параметры", "",
        f"- Сплиттер: `{splitter['splitter']}`, стратегия `{splitter['config']['splitter']}`.",
        f"- Размер/overlap: {splitter['config']['chunk_size']}/{splitter['config']['overlap']} токенов; заголовки в метаданных.",
        f"- Токенизатор: `{splitter['tokenizer']}`.",
        f"- Модель: `{embedding['model']}`, ревизия `{embedding['revision']}`, max_length={embedding['max_length']}.",
        "- Векторы: float32, L2 normalization, Cosine; три отдельные коллекции.",
        "- HNSW: M=16, ef_construction=128, ef_search=128, quantization отсутствует.",
        "- full_scan_threshold=10 KiB, indexing_threshold=1 KiB, default_segment_number=1.",
        "- Перед оценкой во всех коллекциях проверено indexed_vectors_count == points_count == 2950 и status=green.",
        f"- Qdrant {report['qdrant_version']}; устройство эмбеддера {report['device']}.",
        f"- Загрузка модели: {report['query_model_load_seconds']:.3f} с; общий forward 20 вопросов: {report['query_forward_seconds_shared']:.3f} с.",
        "- Инструкция вопроса:", "", "```text", embedding["query_prefix"] + "<вопрос>", "```", "",
        "Qwen поддерживает MRL-размерности 32–1024: [карточка модели](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).",
        "512/256 получены из первых d координат сохранённых 1024-мерных векторов с повторной L2-нормализацией.",
        "Аналогично преобразованы векторы вопросов. Повторный проход корпуса через трансформер не нужен.",
        "Полный forward модели одинаков для всех размерностей; уменьшение размерности экономит хранение и работу индекса, а не вычисления трансформера.", "",
        "## Качество: макросредние по 20 вопросам", "",
        "| Конфигурация | Recall@5 | Recall@10 | Precision@5 | Precision@10 | MRR@5 | MRR@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    keys = ("recall@5", "recall@10", "precision@5", "precision@10", "mrr@5", "mrr@10")
    for variant in variants:
        values = " | ".join(f"{variant['metrics']['all'][k]:.3f}" for k in keys)
        lines.append(f"| {variant['name']} | {values} |")
    lines.extend([
        "", "## Multi-hop: 10 вопросов", "",
        "| Конфигурация | Recall@10 | MRR@10 | Покрытие свидетельств@10 | Полное покрытие@10 | ANN/exact overlap@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for variant in variants:
        metrics = variant["metrics"]["multi_hop"]
        keys = ("recall@10", "mrr@10", "evidence_coverage@10", "complete_evidence@10")
        values = " | ".join(f"{metrics[k]:.3f}" for k in keys)
        lines.append(f"| {variant['name']} | {values} | {variant['ann_vs_exact_overlap_at_10']:.3f} |")
    lines.extend([
        "", "ANN/exact overlap — средняя доля совпавших top-10 по всем 20 вопросам, сравнение с exact-поиском той же размерности.",
        "Полное покрытие multi-hop требует хотя бы одного чанка для каждого обязательного свидетельства.", "",
        "## Затраты", "",
        "| Конфигурация | Чистые float32-векторы, MiB | Относительно 1024 | Загрузка + готовность HNSW, с | Поиск p50, мс | Поиск p95, мс |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for variant in variants:
        cost = variant["cost"]
        ratio = cost["raw_vector_bytes"] / first["cost"]["raw_vector_bytes"]
        lines.append(f"| {variant['name']} | {cost['raw_vector_bytes'] / 2**20:.3f} | {ratio:.2f} | "
                     f"{cost['collection_setup_seconds']:.3f} | {variant['latency']['p50_ms']:.3f} | {variant['latency']['p95_ms']:.3f} |")
    lines.extend([
        "", "Размер в таблице рассчитан как N × dimension × 4 байта. Это объём самих векторов, не суммарная RAM и не размер коллекции на диске:",
        "payload, HNSW, WAL и служебные файлы сюда не входят. Размер NPY и полные конфигурации также сохранены в JSON.",
        f"Латентность: один прогрев всех вопросов, затем {report['latency_repetitions']} повторений × 20 вопросов на конфигурацию; порядок конфигураций чередуется.",
        "Измеряется клиентское время запроса через localhost с получением top-10 ID, без forward эмбеддера.",
        "Сохранены все измерения; статистика p50/p95 локальная и не заменяет нагрузочный тест.",
        "Время загрузки и готовности индекса включает upsert и ожидание оптимизатора; это один запуск, не среднее по пересборкам.", "",
        "## Выбор по качеству и затратам", "",
        "Критерий задан до прогона: выбрать минимальный объём векторов среди конфигураций, у которых относительно 1024",
        "падение Recall@10 и MRR@10 не превышает 0.02, а полное покрытие multi-hop@10 не снижается.",
        "0.02 — практический допуск, не статистическая граница значимости.", "",
        f"**Рекомендуемая конфигурация: `{selected['name']}`.**",
        f"Максимальный Recall@10 (при равенстве — MRR@10): `{maximum['name']}`.",
        f"Выбранная: Recall@10={selected['metrics']['all']['recall@10']:.3f}, MRR@10={selected['metrics']['all']['mrr@10']:.3f}, "
        f"полное покрытие multi-hop@10={selected['metrics']['multi_hop']['complete_evidence@10']:.3f}.",
        f"Объём её векторов составляет {selected['cost']['raw_vector_bytes'] / first['cost']['raw_vector_bytes']:.2f} от 1024-мерного варианта.",
        "Это выбор внутри проверенных трёх конфигураций. Размер чанка, модель и другие параметры индекса здесь не варьировались.", "",
        "Оценка основана на 20 вопросах; перенос вывода на другие вопросы требует дополнительного набора.",
        "Precision учитывает известные положительные метки; прочие чанки unjudged. MRR@k обрезан на k.",
        "Настройки Qdrant с порогами подтверждены [конфигурацией сервера 1.19.2](https://github.com/qdrant/qdrant/blob/v1.19.2/config/config.yaml).", "",
        "## Артефакты", "",
        "- [results.json](results.json): конфигурации, хеши, версии пакетов, измерения, результаты каждого вопроса и top-10.",
        "- configuration_1024.json, configuration_512.json, configuration_256.json: отдельные результаты и полные конфигурации.",
        "- queries_1024.npy, queries_512.npy, queries_256.npy: векторы оценочных запросов.",
        f"- [Вопросы]({questions_md}), [исходная разметка]({questions_json}).", "",
    ])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
