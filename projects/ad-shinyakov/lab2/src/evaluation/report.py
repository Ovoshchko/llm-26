from os.path import relpath
from pathlib import Path

def write_report(report, output):
    dataset = Path(__file__).resolve().parents[2] / "evaluation/dataset"
    questions_md = Path(relpath(dataset / "questions.md", output.resolve())).as_posix()
    questions_json = Path(relpath(dataset / "questions.json", output.resolve())).as_posix()
    variants = report["variants"]
    best = max(variants, key=lambda v: (v["metrics"]["all"]["recall@10"], v["metrics"]["all"]["mrr@10"]))
    lines = [
        "# Результаты retrieval", "",
        "20 вопросов: 10 single-hop и 10 multi-hop. Русские вопросы к английским документам.",
        "Проверяется поиск чанков; генерация ответов и QA-метрики не запускались.", "",
        "## Конфигурация", "",
        f"- Модель: `{report['embedding_manifest']['identity']['embedding']['model']}`.",
        f"- Ревизия: `{report['embedding_manifest']['identity']['embedding']['revision']}`.",
        "- Чанки: 800 токенов, overlap 100, заголовки в метаданных.",
        f"- Векторов: {report['embedding_manifest']['count']}; размерность: {report['embedding_manifest']['dimension']}; Cosine.",
        f"- Qdrant: {report['qdrant_version']}, коллекция `{report['collection']}`.",
        "- HNSW: M=16, ef_construction=128; запросы ef_search=128, top-10.",
        f"- Устройство: {report['device']}; загрузка модели: {report['model_load_seconds']:.2f} с.",
        f"- indexed_vectors_count на старте: {report['collection_info']['indexed_vectors_count']}.",
        "- Dense: Qwen + Qdrant, текущая инструкция Kubernetes без изменений.",
        "- BM25: rank-bm25 / BM25Okapi, k1=1.5, b=0.75, epsilon=0.25.",
        "- Hybrid: равновесное RRF по top-100 dense и top-100 BM25; constant=60, итоговый top-10.",
        "- Для всех методов одинаковые вопросы, текст чанков и разметка. Заголовки не участвуют в поиске.", "",
        "## Все вопросы: макросредние", "",
        "| Метод | Recall@5 | Recall@10 | Precision@5 | Precision@10 | MRR@5 | MRR@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    keys = ("recall@5", "recall@10", "precision@5", "precision@10", "mrr@5", "mrr@10")
    for variant in variants:
        values = " | ".join(f"{variant['metrics']['all'][k]:.3f}" for k in keys)
        lines.append(f"| {variant['name']} | {values} |")
    lines.extend([
        "", "## Multi-hop отдельно", "",
        "Покрытие — доля найденных групп свидетельств. Полное покрытие требует попадания в каждую группу.", "",
        "| Метод | Recall@10 | MRR@10 | Покрытие@10 | Полное@5 | Полное@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for variant in variants:
        metrics = variant["metrics"]["multi_hop"]
        keys = ("recall@10", "mrr@10", "evidence_coverage@10", "complete_evidence@5", "complete_evidence@10")
        values = " | ".join(f"{metrics[k]:.3f}" for k in keys)
        lines.append(f"| {variant['name']} | {values} |")
    lines.extend([
        "", "## Время", "",
        "| Метод | Эмбеддинг 20 вопросов, с | Поиск 20 вопросов, с |",
        "|---|---:|---:|",
    ])
    for variant in variants:
        lines.append(f"| {variant['name']} | {variant['query_encode_seconds']:.3f} | {variant['search_seconds']:.3f} |")
    lines.extend([
        "", f"Загрузка BM25-индекса: {report['bm25_build_seconds']:.3f} с.",
        "Это один запуск без отдельного прогрева и повторений; времена ориентировочные.",
        "Векторы вопросов и списки кандидатов вычислены один раз и переиспользованы в hybrid.",
        "Стоимость hybrid указана как сумма dense, BM25 и RRF; эмбеддинг вопроса учитывается отдельно.",
        "Измерение dense включает локальный HTTP и передачу ID top-100 без текстов.", "",
        "## Неполное покрытие в варианте с максимальным Recall@10", "",
        "| Вопрос | Тип | Покрытие@10 | Отсутствующие свидетельства |",
        "|---|---|---:|---|",
    ])
    gold = {q["question_id"]: q for q in report["labels"]["judgments"]}
    for row in best["questions"]:
        if row["metrics"]["complete_evidence@10"]:
            continue
        hits = {h["chunk_id"] for h in row["hits"]}
        missing = [g["evidence_id"] for g in gold[row["question_id"]]["evidence_groups"]
                   if not any(m["chunk_id"] in hits for m in g["matches"])]
        lines.append(f"| {row['question_id']} | {row['type']} | {row['metrics']['evidence_coverage@10']:.3f} | {', '.join(missing)} |")
    lines.extend([
        "", "## Вывод", "",
        f"Максимальный Recall@10 у `{best['name']}`; при равенстве используется MRR@10.",
        f"Максимальный MRR@10 у `{max(variants, key=lambda v: v['metrics']['all']['mrr@10'])['name']}`.",
        "Нужно учитывать обе метрики и стоимость: небольшой рост Recall не означает улучшения порядка результатов или покрытия multi-hop.",
        "Числа относятся к этому набору и выбранным параметрам. RRF constant=60 и top-100 заданы до прогона без подбора по ответам.",
        "Для multi-hop следующий полезный эксперимент — разложение вопроса на подзапросы с объединением результатов.",
        "Все три метода здесь используют однократный retrieval, без декомпозиции вопроса и без агента.", "",
        "BM25 получает исходные русские вопросы, документы английские. Перевода и стемминга нет;",
        "совпадения обычно обеспечивают названия API, параметры и другие общие термины.",
        "Это lexical baseline текущего межъязыкового сценария, а не оценка BM25 с переводом вопросов.",
        "20 вопросов недостаточно для уверенного обобщения; это сравнение на одном оценочном наборе.",
        "Precision считает только известные положительные метки: остальные чанки не размечены как отрицательные.",
        "MRR@k обрезан на k: отсутствие релевантного результата в первых k даёт ноль.",
        "Если indexed_vectors_count=0, Qdrant использует поиск без построенного HNSW: этот запуск не измеряет качество приближённого индекса.",
        "Точные инструкции, параметры, хеши, результаты каждого вопроса и тексты top-10 сохранены в [results.json](results.json).",
        f"Вопросы и основания ответов: [questions.md]({questions_md}), [questions.json]({questions_json}).", "",
    ])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
